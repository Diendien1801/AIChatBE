from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from agent.openapi_tool import get_openapi_agent
from agent.mongo_chat_history import (
    build_mongo_history,
    messages_to_chat_payload,
    messages_to_history_text,
)
from typing import Optional, List
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AEO AI Assistant Service")

# 1. Định nghĩa cấu trúc Lịch sử
class ChatMessage(BaseModel):
    role: str  # 'user' hoặc 'bot'
    content: str

# 2. Request: history từ client, hoặc session_id để dùng MongoDB (AICHAT1.chat_histories)
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)

@app.get("/assistant/session/{session_id}")
async def get_session(session_id: str):
    """Return chat messages stored in MongoDB for this session."""
    key = (session_id or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="session_id is empty")
    try:
        mongo_history = build_mongo_history(key)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    messages = messages_to_chat_payload(mongo_history.messages)
    return {"status": "success", "session_id": key, "messages": messages}


@app.delete("/assistant/session/{session_id}")
async def delete_session(session_id: str):
    """Remove all MongoDB documents for this session (LangChain clear)."""
    key = (session_id or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="session_id is empty")
    try:
        mongo_history = build_mongo_history(key)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    mongo_history.clear()
    return {"status": "success", "session_id": key, "deleted": True}


@app.post("/assistant/chat")
async def chat_with_agent(
    req: ChatRequest, 
    authorization: Optional[str] = Header(None)
):
    try:
        # Trích xuất Token
        access_token = None
        if authorization and authorization.startswith("Bearer "):
            access_token = authorization.split(" ")[1]

        # 3. Lịch sử: MongoDB (SessionId / History) hoặc mảng history từ client
        history_text = ""
        mongo_history = None
        session_key = (req.session_id or "").strip()
        if session_key:
            try:
                mongo_history = build_mongo_history(session_key)
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            history_text = messages_to_history_text(mongo_history.messages)
        elif req.history:
            history_text = "LỊCH SỬ TRÒ CHUYỆN:\n"
            for msg in req.history:
                role_name = "USER" if msg.role.lower() == "user" else "BOT"
                history_text += f"- {role_name}: {msg.content}\n"
            history_text += "\n"

        # 4. Khởi tạo Agent (Truyền luôn text lịch sử cho Lễ tân)
        agent = get_openapi_agent(access_token, req.message, history_text)
        
        # 5. Chuẩn bị Prompt "Kỷ luật sắt"
        prompt = (
            f"{history_text}"
            f"YÊU CẦU MỚI TỪ USER: '{req.message}'\n\n"
            "HƯỚNG DẪN NGHIÊM NGẶT DÀNH CHO BẠN:\n"
            "1. CẤM TỰ BỊA DỮ LIỆU: Tuyệt đối không gọi API nếu thiếu tham số bắt buộc.\n"
            "2. NGUYÊN TẮC KHÔNG TỰ QUYẾT (ZERO ASSUMPTION): Khi dùng API tra cứu để tìm một đối tượng, dù kết quả trả về CHỈ CÓ 1 ĐỐI TƯỢNG DUY NHẤT, bạn KHÔNG ĐƯỢC phép tự ý sử dụng ID đó mà phải hỏi người dùng để xác nhận.\n"
            "3. ĐẶC QUYỀN API PATCH: Đối với các API cập nhật (PATCH), bạn CHỈ CẦN gửi đúng những trường mà người dùng yêu cầu cập nhật. HÃY BỎ QUA các trường required khác trong tài liệu Swagger nếu người dùng không nhắc tới.\n"
            "4. CẤU TRÚC HỎI THÔNG TIN BẮT BUỘC: Khi thiếu thông tin để gọi API, bạn TUYỆT ĐỐI KHÔNG dùng tên biến kỹ thuật (như 'voiceAndTone'). Bạn BẮT BUỘC phải hỏi người dùng theo đúng cấu trúc gạch đầu dòng sau đây:\n"
            "   - **[Tên thông tin viết bằng ngôn ngữ tự nhiên]** ([Trạng thái: Bắt buộc hoặc Tùy chọn]): [Giải thích ngắn gọn] - Ví dụ: [Đưa ra 1-2 ví dụ cụ thể].\n"
            "   (Ví dụ mẫu: '- **Đối tượng độc giả** (Bắt buộc): Nhóm người sẽ đọc nội dung này - Ví dụ: Sinh viên, Người đi làm...').\n"
            "5. CÁCH HỎI NGƯỢC LẠI USER: BẮT BUỘC dùng cú pháp 'Final Answer: [Câu trả lời/Câu hỏi của bạn]' để thoát vòng lặp và giao tiếp. Không được để trống Action.\n"
            "6. LỖI FORMAT JSON: Khi gọi API, Action Input phải là JSON thô, CẤM bọc trong ```json.\n"
            "7. NGÔN NGỮ: Luôn giao tiếp bằng Tiếng Việt thân thiện, rõ ràng.\n"
            "8. QUY TRÌNH TẠO PROJECT BẮT BUỘC (2 BƯỚC): Backend hiện tại KHÔNG nhận tham số khi khởi tạo Project. Do đó, khi người dùng yêu cầu tạo Project mới kèm theo các thông tin (tên, mô tả...), bạn BẮT BUỘC phải thực hiện chuỗi hành động sau:\n"
            "   - Bước 1 (Tạo nháp): Gọi API POST để tạo Project. TUYỆT ĐỐI KHÔNG truyền các tham số (như name, description...) vào payload của API này để tránh lỗi. Chỉ gọi POST thuần túy.\n"
            "   - Bước 2 (Cập nhật): Trích xuất `id` của Project vừa được tạo từ kết quả của Bước 1. Sau đó, LẬP TỨC tìm và gọi API PATCH (hoặc PUT) để cập nhật các thông tin người dùng đã yêu cầu (name, description...) vào đúng `id` đó.\n"
            "   - CHÚ Ý: Tuyệt đối không trả về Final Answer cho người dùng nếu chưa thực hiện thành công Bước 2."
        )
        
        # 6. Gọi AI xử lý
        response = agent.invoke({"input": prompt})
        bot_reply = response["output"]

        if mongo_history is not None:
            mongo_history.add_user_message(req.message)
            mongo_history.add_ai_message(bot_reply)

        return {"status": "success", "reply": bot_reply}

    except Exception as e:
        print("\n=== LỖI CHI TIẾT ===")
        traceback.print_exc() 
        print("====================\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)