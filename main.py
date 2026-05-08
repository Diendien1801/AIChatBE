from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from agent.openapi_tool import get_openapi_agent
from typing import Optional, List
import traceback

app = FastAPI(title="AEO AI Assistant Service (No DB)")

# 1. Định nghĩa cấu trúc Lịch sử
class ChatMessage(BaseModel):
    role: str  # 'user' hoặc 'bot'
    content: str

# 2. Cập nhật Request để nhận mảng history
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = [] # Mặc định là mảng rỗng nếu chưa có

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

        # 3. Chuyển mảng history từ Client thành Text cho AI đọc
        history_text = ""
        if req.history:
            history_text = "LỊCH SỬ TRÒ CHUYỆN:\n"
            for msg in req.history:
                role_name = "USER" if msg.role.lower() == 'user' else "BOT"
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
            "7. NGÔN NGỮ: Luôn giao tiếp bằng Tiếng Việt thân thiện, rõ ràng."
        )
        
        # 6. Gọi AI xử lý
        response = agent.invoke({"input": prompt})
        bot_reply = response["output"]

        return {"status": "success", "reply": bot_reply}

    except Exception as e:
        print("\n=== LỖI CHI TIẾT ===")
        traceback.print_exc() 
        print("====================\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)