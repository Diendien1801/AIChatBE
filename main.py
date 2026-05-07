from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from agent.openapi_tool import get_openapi_agent
from typing import Optional, List
import traceback

app = FastAPI(title="AEO AI Assistant Service")

# 1. Định nghĩa cấu trúc tin nhắn lịch sử
class ChatMessage(BaseModel):
    role: str # 'user' hoặc 'bot'
    content: str

# 2. Cập nhật Request để nhận thêm history từ Mobile
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = [] # Mặc định là mảng rỗng nếu mới chat

@app.post("/assistant/chat")
async def chat_with_agent(
    req: ChatRequest, 
    authorization: Optional[str] = Header(None)
):
    try:
        access_token = None
        if authorization and authorization.startswith("Bearer "):
            access_token = authorization.split(" ")[1]

        agent = get_openapi_agent(access_token)

        # 3. Gom lịch sử chat thành một chuỗi văn bản cho AI đọc
        history_text = ""
        if req.history:
            history_text = "LỊCH SỬ TRÒ CHUYỆN:\n"
            for msg in req.history:
                history_text += f"- {msg.role.upper()}: {msg.content}\n"
            history_text += "\n"

        # 4. Viết lại Prompt "Kỷ luật sắt"
        prompt = (
            f"{history_text}"
            f"YÊU CẦU MỚI TỪ USER: '{req.message}'\n\n"
            "HƯỚNG DẪN NGHIÊM NGẶT DÀNH CHO BẠN:\n"
            "1. ĐỌC KỸ TÀI LIỆU API: Trước khi quyết định dùng công cụ gọi API, hãy xem API đó yêu cầu những tham số bắt buộc (required parameters/body) nào.\n"
            "2. CẤM TỰ BỊA DỮ LIỆU: Nếu Yêu cầu của User VÀ Lịch sử trò chuyện chưa có đủ thông tin cho các tham số bắt buộc, TUYỆT ĐỐI KHÔNG ĐƯỢC gọi API. Thay vào đó, hãy trả lời bằng cách HỎI NGƯỢC LẠI user để xin các thông tin còn thiếu.\n"
            "3. CHỈ GỌI API KHI ĐÃ ĐỦ DATA: Khi user đã cung cấp đủ thông tin, hãy gọi API để thực thi.\n"
            "4. LỖI FORMAT: Cấm bọc Action Input trong các dấu backtick (```json). Hãy viết dạng JSON thô trên 1 dòng.\n"
            "5. NGÔN NGỮ: Luôn giao tiếp với user bằng Tiếng Việt thân thiện."
        )
        
        response = agent.invoke({"input": prompt})

        return {"status": "success", "reply": response["output"]}

    except Exception as e:
        print("\n=== LỖI CHI TIẾT ===")
        traceback.print_exc() 
        print("====================\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)