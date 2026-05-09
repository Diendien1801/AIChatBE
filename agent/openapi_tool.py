import os
import requests # <--- Thêm thư viện này
from dotenv import load_dotenv # <--- Thêm thư viện này
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.utilities.requests import RequestsWrapper
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec

def fix_swagger_refs(data):
    """
    Hàm đệ quy để quét và vá các liên kết ($ref) bị lỗi trong Swagger JSON.
    """
    if isinstance(data, dict):
        # Nếu phát hiện link cụt lủn không có tên Schema
        if "$ref" in data and data["$ref"] == "#/components/schemas/":
            # Xóa $ref lỗi và thay bằng kiểu object cơ bản để LangChain không crash
            return {"type": "object", "description": "Auto-fixed missing schema"}
        
        # Tiếp tục quét sâu vào các node con
        return {k: fix_swagger_refs(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [fix_swagger_refs(item) for item in data]
    return data
# 1. Hàm lọc Swagger thông minh (Giữ nguyên của bạn)
def filter_swagger_smart(raw_spec, target_tags):
    filtered_spec = raw_spec.copy()
    filtered_paths = {}
    
    always_allow_get_tags = ["projects", "brands", "user","project-members","subscriptions","contents","content-agents","customer-personas","content-insights","topics","contents-profiles","Keyword","prompts"] 
    
    for path, methods in raw_spec.get("paths", {}).items():
        new_methods = {}
        for method, details in methods.items():
            tags_of_api = details.get("tags", [])
            
            is_in_target_tags = any(tag in tags_of_api for tag in target_tags)
            is_core_get_api = method.lower() == "get" and any(t in tags_of_api for t in always_allow_get_tags)
            
            if is_in_target_tags or is_core_get_api:
                new_methods[method] = details
        
        if new_methods:
            filtered_paths[path] = new_methods
            
    filtered_spec["paths"] = filtered_paths
    return filtered_spec

# 2. Hàm chính được gọi từ main.py
def get_openapi_agent(access_token: str, user_message: str, history_text: str):
    # ==========================================
    # SỬA Ở ĐÂY: KÉO SWAGGER TỪ URL
    # ==========================================
    load_dotenv() # Load file .env
    
    swagger_url = os.getenv("SWAGGER_JSON_URL")
    if not swagger_url:
        raise ValueError("LỖI: Không tìm thấy SWAGGER_JSON_URL trong file .env")

    # Dùng requests để kéo JSON trực tiếp từ Backend
    response = requests.get(swagger_url)
    response.raise_for_status() 
    raw_spec = response.json()
        
    if "servers" not in raw_spec or not raw_spec["servers"]:
        raw_spec["servers"] = [{"url": "https://api.aeo.how", "description": "Production"}]

    # Khởi tạo model (Bạn đang dùng 3.0-flash-lite, rất tốt để tiết kiệm token)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # ==========================================
    # BƯỚC 1: AI LỄ TÂN PHÂN LUỒNG
    # ==========================================
    available_tags = [t.get("name") for t in raw_spec.get("tags", [])]
    
    router_prompt = f"""
    Danh sách chủ đề hệ thống hỗ trợ: {available_tags}.
    Lịch sử chat và Yêu cầu của người dùng: "{history_text} \n User: {user_message}"
    
    Nhiệm vụ: Phân tích xem yêu cầu này cần gọi API thuộc những chủ đề nào? 
    Chỉ trả về danh sách các chủ đề phù hợp, cách nhau bằng dấu phẩy (Ví dụ: brands, projects). Không giải thích gì thêm.
    """
    
    # --- ĐOẠN CODE MỚI CẦN THAY THẾ BẮT ĐẦU TỪ ĐÂY ---
    ai_msg = llm.invoke(router_prompt)
    raw_content = ai_msg.content
    
    # Kiểm tra và bóc tách text an toàn nếu LangChain trả về List
    if isinstance(raw_content, list):
        response_text = "".join(
            [block.get("text", str(block)) if isinstance(block, dict) else str(block) for block in raw_content]
        )
    else:
        response_text = str(raw_content)
        
    response_text = response_text.strip()
    # --- KẾT THÚC ĐOẠN MỚI ---

    chosen_tags = [tag.strip() for tag in response_text.split(',') if tag.strip() in available_tags]
    
    # ==========================================
    # BƯỚC 2: TẠO AGENT CHUYÊN GIA
    # ==========================================
    headers = {"Authorization": f"Bearer {access_token}"}
    requests_wrapper = RequestsWrapper(headers=headers)
    
    # --- THÊM DÒNG NÀY ĐỂ VÁ LỖI SWAGGER ---
    raw_spec = fix_swagger_refs(raw_spec)
    # --------------------------------------

    api_spec_reduced = reduce_openapi_spec(raw_spec)
    
    agent = planner.create_openapi_agent(
        api_spec=api_spec_reduced,
        requests_wrapper=requests_wrapper,
        llm=llm,
        allow_dangerous_requests=True,
        allowed_operations=("GET", "POST", "PUT", "DELETE", "PATCH"),
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )
    
    return agent