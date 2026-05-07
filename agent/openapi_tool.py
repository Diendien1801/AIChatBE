import os
import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.utilities.requests import RequestsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def sanitize_spec(data):
    """
    Hàm đệ quy để dọn dẹp các tham chiếu ($ref) bị lỗi do Swagger sinh ra.
    Xóa bỏ các thẻ $ref trỏ đến "#/components/schemas/" (bị rỗng tên DTO).
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k == "$ref" and v == "#/components/schemas/":
                # Bỏ qua dòng bị lỗi này
                continue
            new_dict[k] = sanitize_spec(v)
        return new_dict
    elif isinstance(data, list):
        return [sanitize_spec(item) for item in data]
    return data

def get_openapi_agent(access_token: str = None):
    swagger_url = os.getenv("SWAGGER_JSON_URL")
    
    # 1. Fetch Swagger Spec
    try:
        response = requests.get(swagger_url)
        response.raise_for_status()
        raw_spec = response.json()
    except Exception as e:
        raise Exception(f"Lỗi kết nối Swagger: {str(e)}")

    # ================= CÁCH XỬ LÝ LỖI SCHEMA =================
    # Bước A: Đảm bảo cấu trúc components/schemas luôn tồn tại
    if "components" not in raw_spec:
        raw_spec["components"] = {"schemas": {}}
    elif "schemas" not in raw_spec["components"]:
        raw_spec["components"]["schemas"] = {}
        
    # Bước B: Dọn dẹp JSON để xóa các dòng $ref rỗng
    raw_spec = sanitize_spec(raw_spec)
    # =========================================================
    # Bổ sung base URL cho API nếu Swagger của Backend bị thiếu
    if "servers" not in raw_spec or not raw_spec["servers"]:
        raw_spec["servers"] = [
            {"url": "https://api.aeo.how", "description": "Production Server"}
        ]

    # 2. Khởi tạo Gemini Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0, 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 3. Setup Request Wrapper
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    
    requests_wrapper = RequestsWrapper(headers=headers)

    # 4. Tạo OpenAPI Agent
    openapi_spec = reduce_openapi_spec(raw_spec)
    
    agent = planner.create_openapi_agent(
        api_spec=openapi_spec,
        requests_wrapper=requests_wrapper,
        llm=llm,
        allow_dangerous_requests=True,
        verbose=True 
    )
    
    return agent