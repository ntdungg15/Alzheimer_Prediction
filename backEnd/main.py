from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.endpoints import router as prediction_router
import logging

# Khởi tạo FastAPI app
app = FastAPI(
    title="Multi-Model Alzheimer Prediction API",
    description="API dự đoán bệnh Alzheimer từ ảnh não bộ hỗ trợ nhiều framework",
    version="2.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn router
app.include_router(prediction_router)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
