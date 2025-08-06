from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import uvicorn
from typing import Dict, Any
import logging

# Khởi tạo FastAPI app
app = FastAPI(
    title="Alzheimer Prediction API",
    description="API dự đoán bệnh Alzheimer từ ảnh não bộ",
    version="1.0.0"
)

# Cấu hình CORS để frontend có thể gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable để lưu model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cấu hình model
MODEL_PATH = "models/resnet50-aug.pth"  # Đường dẫn đến model PyTorch
IMG_SIZE = (128, 128)  # Kích thước ảnh input cho model
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]  # Tên các class

def load_model():
    """Load model AI đã train"""
    global model
    try:
        model = models.resnet50(weights=None)  # Không load pretrained weights
        model.fc = nn.Identity()
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),                   # dropout_0
            nn.Flatten(),                        # flatten
            
            nn.BatchNorm1d(2048),                # batch_normalization
            nn.Linear(2048, 1024),               # dense
            
            nn.BatchNorm1d(1024),                # batch_normalization_1
            nn.Linear(1024, 512),                # dense_1
            
            nn.BatchNorm1d(512),                 # batch_normalization_2
            nn.ReLU(),                           # activation
            nn.Dropout(p=0.3),                   # dropout_2
        
            nn.Linear(512, 256),                 # dense_2
            nn.BatchNorm1d(256),                 # batch_normalization_3
            nn.ReLU(),                           # activation_1
            nn.Dropout(p=0.3),                   # dropout_3
        
            nn.Linear(256, 4)          # dense_3 (output)
        )
        # Với PyTorch
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Tiền xử lý ảnh để đưa vào model với PyTorch"""
    try:
        # Định nghĩa transform pipeline
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.278, 0.278, 0.27828214], 
                std=[0.32666304, 0.32666304, 0.32666304]
            )
        ])
        
        # Chuyển sang RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms và thêm batch dimension
        img_tensor = transform(image).unsqueeze(0)
        
        return img_tensor.to(device)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Không thể xử lý ảnh")

def postprocess_prediction(prediction: torch.Tensor) -> Dict[str, Any]:
    """Xử lý kết quả prediction từ model PyTorch"""
    try:
        if hasattr(prediction, 'cpu'):
            prediction = prediction.cpu().detach().numpy()
        elif hasattr(prediction, 'detach'):
            prediction = prediction.detach().numpy()
        
        probabilities = prediction[0]
        
        logger.info("probabilities: " + str(probabilities))
        
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx]) * 100
        
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probability": {
                CLASS_NAMES[0]: float(probabilities[0]) * 100,
                CLASS_NAMES[1]: float(probabilities[1]) * 100,
                CLASS_NAMES[2]: float(probabilities[2]) * 100,
                CLASS_NAMES[3]: float(probabilities[3]) * 100,
            },
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error postprocessing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý kết quả dự đoán")

@app.on_event("startup")
async def startup_event():
    """Load model khi khởi động server"""
    logger.info("Starting Alzheimer Prediction API...")
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Alzheimer Prediction API is running",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "api_version": "1.0.0"
    }

@app.post("/predict")
async def predict_alzheimer(file: UploadFile = File(...)):
    """
    API endpoint dự đoán bệnh Alzheimer từ ảnh
    
    Args:
        file: UploadFile - File ảnh được upload
    
    Returns:
        JSON response với kết quả dự đoán
    """
    try:
        # Kiểm tra model đã load chưa
        if model is None:
            raise HTTPException(status_code=500, detail="Model chưa được load")
        
        # Kiểm tra file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File phải là ảnh (JPEG, PNG, etc.)"
            )
        
        # Đọc và xử lý ảnh
        logger.info(f"Processing image: {file.filename}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Tiền xử lý ảnh
        processed_image = preprocess_image(image)
        
        # Dự đoán với model
        logger.info("Making prediction...")
        with torch.no_grad():
            prediction = model(processed_image)
        
        # Xử lý kết quả
        logger.info("handling results...")
        result = postprocess_prediction(prediction)
        
        logger.info(f"Prediction completed: {result['prediction']} ({result['confidence']:.2f}%)")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi server nội bộ")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    API endpoint dự đoán nhiều ảnh cùng lúc
    """
    if len(files) > 10:  # Giới hạn số lượng file
        raise HTTPException(status_code=400, detail="Tối đa 10 file mỗi lần")
    
    results = []
    
    for file in files:
        try:
            # Xử lý từng file
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            result = postprocess_prediction(prediction)
            
            results.append({
                "filename": file.filename,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})

@app.get("/model-info")
async def get_model_info():
    """Thông tin về model"""
    return {
        "model_path": MODEL_PATH,
        "input_size": IMG_SIZE,
        "classes": CLASS_NAMES,
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    # Chạy server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto reload khi code thay đổi
        log_level="info"
    )