from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
# import torch  # Nếu bạn dùng PyTorch
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

# Cấu hình model
MODEL_PATH = "path/to/your/alzheimer_model.h5"  # Đường dẫn đến model của bạn
IMG_SIZE = (224, 224)  # Kích thước ảnh input cho model
CLASS_NAMES = ["Normal", "Alzheimer"]  # Tên các class

def load_model():
    """Load model AI đã train"""
    global model
    try:
        # Với TensorFlow/Keras
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Với PyTorch (uncomment nếu dùng PyTorch)
        # model = torch.load(MODEL_PATH, map_location='cpu')
        # model.eval()
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Tạo mock model cho demo
        model = create_mock_model()

def create_mock_model():
    """Tạo mock model cho demo khi không có model thật"""
    logger.info("Creating mock model for demo purposes")
    
    class MockModel:
        def predict(self, x):
            # Tạo kết quả random để demo
            batch_size = x.shape[0]
            # Random prediction với xác suất nghiêng về Normal (80%)
            predictions = np.random.random((batch_size, 2))
            predictions[:, 0] = predictions[:, 0] * 0.8 + 0.1  # Normal: 10-90%
            predictions[:, 1] = 1 - predictions[:, 0]  # Alzheimer: phần còn lại
            return predictions
    
    return MockModel()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Tiền xử lý ảnh để đưa vào model"""
    try:
        # Chuyển sang RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ảnh về kích thước model yêu cầu
        image = image.resize(IMG_SIZE)
        
        # Chuyển thành numpy array
        img_array = np.array(image)
        
        # Normalize pixel values về [0,1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Thêm batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Không thể xử lý ảnh")

def postprocess_prediction(prediction: np.ndarray) -> Dict[str, Any]:
    """Xử lý kết quả prediction từ model"""
    try:
        # Lấy xác suất của từng class
        probabilities = prediction[0]
        
        # Tìm class có xác suất cao nhất
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx]) * 100
        
        # Tạo response
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probability": {
                "normal": float(probabilities[0]) * 100,
                "alzheimer": float(probabilities[1]) * 100
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
        prediction = model.predict(processed_image)
        
        # Xử lý kết quả
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