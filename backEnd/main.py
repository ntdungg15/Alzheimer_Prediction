from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Dict, Any
import logging

# Cấu hình logging trước
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import với try-catch để tránh warnings
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

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

# Global variables để lưu models
models = {}  # Dictionary để lưu nhiều model
current_model = None

# Cấu hình các model
MODEL_CONFIGS = {
    "cnn": {
        "path": "models/cnn_model.h5",
        "type": "tensorflow",
        "input_size": (224, 224),
        "preprocessing": "standard"
    },
    "resnet50": {
        "path": "models/resnet50_model.h5", 
        "type": "tensorflow",
        "input_size": (224, 224),
        "preprocessing": "resnet"
    },
    "vgg16": {
        "path": "models/vgg16_finetuned.pth",  # Sử dụng relative path
        "type": "pytorch",
        "input_size": (128, 128),  # Match Kaggle training
        "preprocessing": "vgg"
    },
    "inception-v3": {
        "path": "models/inception_v3_model.h5",
        "type": "tensorflow", 
        "input_size": (128, 128),  # Match Kaggle training
        "preprocessing": "inception"
    }
}

CLASS_NAMES = ["Normal", "Alzheimer"]  # Tên các class

# Định nghĩa VGG16 model class cho PyTorch - MATCH với Kaggle notebook
class VGG16AlzheimerModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, freeze_features=True):
        super(VGG16AlzheimerModel, self).__init__()
        from torchvision import models
        self.vgg16 = models.vgg16(pretrained=pretrained)
        
        # Replace pooling with global pooling (match Kaggle)
        self.vgg16.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optionally freeze all convolutional layers
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
        
        # New classifier for 512→4096→4096→num_classes (match Kaggle exactly)
        self.vgg16.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.vgg16.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.vgg16(x)

    def unfreeze_layers(self, num_layers=10):
        params = list(self.vgg16.features.parameters())
        for p in params[-num_layers:]: 
            p.requires_grad = True
        logger.info(f"Unfroze {num_layers} conv layers.")

# Compatibility class for 2-class prediction (Alzheimer binary)
class VGG16Binary(nn.Module):
    def __init__(self, pretrained_4class_model):
        super(VGG16Binary, self).__init__()
        self.vgg16 = pretrained_4class_model.vgg16
        
        # Replace final layer: 4096 -> 2 classes instead of 4
        self.vgg16.classifier[-1] = nn.Linear(4096, 2)
        
        # Initialize new final layer
        nn.init.normal_(self.vgg16.classifier[-1].weight, 0, 0.01)
        nn.init.constant_(self.vgg16.classifier[-1].bias, 0)
    
    def forward(self, x):
        return self.vgg16(x)

# Định nghĩa Inception V3 model class cho TensorFlow - MATCH với Kaggle notebook
class InceptionV3AlzheimerModel:
    """
    Inception V3 model for Alzheimer prediction matching Kaggle implementation
    """
    def __init__(self, num_classes=4, input_size=(128, 128, 3)):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Inception V3 model")
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """Build Inception V3 model exactly matching Kaggle notebook"""
        try:
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.layers import Dense, Dropout, Flatten
            from tensorflow.keras.models import Model
            import tensorflow as tf
            
            # Base Inception V3 model - MATCH Kaggle
            inception = InceptionV3(
                input_shape=self.input_size,  # (128, 128, 3) 
                weights='imagenet', 
                include_top=False
            )
            
            # Freeze all layers - MATCH Kaggle
            for layer in inception.layers:
                layer.trainable = False
                
            # Custom classifier - MATCH Kaggle exactly
            x = Dropout(0.5)(inception.output)      
            x = Flatten()(x)                        
            x = Dense(1024, activation='relu')(x)   
            x = Dense(512, activation='relu')(x)    
            x = Dense(256, activation='relu')(x)    
            x = Dense(128, activation='relu')(x)    
            
            # Output layer
            prediction = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            self.model = Model(inputs=inception.input, outputs=prediction)
            
            # Compile model - MATCH Kaggle
            METRICS = [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                tf.keras.metrics.AUC(name='auc'),
            ]
            
            self.model.compile(
                optimizer='adam',
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=METRICS
            )
            
            logger.info("Inception V3 model built successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            logger.error(f"Error building Inception V3 model: {str(e)}")
            raise e
    
    def get_model(self):
        """Return the compiled model"""
        return self.model
    
    def predict(self, x):
        """Prediction method"""
        if self.model:
            return self.model.predict(x)
        else:
            raise ValueError("Model not built yet")

# Compatibility class for 2-class prediction (Inception V3 binary)
class InceptionV3Binary:
    """Convert 4-class Inception V3 to binary classifier"""
    def __init__(self, pretrained_4class_model):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")
            
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Model
        import tensorflow as tf
        
        # Get the base model without the final classification layer
        base_model = pretrained_4class_model.model
        
        # Remove the last layer (4-class output)
        x = base_model.layers[-2].output  # Get layer before final Dense
        
        # Add new binary classification layer
        binary_output = Dense(2, activation='softmax', name='binary_output')(x)
        
        # Create new model
        self.model = Model(inputs=base_model.input, outputs=binary_output)
        
        # Compile for binary classification
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                tf.keras.metrics.AUC(name='auc'),
            ]
        )
        
        logger.info("Created binary Inception V3 classifier (Normal vs Alzheimer)")
    
    def predict(self, x):
        return self.model.predict(x)
    
    def get_model(self):
        return self.model

def load_model(model_name: str = "cnn"):
    """Load model AI đã train"""
    global models, current_model
    
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Model {model_name} không tồn tại trong cấu hình")
        return False
    
    config = MODEL_CONFIGS[model_name]
    
    try:
        # Kiểm tra file có tồn tại không
        import os
        if not os.path.exists(config["path"]):
            raise FileNotFoundError(f"Model file không tồn tại: {config['path']}")
        
        if config["type"] == "tensorflow":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow không có sẵn")
            
            # Special handling for Inception V3
            if model_name == "inception-v3":
                try:
                    # Try to load saved model first
                    model = tf.keras.models.load_model(config["path"])
                    logger.info(f"Loaded pre-trained Inception V3 from {config['path']}")
                    
                except Exception as load_error:
                    logger.warning(f"Failed to load pre-trained model: {load_error}")
                    logger.info("Creating new Inception V3 model from scratch")
                    
                    # Create new model matching Kaggle architecture
                    inception_builder = InceptionV3AlzheimerModel(
                        num_classes=4, 
                        input_size=(128, 128, 3)  # Match Kaggle
                    )
                    model_4class = inception_builder.get_model()
                    
                    # Convert to binary classifier
                    model = InceptionV3Binary(inception_builder).get_model()
                    logger.info("Created new Inception V3 binary classifier")
            else:
                # Regular TensorFlow model loading
                model = tf.keras.models.load_model(config["path"])
                logger.info(f"TensorFlow model {model_name} loaded successfully from {config['path']}")
                
        elif config["type"] == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch không có sẵn")
            # Load PyTorch model (VGG16)
            if model_name == "vgg16":
                try:
                    checkpoint = torch.load(config["path"], map_location='cpu')
                    logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Direct model'}")
                    
                    # Load 4-class model first (match Kaggle training)
                    try:
                        model_4class = VGG16AlzheimerModel(num_classes=4, pretrained=False, freeze_features=False)
                        
                        if isinstance(checkpoint, dict):
                            if 'model_state_dict' in checkpoint:
                                state_dict = checkpoint['model_state_dict']
                            else:
                                state_dict = checkpoint
                            
                            # Remove 'module.' prefix nếu có
                            if any(key.startswith('module.') for key in state_dict.keys()):
                                state_dict = {key[7:]: value for key, value in state_dict.items()}
                            
                            model_4class.load_state_dict(state_dict, strict=True)
                            logger.info("Loaded 4-class VGG16 from checkpoint")
                        else:
                            model_4class = checkpoint
                        
                        # Convert to binary classifier (Normal vs Alzheimer)
                        model = VGG16Binary(model_4class)
                        logger.info("Created binary classifier from 4-class model")
                        
                        # Map 4 classes to 2: [MildDemented, ModerateDemented, NonDemented, VeryMildDemented] -> [Alzheimer, Normal]
                        # Classes 0,1,3 = Alzheimer variants -> class 1
                        # Class 2 = NonDemented -> class 0 (Normal)
                        
                    except Exception as e1:
                        logger.error(f"Failed to load as 4-class model: {e1}")
                        raise e1
                    
                    model.eval()
                    logger.info(f"PyTorch VGG16 model loaded successfully")
                        
                except Exception as load_error:
                    logger.error(f"Error loading VGG16: {str(load_error)}")
                    raise load_error
                        
            else:
                raise Exception(f"PyTorch model {model_name} chưa được implement")
        
        models[model_name] = {
            "model": model,
            "config": config
        }
        current_model = model_name
        return True
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Tạo mock model cho demo
        models[model_name] = {
            "model": create_mock_model(),
            "config": config
        }
        current_model = model_name
        return False

def create_mock_model():
    """Tạo mock model cho demo khi không có model thật"""
    logger.info("Creating mock model for demo purposes")
    
    class MockModel:
        def predict(self, x):
            # Tạo kết quả random để demo
            if isinstance(x, torch.Tensor):
                batch_size = x.shape[0]
            else:
                batch_size = x.shape[0]
            
            # Random prediction với xác suất nghiêng về Normal (80%)
            predictions = np.random.random((batch_size, 2))
            predictions[:, 0] = predictions[:, 0] * 0.8 + 0.1  # Normal: 10-90%
            predictions[:, 1] = 1 - predictions[:, 0]  # Alzheimer: phần còn lại
            return predictions
        
        def __call__(self, x):
            # Để hỗ trợ PyTorch-style calling
            return torch.tensor(self.predict(x), dtype=torch.float32)
        
        def eval(self):
            # Để hỗ trợ PyTorch eval mode
            return self
    
    return MockModel()

def preprocess_image(image: Image.Image, model_name: str = "cnn") -> np.ndarray:
    """Tiền xử lý ảnh để đưa vào model"""
    try:
        config = MODEL_CONFIGS[model_name]
        input_size = config["input_size"]
        preprocessing_type = config["preprocessing"]
        
        # Chuyển sang RGB nếu cần
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ảnh về kích thước model yêu cầu
        image = image.resize(input_size)
        
        if config["type"] == "pytorch":
            # PyTorch preprocessing (VGG16) - MATCH Kaggle notebook
            if preprocessing_type == "vgg":
                # Kaggle notebook preprocessing: grayscale->3ch, resize 128, ImageNet normalize
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),  # Match Kaggle: 128x128
                    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])  # ImageNet normalization
                ])
                # Apply transform directly to PIL Image
                img_tensor = transform(image)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                return img_tensor
        
        elif config["type"] == "tensorflow":
            # TensorFlow preprocessing
            # Chuyển thành numpy array
            img_array = np.array(image)
            
            if preprocessing_type == "resnet":
                # ResNet preprocessing
                try:
                    from tensorflow.keras.applications.resnet50 import preprocess_input
                    img_array = preprocess_input(img_array)
                except ImportError:
                    logger.warning("TensorFlow not available, using standard preprocessing")
                    img_array = img_array.astype(np.float32) / 255.0
            elif preprocessing_type == "inception":
                # Inception V3 preprocessing - MATCH Kaggle
                # Kaggle uses 128x128 input size for Inception V3
                try:
                    from tensorflow.keras.applications.inception_v3 import preprocess_input
                    img_array = preprocess_input(img_array)
                    logger.info("Applied Inception V3 specific preprocessing")
                except ImportError:
                    logger.warning("TensorFlow not available, using standard preprocessing")
                    # Standard normalization: scale to [-1, 1] range (Inception V3 expects this)
                    img_array = img_array.astype(np.float32) / 127.5 - 1.0
            else:
                # Standard preprocessing
                img_array = img_array.astype(np.float32) / 255.0
            
            # Thêm batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Không thể xử lý ảnh")

def predict_with_model(processed_image, model_name: str):
    """Thực hiện prediction với model được chọn"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} chưa được load")
    
    model_info = models[model_name]
    model = model_info["model"]
    config = model_info["config"]
    
    try:
        if config["type"] == "tensorflow":
            # TensorFlow prediction
            prediction = model.predict(processed_image)
            return prediction
        
        elif config["type"] == "pytorch":
            # PyTorch prediction
            with torch.no_grad():
                if isinstance(processed_image, torch.Tensor):
                    logger.info(f"Input tensor shape: {processed_image.shape}")
                    logger.info(f"Input tensor min/max: {processed_image.min():.3f}/{processed_image.max():.3f}")
                    
                    prediction = model(processed_image)
                    logger.info(f"Raw prediction: {prediction}")
                    logger.info(f"Raw prediction shape: {prediction.shape}")
                    
                    # Apply softmax if needed
                    if not torch.allclose(prediction.sum(dim=1), torch.ones(prediction.shape[0]), atol=1e-5):
                        prediction = torch.softmax(prediction, dim=1)
                        logger.info("Applied softmax to raw predictions")
                    
                    prediction = prediction.cpu().numpy()
                    logger.info(f"Final prediction after softmax: {prediction}")
                    
                    # For VGG16: model outputs 2 classes directly (Normal, Alzheimer)
                    return prediction
                else:
                    raise ValueError("PyTorch model requires tensor input")
        
        else:
            raise ValueError(f"Unsupported model type: {config['type']}")
            
    except Exception as e:
        logger.error(f"Error in predict_with_model for {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Lỗi prediction: {str(e)}")

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
    """Load models khi khởi động server"""
    logger.info("Starting Alzheimer Prediction API...")
    # Load default model (CNN) at startup
    load_model("cnn")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Alzheimer Prediction API is running",
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "available_models": list(MODEL_CONFIGS.keys())
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "available_models": list(MODEL_CONFIGS.keys()),
        "api_version": "1.0.0"
    }

@app.post("/predict")
async def predict_alzheimer(file: UploadFile = File(...), model_name: str = "cnn"):
    """
    API endpoint dự đoán bệnh Alzheimer từ ảnh
    
    Args:
        file: UploadFile - File ảnh được upload
        model_name: str - Tên model muốn sử dụng (cnn, resnet50, vgg16, inception-v3)
    
    Returns:
        JSON response với kết quả dự đoán
    """
    try:
        # Load model nếu chưa được load
        if model_name not in models:
            logger.info(f"Loading model {model_name}...")
            success = load_model(model_name)
            if not success:
                logger.warning(f"Failed to load {model_name}, using mock model")
        
        # Kiểm tra file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File phải là ảnh (JPEG, PNG, etc.)"
            )
        
        # Đọc và xử lý ảnh
        logger.info(f"Processing image: {file.filename} with model: {model_name}")
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Tiền xử lý ảnh theo model
        processed_image = preprocess_image(image, model_name)
        
        # Dự đoán với model
        logger.info("Making prediction...")
        prediction = predict_with_model(processed_image, model_name)
        
        # Xử lý kết quả
        result = postprocess_prediction(prediction)
        result["model_used"] = model_name  # Thêm thông tin model đã sử dụng
        
        logger.info(f"Prediction completed: {result['prediction']} ({result['confidence']:.2f}%) using {model_name}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi server nội bộ")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...), model_name: str = "cnn"):
    """
    API endpoint dự đoán nhiều ảnh cùng lúc
    """
    if len(files) > 10:  # Giới hạn số lượng file
        raise HTTPException(status_code=400, detail="Tối đa 10 file mỗi lần")
    
    # Load model nếu chưa được load
    if model_name not in models:
        load_model(model_name)
    
    results = []
    
    for file in files:
        try:
            # Xử lý từng file
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image, model_name)
            prediction = predict_with_model(processed_image, model_name)
            result = postprocess_prediction(prediction)
            result["model_used"] = model_name
            
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
    """Thông tin về các model"""
    return {
        "available_models": MODEL_CONFIGS,
        "loaded_models": list(models.keys()),
        "classes": CLASS_NAMES
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