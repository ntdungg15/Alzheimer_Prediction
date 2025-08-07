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
    import torchvision.models as torch_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
loaded_models = {}  # Dictionary để lưu nhiều model
current_model = None

# Cấu hình các model
MODEL_CONFIGS = {
    "cnn": {
        "path": "C:/Users/PC/Downloads/cnn_model.h5",
        "type": "tensorflow",
        "input_size": (224, 224),
        "preprocessing": "standard"
    },
    "resnet50": {
        "path": "models/resnet50-aug.pth", 
        "type": "pytorch",
        "input_size": (128, 128),
        "preprocessing": "resnet"
    },
    "vgg16": {
        "path": "C:/Users/PC/Downloads/vgg16_finetuned.pth",
        "type": "pytorch",
        "input_size": (128, 128),  # Match Kaggle training
        "preprocessing": "vgg"
    },
    "inception-v3": {
        "path": "C:/Users/PC/Downloads/inception_v3_model.h5",
        "type": "tensorflow", 
        "input_size": (128, 128),  # Match Kaggle training
        "preprocessing": "inception"
    }
}

# Class names - supporting both 4-class and binary classification
CLASS_NAMES_4 = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_NAMES_2 = ["Normal", "Alzheimer"]

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

# Định nghĩa ResNet50 model class cho PyTorch - MATCH với Kaggle notebook
class ResNet50AlzheimerModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet50AlzheimerModel, self).__init__()
        self.resnet50 = torch_models.resnet50(weights=None)  # Không load pretrained weights
        self.resnet50.fc = nn.Identity()
        
        # Custom classifier matching Kaggle exactly
        self.classifier = nn.Sequential(
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
        
            nn.Linear(256, num_classes)          # dense_3 (output)
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = self.classifier(x)
        return x

# Định nghĩa Inception V3 model class cho TensorFlow - MATCH với Kaggle notebook
class InceptionV3AlzheimerModel:
    """Inception V3 model for Alzheimer prediction matching Kaggle implementation"""
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

def load_model(model_name: str = "cnn"):
    """Load model AI đã train"""
    global loaded_models, current_model
    
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Model {model_name} không tồn tại trong cấu hình")
        return False

    config = MODEL_CONFIGS[model_name]
    
    try:
        # Kiểm tra file có tồn tại không
        import os
        if not os.path.exists(config["path"]):
            logger.warning(f"Model file không tồn tại: {config['path']}")
            # Tạo mock model cho demo
            loaded_models[model_name] = {
                "model": create_mock_model(model_name),
                "config": config
            }
            current_model = model_name
            return False
        
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
                    model = inception_builder.get_model()
                    logger.info("Created new Inception V3 model")
            else:
                # Regular TensorFlow model loading
                model = tf.keras.models.load_model(config["path"])
                logger.info(f"TensorFlow model {model_name} loaded successfully from {config['path']}")
                
        elif config["type"] == "pytorch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch không có sẵn")
                
            if model_name == "vgg16":
                try:
                    checkpoint = torch.load(config["path"], map_location='cpu')
                    logger.info(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Direct model'}")
                    
                    # Load 4-class model first (match Kaggle training)
                    model = VGG16AlzheimerModel(num_classes=4, pretrained=False, freeze_features=False)
                    
                    if isinstance(checkpoint, dict):
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # Remove 'module.' prefix nếu có
                        if any(key.startswith('module.') for key in state_dict.keys()):
                            state_dict = {key[7:]: value for key, value in state_dict.items()}
                        
                        model.load_state_dict(state_dict, strict=True)
                        logger.info("Loaded 4-class VGG16 from checkpoint")
                    else:
                        model = checkpoint
                    
                    # Move to device and set eval mode
                    model.to(device)
                    model.eval()
                    logger.info(f"PyTorch VGG16 model loaded successfully on device: {device}")
                        
                except Exception as load_error:
                    logger.error(f"Error loading VGG16: {str(load_error)}")
                    raise load_error
                    
            elif model_name == "resnet50":
                try:
                    model = ResNet50AlzheimerModel(num_classes=4)
                    
                    logger.info('Loading params for resnet50')
                    model.load_state_dict(torch.load(config["path"], map_location=device))
                    logger.info('ResNet50 loaded successfully')
                    model.to(device)
                    model.eval()
                    
                except Exception as load_error:
                    logger.error(f"Error loading ResNet50: {str(load_error)}")
                    raise load_error
            else:
                raise Exception(f"PyTorch model {model_name} chưa được implement")
        
        loaded_models[model_name] = {
            "model": model,
            "config": config
        }
        current_model = model_name
        return True
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Tạo mock model cho demo
        loaded_models[model_name] = {
            "model": create_mock_model(model_name),
            "config": config
        }
        current_model = model_name
        return False

def create_mock_model(model_name: str):
    """Tạo mock model cho demo khi không có model thật"""
    logger.info(f"Creating mock model for {model_name}")
    
    class MockModel:
        def __init__(self, model_name):
            self.model_name = model_name
            
        def predict(self, x):
            # Tạo kết quả deterministic để demo (không random)
            if isinstance(x, torch.Tensor):
                batch_size = x.shape[0]
                # Sử dụng mean của tensor để tạo seed deterministic
                seed = int(torch.mean(x).item() * 1000) % 1000
            else:
                batch_size = x.shape[0]
                # Sử dụng mean của array để tạo seed deterministic
                seed = int(np.mean(x) * 1000) % 1000
            
            # Set random seed để có kết quả nhất quán
            np.random.seed(seed)
            
            # Prediction với phân phối khác nhau cho từng model
            if self.model_name in ["vgg16", "resnet50", "inception-v3"]:
                # 4-class prediction - deterministic
                if self.model_name == "vgg16":
                    # VGG16 mock: thiên về NonDemented
                    predictions = np.array([[0.1, 0.15, 0.6, 0.15]])  # [Mild, Moderate, Non, VeryMild]
                elif self.model_name == "resnet50":
                    # ResNet50 mock: thiên về VeryMildDemented  
                    predictions = np.array([[0.15, 0.1, 0.25, 0.5]])
                else:  # inception-v3
                    # Inception mock: cân bằng hơn
                    predictions = np.array([[0.2, 0.2, 0.3, 0.3]])
            else:
                # CNN 2-class prediction - deterministic
                predictions = np.array([[0.7, 0.3]])  # [Normal, Alzheimer]
            
            return predictions
        
        def __call__(self, x):
            # Để hỗ trợ PyTorch-style calling
            return torch.tensor(self.predict(x), dtype=torch.float32)
        
        def eval(self):
            # Để hỗ trợ PyTorch eval mode
            return self
    
    return MockModel(model_name)
def preprocess_image(image: Image.Image, model_name: str = "cnn"):
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
            if preprocessing_type == "vgg":
                # VGG16 preprocessing - MATCH Kaggle notebook
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),  # Match Kaggle: 128x128
                    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])  # ImageNet normalization
                ])
                img_tensor = transform(image)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                return img_tensor.to(device)
                
            elif preprocessing_type == "resnet":
                # ResNet50 preprocessing - MATCH Kaggle notebook
                transform = transforms.Compose([
                    transforms.Resize(input_size),  
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.27828214, 0.27828214, 0.27828214], 
                                       std=[0.32666304, 0.32666304, 0.32666304])  
                ])
                img_tensor = transform(image).unsqueeze(0)
                return img_tensor.to(device)
                
        elif config["type"] == "tensorflow":
            # TensorFlow preprocessing
            img_array = np.array(image)
            
            if preprocessing_type == "inception":
                # Inception V3 preprocessing - MATCH Kaggle
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
    if model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} chưa được load")
    
    model_info = loaded_models[model_name]
    model = model_info["model"]
    config = model_info["config"]
    
    # Log để debug
    logger.info(f"=== PREDICTION DEBUG for {model_name} ===")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Config type: {config['type']}")
    
    try:
        if config["type"] == "tensorflow":
            # TensorFlow prediction
            prediction = model.predict(processed_image, verbose=0)
            logger.info(f"TensorFlow prediction shape: {prediction.shape}")
            logger.info(f"TensorFlow prediction: {prediction}")
            return prediction
        
        elif config["type"] == "pytorch":
            # PyTorch prediction
            with torch.no_grad():
                if isinstance(processed_image, torch.Tensor):
                    logger.info(f"Input tensor shape: {processed_image.shape}")
                    logger.info(f"Input tensor device: {processed_image.device}")
                    logger.info(f"Input tensor min/max: {processed_image.min():.3f}/{processed_image.max():.3f}")
                    logger.info(f"Input tensor dtype: {processed_image.dtype}")
                    
                    # Check if model is on correct device
                    if hasattr(model, 'parameters'):
                        model_device = next(model.parameters()).device
                        logger.info(f"Model device: {model_device}")
                        
                        # Move tensor to model device if needed
                        if processed_image.device != model_device:
                            processed_image = processed_image.to(model_device)
                            logger.info(f"Moved input to device: {model_device}")
                    
                    prediction = model(processed_image)
                    logger.info(f"Raw prediction: {prediction}")
                    logger.info(f"Raw prediction shape: {prediction.shape}")
                    logger.info(f"Raw prediction device: {prediction.device}")
                    
                    # Apply softmax if needed
                    if not torch.allclose(prediction.sum(dim=1), torch.ones(prediction.shape[0]), atol=1e-5):
                        prediction = torch.softmax(prediction, dim=1)
                        logger.info("Applied softmax to raw predictions")
                    
                    prediction = prediction.cpu().numpy()
                    logger.info(f"Final prediction after softmax: {prediction}")
                    
                    return prediction
                else:
                    raise ValueError("PyTorch model requires tensor input")
        
        else:
            raise ValueError(f"Unsupported model type: {config['type']}")
            
    except Exception as e:
        logger.error(f"Error in predict_with_model for {model_name}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Lỗi prediction: {str(e)}")

def postprocess_prediction(prediction: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Xử lý kết quả prediction từ model"""
    try:
        # Lấy xác suất của từng class
        probabilities = prediction[0]
        logger.info(f"Probabilities for {model_name}: {probabilities}")
        
        # Xác định class names dựa trên model
        if model_name in ["vgg16", "resnet50", "inception-v3"]:
            class_names = CLASS_NAMES_4
        else:
            class_names = CLASS_NAMES_2
        
        # Tìm class có xác suất cao nhất
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx]) * 100
        
        # Tạo response
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probability": {},
            "status": "success",
            "model_used": model_name
        }
        
        # Thêm probabilities cho từng class
        for i, class_name in enumerate(class_names):
            if i < len(probabilities):
                result["probability"][class_name] = float(probabilities[i]) * 100
        
        return result
        
    except Exception as e:
        logger.error(f"Error postprocessing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý kết quả dự đoán")

@app.on_event("startup")
async def startup_event():
    """Load models khi khởi động server"""
    logger.info("Starting Alzheimer Prediction API...")
    logger.info(f"PyTorch available: {TORCH_AVAILABLE}")
    logger.info(f"TensorFlow available: {TF_AVAILABLE}")
    logger.info(f"Device: {device}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Alzheimer Prediction API is running",
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "available_models": list(MODEL_CONFIGS.keys()),
        "pytorch_available": TORCH_AVAILABLE,
        "tensorflow_available": TF_AVAILABLE,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": list(loaded_models.keys()),
        "available_models": list(MODEL_CONFIGS.keys()),
        "api_version": "1.0.0",
        "pytorch_available": TORCH_AVAILABLE,
        "tensorflow_available": TF_AVAILABLE,
        "device": str(device)
    }

@app.post("/predict")
async def predict_alzheimer(file: UploadFile = File(...), model_name: str = "vgg16"):
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
        if model_name not in loaded_models:
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
        result = postprocess_prediction(prediction, model_name)
        
        logger.info(f"Prediction completed: {result['prediction']} ({result['confidence']:.2f}%) using {model_name}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi server nội bộ")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...), model_name: str = "vgg16"):
    """API endpoint dự đoán nhiều ảnh cùng lúc"""
    if len(files) > 10:  # Giới hạn số lượng file
        raise HTTPException(status_code=400, detail="Tối đa 10 file mỗi lần")
    
    # Load model nếu chưa được load
    if model_name not in loaded_models:
        load_model(model_name)
    
    results = []
    
    for file in files:
        try:
            # Xử lý từng file
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image, model_name)
            prediction = predict_with_model(processed_image, model_name)
            result = postprocess_prediction(prediction, model_name)
            
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
        "loaded_models": list(loaded_models.keys()),
        "class_names_4": CLASS_NAMES_4,
        "class_names_2": CLASS_NAMES_2,
        "pytorch_available": TORCH_AVAILABLE,
        "tensorflow_available": TF_AVAILABLE,
        "device": str(device)
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