from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io

from models.loader import load_model, models_dict
from utils.preprocessing import preprocess_image_pytorch, preprocess_image_tensorflow
from utils.postprocessing import postprocess_prediction
from config import MODEL_CONFIG, TF_AVAILABLE, logger

router = APIRouter()

@router.on_event("startup")
async def startup_event():
    logger.info("Starting API and loading default model...")
    try:
        load_model(MODEL_CONFIG["default_model"])
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")

@router.get("/")
async def root():
    return {
        "message": "Multi-Model Alzheimer Prediction API is running",
        "status": "healthy",
        "loaded_models": list(models_dict.keys()),
        "frameworks_available": {
            "pytorch": True,
            "tensorflow": TF_AVAILABLE
        }
    }

@router.get("/models")
async def list_models():
    return {
        "loaded_models": {
            name: {
                "framework": info["framework"],
                "model_type": info["model_type"],
                "model_path": info["model_path"]
            }
            for name, info in models_dict.items()
        },
        "available_models_dir": MODEL_CONFIG["models_dir"]
    }

@router.post("/load-model")
async def load_model_endpoint(model_name: str, model_path: str = None):
    try:
        load_model(model_name, model_path)
        return {
            "message": f"Model {model_name} loaded successfully",
            "model_info": {
                "framework": models_dict[model_name]["framework"],
                "model_type": models_dict[model_name]["model_type"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def predict_alzheimer(
    file: UploadFile = File(...),
    model_name: str = Query(default=MODEL_CONFIG["default_model"])
):
    if model_name not in models_dict:
        try:
            load_model(model_name)
        except:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    model_info = models_dict[model_name]
    framework = model_info["framework"]

    if framework == "pytorch":
        input_tensor = preprocess_image_pytorch(image)
        prediction = model_info["model"](input_tensor).detach().cpu().numpy()
    else:
        input_array = preprocess_image_tensorflow(image)
        prediction = model_info["model"](input_array).numpy()

    result = postprocess_prediction(prediction)
    result["model_used"] = model_name
    result["framework"] = framework
    return JSONResponse(content=result)

@router.get("/model-info/{model_name}")
async def get_model_info(model_name: str):
    if model_name not in models_dict:
        raise HTTPException(status_code=404, detail="Model not found")

    info = models_dict[model_name]
    return {
        "model_name": model_name,
        "framework": info["framework"],
        "model_type": info["model_type"],
        "model_path": info["model_path"],
        "input_size": MODEL_CONFIG["img_size"],
        "classes": MODEL_CONFIG["class_names"]
    }
