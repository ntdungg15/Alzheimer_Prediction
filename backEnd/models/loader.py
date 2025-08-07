import torch
import tensorflow as tf
from pathlib import Path
from models.architecture import create_pytorch_resnet50_architecture, create_tensorflow_model_architecture
from config import MODEL_CONFIG, logger, MODEL_PATHS

models_dict = {}

class ModelType:
    PYTORCH_PTH = "pytorch_pth"
    PYTORCH_CHECKPOINT = "pytorch_checkpoint"
    TENSORFLOW_H5 = "tensorflow_h5"
    TENSORFLOW_SAVEDMODEL = "tensorflow_savedmodel"

def detect_model_type(model_path: str) -> str:
    path = Path(model_path)
    if path.suffix == '.pth':
        return ModelType.PYTORCH_PTH
    elif path.suffix == '.pt':
        return ModelType.PYTORCH_CHECKPOINT
    elif path.suffix == '.h5':
        return ModelType.TENSORFLOW_H5
    elif path.is_dir() and (path / 'saved_model.pb').exists():
        return ModelType.TENSORFLOW_SAVEDMODEL
    raise ValueError(f"Unsupported model format: {path}")

def load_model(model_name: str, model_path: str = None):
    from config import TF_AVAILABLE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_path is None:
        model_path = MODEL_PATHS[model_name]
        if model_path == None:
            base = Path(MODEL_CONFIG["models_dir"])
            for ext in ['.pth', '.pt', '.h5']:
                candidate = base / f"{model_name}{ext}"
                if candidate.exists():
                    model_path = str(candidate)
                    break
            if not model_path:
                saved_dir = base / model_name
                if saved_dir.is_dir():
                    model_path = str(saved_dir)

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model {model_name} not found")

    model_type = detect_model_type(model_path)
    logger.info(f"Loading model {model_name} as {model_type}")
    model = None
    if model_type in [ModelType.PYTORCH_PTH, ModelType.PYTORCH_CHECKPOINT]:
        if (model_name == "resnet50"):
            model = create_pytorch_resnet50_architecture()
        if model_type == ModelType.PYTORCH_PTH:
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        framework = 'pytorch'

    elif model_type == ModelType.TENSORFLOW_H5:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        model = tf.keras.models.load_model(model_path)
        framework = 'tensorflow'

    elif model_type == ModelType.TENSORFLOW_SAVEDMODEL:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        model = tf.saved_model.load(model_path)
        framework = 'tensorflow'

    else:
        raise ValueError("Unknown model type")

    models_dict[model_name] = {
        "model": model,
        "framework": framework,
        "model_type": model_type,
        "model_path": model_path
    }
