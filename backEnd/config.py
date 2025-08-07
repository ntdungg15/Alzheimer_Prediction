import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlzheimerAPI")

MODEL_CONFIG = {
    "models_dir": "models/",
    "default_model": "resnet50",
    "img_size": (128, 128),
    "class_names": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
}

MODEL_PATHS = {
    "resnet50": "models/resnet50-aug.pth",
    "cnn":"models/cnn",
    "vgg16": "models/vgg16",
    "inception-v3": "model/inception-v3"
}

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
