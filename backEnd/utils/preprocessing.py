import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from config import MODEL_CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_image_pytorch(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(MODEL_CONFIG["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.278]*3, std=[0.326]*3)
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def preprocess_image_tensorflow(image: Image.Image) -> np.ndarray:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(MODEL_CONFIG["img_size"])
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - [0.278]*3) / [0.326]*3
    return np.expand_dims(img_array, axis=0)
