import numpy as np
from fastapi import HTTPException
from config import MODEL_CONFIG, logger

def postprocess_prediction(prediction: np.ndarray):
    try:
        probs = prediction[0]
        probs = np.exp(probs) / np.sum(np.exp(probs))  # softmax
        idx = np.argmax(probs)
        return {
            "prediction": MODEL_CONFIG["class_names"][idx],
            "confidence": float(probs[idx]) * 100,
            "probability": {
                name: float(p) * 100 for name, p in zip(MODEL_CONFIG["class_names"], probs)
            },
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in postprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi xử lý kết quả dự đoán")
