import torch.nn as nn
import torchvision.models as models
import tensorflow as tf

def create_pytorch_resnet50_architecture():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Flatten(),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 4)
    )
    return model

def create_tensorflow_model_architecture():
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(128, 128, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return model
