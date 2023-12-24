from ultralytics import YOLO
import os, itertools, datetime, json

#os.chdir('/kaggle/working/origin-data/yolov8-ears')
#dname = os.path.dirname(os.path.abspath(__file__))
#print(dname)
#os.chdir("/kaggle/working/origin-data/yolov8-ears/")
# Load a model
model = YOLO("yolov8n.pt")

# Best hyperparameters (previously found)
learning_rate = 0.001
dropout = 0.2
weight_decay = 0.0001

# Data augmentation parameters
augmentation_params = {
    'hsv_h': 0.015,  # Small adjustment
    'hsv_s': 0.35,   # Moderate adjustment for saturation
    'hsv_v': 0.35,   # Moderate adjustment for value/brightness
    'degrees': 10,  # Small rotation
    'translate': 0.1,  # Moderate translation
    'scale': 0.25,   # Moderate scaling
    'shear': 10,    # Mild shearing
    'perspective': 0.001,  # Mild perspective transformation
    'flipud': 0.0,  # No vertical flipping
    'fliplr': 0.5,  # 50% chance of horizontal flipping
    'mosaic': 1.0,
    'mixup': 0.1,   # Low mixup
    'copy_paste': 0.1  # Low copy-paste
}

# Training with best hyperparameters and augmentation parameters
try:
    model.train(
        data="ears.yaml", 
        epochs=20, 
        optimizer='AdamW', 
        pretrained=True, 
        patience=3,
        plots=False,
        val=True,
        augment=True,
        dropout=dropout,
        lr0=learning_rate,
        lrf=0.2,
        momentum=0.937,
        weight_decay=weight_decay,
        warmup_epochs=3,
        warmup_momentum=0.5,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        **augmentation_params
    )
    # Save the model checkpoint
except Exception as e:
    print(f"An error occurred during training: {e}")

# Optionally, you can also save the training results and parameters for future reference
