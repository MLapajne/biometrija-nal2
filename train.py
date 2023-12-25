from ultralytics import YOLO
import os, itertools, datetime, json

#os.chdir('/kaggle/working/origin-data/yolov8-ears')
#dname = os.path.dirname(os.path.abspath(__file__))
#print(dname)
#os.chdir(dname)
# Load a model
model = YOLO("yolov8n.pt")

# Best hyperparameters (previously found)
learning_rate = 0.001
dropout = 0.2
weight_decay = 0.0001

# Data augmentation parameters


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
    )
    # Save the model checkpoint
except Exception as e:
    print(f"An error occurred during training: {e}")

# Optionally, you can also save the training results and parameters for future reference
