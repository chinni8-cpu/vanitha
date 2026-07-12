
import os

dirs_to_create = [
    "gestures_dataset/train/thumbs_up",
    "gestures_dataset/train/fist",
    "gestures_dataset/train/peace",
    "gestures_dataset/train/ok",
    "gestures_dataset/val/thumbs_up",
    "gestures_dataset/val/fist",
    "gestures_dataset/val/peace",
    "gestures_dataset/val/ok",
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

print("Directories created.")
