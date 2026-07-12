# Hand Gesture Recognition

This project provides a comprehensive suite for hand gesture recognition, implementing two distinct approaches:
1.  A **Deep Learning pipeline** for training and deploying high-accuracy gesture detection and classification models on large datasets like HaGRID.
2.  A lightweight, real-time **MediaPipe-based recognizer** with a futuristic Heads-Up Display (HUD) for immediate, on-the-fly gesture classification without extensive training.

![Demo GIF](images/demo.gif)

## Features

### 1. Deep Learning Module
- **Multiple Architectures**: Supports a wide range of models, including:
    - ConvNeXt (Base)
    - MobileNetV3 (Large & Small)
    - ResNet (18, 152)
    - SSD-Lite with MobileNetV3 Large
    - Vision Transformer (ViT-B16)
- **Training & Evaluation**: Scripts for training and testing models, with support for multi-GPU (DDP).
- **Configuration Driven**: Easily manage experiments by modifying YAML configuration files.
- **Live Demo**: A real-time demo script (`demo.py`) to visualize bounding box predictions from a trained model using a webcam.

### 2. MediaPipe HUD Module
- **Lightweight & Fast**: Runs in real-time with minimal resource usage, leveraging Google's MediaPipe for hand tracking.
- **Hybrid Classification**: Combines a nearest-neighbor search on gesture templates with a robust rule-based engine for high accuracy on a wide range of gestures.
- **Interactive HUD**: A visually appealing and informative Heads-Up Display showing the detected gesture, confidence score, and FPS.
- **On-the-Fly Training**: Interactively add new custom gestures by simply showing them to the camera and giving them a name.

## HaGRID Dataset Information

The primary dataset for the deep learning module is [HaGRID (Hand Gesture Recognition Image Dataset)](https://github.com/hukenovs/hagrid).

The dataset contains **65,977** unique persons and at least this number of unique scenes. The subjects are people over 18 years old. The dataset was collected mainly indoors with considerable variation in lighting, including artificial and natural light. Besides, the dataset includes images taken in extreme conditions such as facing and backing to a window. Also, the subjects had to show gestures at a distance of 0.5 to 4 meters from the camera.

For more information see the original arxiv [paper](https://arxiv.org/abs/2206.08219).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chinnu8-cpu/vanitha.git
    cd vanitha
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

### Part 1: Deep Learning Module

This module is intended for training high-performance models on a structured dataset like HaGRID.

#### 1. Dataset Setup
You will need to download and structure your dataset. Use the provided scripts as a reference for preparing your data:
- `download.py`
- `organize_dataset.py`
- `converters/`

To convert annotations to [Coco](https://cocodataset.org/#home) format, run the following command:

```bash
python -m converters.hagrid_to_coco --cfg <CONFIG_PATH> --mode <'hands' or 'gestures'>
```

#### 2. Training a Model
To train a model, choose a configuration file from the `configs/` directory and run the `run.py` script.

```bash
python run.py --command train --path_to_config configs/MobileNetV3_small.yaml --n_gpu 1
```
- `--command train`: Specifies the training pipeline.
- `--path_to_config`: Path to the model and dataset configuration.
- `--n_gpu`: Number of GPUs to use for training.

Checkpoints will be saved to the directory specified in the config file.

#### 3. Evaluating a Model
To evaluate a model on the test set, use the `test` command. Make sure the `checkpoint` path in your config file points to a trained model.

```bash
python run.py --command test --path_to_config configs/MobileNetV3_small.yaml --n_gpu 1
```

#### 4. Live Demo
To run the live demo with a trained deep learning model, update the `checkpoint` path in the config file and run `demo.py`.

```bash
python demo.py --path_to_config configs/MobileNetV3_small.yaml
```

---

### Part 2: MediaPipe HUD Module

This module is a standalone, zero-shot recognizer that works out-of-the-box.

#### 1. First-Time Setup & Running
Simply run the `gesture_recognizer.py` script.

```bash
python gesture_recognizer.py
```

On the first run, it will automatically:
1.  Download a small set of reference images for basic gestures.
2.  Process these images to create a template database (`gesture_database.json`).
3.  Launch the webcam with the real-time HUD.

#### 2. In-App Controls
- **`q`**: Quit the application.
- **`r`**: Retrain the gesture templates from the source images.
- **`c`**: Capture the current hand pose, prompt for a name in the terminal, and save it as a new custom gesture.

## Project Structure
```
├── configs/              # Model and training configurations
├── models/               # Model definitions (detectors, classifiers)
├── custom_utils/         # Helper scripts for training, DDP, and utilities
├── dataset/              # Dataset-related scripts and annotations
├── converters/           # Scripts to convert dataset formats
├── run.py                # Main entry point for training/testing the DL models
├── demo.py               # Live demo for the trained DL models
└── gesture_recognizer.py # Standalone MediaPipe HUD recognizer
```

## License
This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. Please see the specific [license](license/en_us.pdf) file.

## Citation
You can cite the original HaGRID paper using the following BibTeX entry:

    @InProceedings{Kapitanov_2022_WACV,
        author    = {Kapitanov, Aleksandr and Nuzhdin, Aleksandr and Kenin, Roman and Shpilman, Aleksandr},
        title     = {HaGRID -- HAnd Gesture Recognition Image Dataset},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
        month     = {January},
        year      = {2022},
        pages     = {4572-4581}
    }