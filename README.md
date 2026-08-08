````markdown
# 🪖 Helmet Violation Detection System

> AI-powered traffic surveillance system for detecting helmet violations and triple riding from dashcam footage.

## 🚦 Overview

The **Helmet Violation Detection System** is an AI-based computer vision project designed to automatically detect traffic violations involving motorcycles.

The system processes traffic and dashcam videos to:

- 🏍️ Detect motorcycles
- 👤 Detect riders
- 🆔 Track motorcycles and riders
- 🔗 Associate riders with motorcycles
- 🪖 Detect helmet usage
- ⚠️ Detect riders without helmets
- 👥 Detect triple riding
- 📸 Generate evidence images for confirmed violations

The system is designed for traffic-monitoring scenarios and has been tested with both **front-view and rear-view motorcycle footage**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏍️ Motorcycle Detection | Detects motorcycles in traffic videos |
| 👤 Person Detection | Detects people/riders |
| 🆔 Object Tracking | Tracks motorcycles and riders across frames |
| 🔗 Rider Association | Associates riders with the correct motorcycle |
| 🪖 Helmet Detection | Classifies helmet usage |
| 🚨 No Helmet Detection | Detects riders without helmets |
| 👥 Triple Riding Detection | Detects motorcycles carrying three or more riders |
| ⏱️ Temporal Verification | Uses multiple frames to reduce false detections |
| 📸 Evidence Generation | Automatically saves violation evidence |
| 🎥 Dashcam Support | Supports traffic surveillance and dashcam footage |

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │    Dashcam Video     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Motorcycle / Person │
                    │     Detection       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Object Tracking   │
                    │ Motorcycle + Person  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Rider Association  │
                    │ Person → Motorcycle  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Helmet Detection   │
                    │ With / Without       │
                    │ Helmet Classification│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Violation Engine    │
                    ├──────────────────────┤
                    │ • No Helmet          │
                    │ • Triple Riding      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Evidence Generation  │
                    └──────────────────────┘
````

---

# 🔄 Processing Pipeline

```text
Input Traffic Video
        │
        ▼
Motorcycle Detection
        │
        ▼
Person Detection
        │
        ▼
Object Tracking
        │
        ▼
Rider Association
        │
        ▼
Helmet Classification
        │
        ├───────────────┐
        │               │
        ▼               ▼
 With Helmet       Without Helmet
                        │
                        ▼
                 No Helmet Violation
                        
        Rider Count
             │
      ┌──────┼──────┐
      ▼      ▼      ▼
     1       2      3+
   Rider   Riders  Riders
                   │
                   ▼
             Triple Riding
                   │
                   ▼
           Evidence Generation
```

---

# 🪖 Helmet Detection

The project uses a custom-trained helmet detection model with two classes:

```text
With Helmet
Without Helmet
```

Helmet predictions are associated with individual riders instead of simply classifying the entire video frame.

Example:

```text
Bike ID 380

Rider ID 293 → With Helmet
Rider ID 450 → Without Helmet
```

The system also uses confidence thresholds and temporal information to reduce false detections.

---

# 🚨 Violation Detection

## No Helmet

A rider can be classified as a helmet violation when the helmet model detects:

```text
Without Helmet
```

with sufficient confidence.

The system checks helmet predictions across video frames to improve reliability.

---

## Triple Riding

Triple riding is detected when **three or more riders are associated with the same motorcycle**.

Conceptually:

```text
1 Rider → Normal
2 Riders → Normal
3+ Riders → Triple Riding
```

The system uses tracking and temporal verification to reduce false positives caused by temporary person detections.

---

# 🎯 Object Tracking

The system assigns persistent IDs to motorcycles and riders.

Example:

```text
Bike ID: 380
Rider ID: 293
Rider ID: 450
```

Tracking allows the system to follow the same objects across multiple video frames.

This is important for:

* Rider association
* Violation verification
* Duplicate prevention
* Evidence generation

---

# 📸 Evidence Generation

When a violation is confirmed, the system automatically saves an evidence image.

Evidence is stored in:

```text
outputs/evidence/
```

Example:

```text
bike_380_No_Helmet_20260808_171515.jpg
bike_9_Triple_Riding_20260808_185504.jpg
```

The filename contains:

```text
Bike ID
Violation Type
Timestamp
```

This makes violation evidence easier to identify and organize.

---

# 📊 Model Performance

The custom helmet model was evaluated using a validation dataset.

Current validation results:

| Class          | Precision | Recall | mAP@50 | mAP@50-95 |
| -------------- | --------: | -----: | -----: | --------: |
| With Helmet    |     0.695 |  0.846 |  0.823 |     0.480 |
| Without Helmet |     0.603 |  0.739 |  0.704 |     0.401 |
| Overall        |     0.649 |  0.793 |  0.763 |     0.440 |

The model performs better on **With Helmet** detection than **Without Helmet** detection.

Further improvement can be achieved by increasing dataset diversity and including more rear-view, low-resolution, occluded, and difficult traffic examples.

---

# 🎥 Camera Views

The system has been tested with different motorcycle viewing angles.

### Front View

```text
        Camera
           ↓
       👤 Rider
       👤 Rider
        🏍️ Bike
```

### Rear View

```text
       Dashcam
          ↓
       👤 Rider
       👤 Rider
        🏍️ Bike
```

Rear-view processing is particularly important for real-world dashcam deployment.

---

# 🗂️ Project Structure

```text
HelmetViolationProject/
│
├── data/
│   └── test_videos/
│
├── models/
│   └── helmet/
│
├── outputs/
│   └── evidence/
│
├── src/
│   ├── association.py
│   ├── check_gpu.py
│   ├── check_model.py
│   ├── detector.py
│   ├── evidence.py
│   ├── extract_rider_crops.py
│   ├── helmet_detector.py
│   ├── main.py
│   ├── video_processor.py
│   ├── video_reader.py
│   ├── violation_engine.py
│   ├── visualizer.py
│   └── utils/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Computer Vision

* OpenCV
* NumPy

### Deep Learning

* PyTorch
* YOLO

### Detection

* Motorcycle detection
* Person detection
* Custom helmet detection

### Tracking

* Motorcycle tracking
* Rider tracking

---

# ⚙️ Installation

## Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd HelmetViolationProject
```

## Create a virtual environment

```bash
python -m venv .venv
```

## Activate the environment

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Place your test video inside:

```text
data/test_videos/
```

Configure the input video path in:

```text
src/main.py
```

Then run:

```powershell
python src/main.py
```

The system processes the video and generates the required output and evidence.

---

# 📁 Output

Violation evidence is stored in:

```text
outputs/evidence/
```

Example terminal output:

```text
Bike 492 Rider ID 450: Without Helmet (0.728)

Evidence saved:
outputs/evidence/bike_492_No_Helmet_20260808_185504.jpg
```

For triple riding:

```text
Evidence saved:
outputs/evidence/bike_9_Triple_Riding_20260808_185504.jpg
```

---

# 🧪 Testing

The system has been tested using traffic footage containing:

* Single riders
* Two riders
* Multiple motorcycles
* Helmeted riders
* Riders without helmets
* Triple-riding scenarios
* Front-view motorcycles
* Rear-view motorcycles
* Moving traffic
* Partially occluded riders

---

# 🔧 Model Training

The helmet model can be retrained using a custom dataset containing:

```text
With Helmet
Without Helmet
```

Example YOLO training command:

```bash
yolo detect train model=<base-model>.pt data=<dataset>.yaml epochs=200 imgsz=640
```

Training performance should be evaluated using validation metrics rather than simply increasing the number of epochs.

---

# ⚠️ Current Limitations

The system can still face challenges in difficult traffic conditions.

### Small Objects

Riders and helmets may become extremely small when motorcycles are far from the camera.

### Occlusion

Riders can partially block each other, especially during:

* Dense traffic
* Two-rider situations
* Triple riding

### Motion Blur

Fast-moving motorcycles can produce blurred rider and helmet regions.

### Lighting

Performance can decrease under:

* Very low light
* Strong shadows
* Backlighting
* Night-time conditions

### Detection Confidence

Helmet predictions can become less reliable when the rider is:

* Very far away
* Partially hidden
* Poorly illuminated
* At an unusual viewing angle

---

# 🚀 Future Improvements

* [ ] Improve helmet detection dataset
* [ ] Add more rear-view training samples
* [ ] Improve small-object detection
* [ ] Improve rider association
* [ ] Automatic number plate detection
* [ ] Number plate OCR
* [ ] Violation database
* [ ] Web-based monitoring dashboard
* [ ] Real-time CCTV support
* [ ] Real-time camera streaming
* [ ] Automated violation reports
* [ ] Cloud evidence storage
* [ ] Multi-camera support
* [ ] Traffic analytics
* [ ] Vehicle speed estimation
* [ ] Lane detection
* [ ] Edge-device deployment

---

# 🔐 Git Repository

Large and generated files should not be committed to the repository.

The `.gitignore` file excludes files such as:

```text
.venv/
__pycache__/
outputs/
*.pt
*.pth
*.onnx
large video files
datasets
generated evidence
```

This keeps the GitHub repository lightweight and focused on source code and configuration.

---

# 📌 Project Status

## Core System

* [x] Motorcycle detection
* [x] Person detection
* [x] Motorcycle tracking
* [x] Rider tracking
* [x] Rider association
* [x] Helmet classification
* [x] No Helmet detection
* [x] Triple Riding detection
* [x] Temporal violation verification
* [x] Evidence generation
* [x] Dashcam testing

## Future Development

* [ ] Number plate recognition
* [ ] OCR
* [ ] Database integration
* [ ] Web dashboard
* [ ] Real-time deployment
* [ ] Multi-camera support
* [ ] Automated reporting

---

# 🎓 Project Objective

The primary objective of this project is to demonstrate how **deep learning, computer vision, and object tracking** can be combined to create an automated traffic-rule monitoring system.

The project combines:

```text
Object Detection
       +
Object Tracking
       +
Rider Association
       +
Helmet Classification
       +
Temporal Verification
       +
Violation Detection
       +
Evidence Generation
```

This provides a foundation for intelligent transportation and automated traffic surveillance applications.

---

# 👨‍💻 Author

**Helmet Violation Detection System**

An academic computer vision and intelligent transportation project.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

```
```
