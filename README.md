# TRAFFIQ AI
### AI-Powered Adaptive Traffic Signal Optimization

TRAFFIQ AI is an offline traffic analysis and adaptive signal scheduling prototype that uses computer vision to estimate lane-wise traffic demand and simulate traffic signal allocation.

The system analyses recorded intersection footage using a pretrained YOLO11n model and ByteTrack, extracts lane-wise traffic metrics, and uses a hybrid adaptive scheduling algorithm to allocate green signal time.

## 1. Problem Statement

Conventional fixed-time traffic signals allocate predetermined green durations to each approach, regardless of changing traffic demand.

TRAFFIQ AI explores a demand-based alternative by analysing vehicle activity in recorded traffic footage and allocating green time according to estimated lane conditions.

## 2. Proposed Solution

The system combines:

- **YOLO11n:** Pretrained vehicle detection.
- **ByteTrack:** Multi-object tracking across video frames.
- **Lane-wise analytics:** Estimates vehicle count, queue length, waiting time and traffic intensity.
- **Hybrid adaptive scheduler:** Combines priority-based selection with Green Denial Counter (GDC) starvation prevention.
- **Fixed-time baseline:** Provides a round-robin scheduling comparison.
- **Streamlit dashboard:** Visualizes detection, analytics and simulated signal decisions.

## 3. System Architecture

```text
Recorded Traffic Video
          |
          v
 YOLO11n Vehicle Detection
          |
          v
   ByteTrack Tracking
          |
          v
   Lane-wise Assignment
          |
          v
   Traffic Analytics CSV
          |
          +----------------------+
          |                      |
          v                      v
 Hybrid Adaptive Scheduler   Fixed-Time Scheduler
          |                      |
          v                      v
 Adaptive Signal Log        Fixed Signal Log
          |                      |
          +----------+-----------+
                     |
                     v
             Scheduler Comparison
                     |
                     v
             Streamlit Dashboard
```

## 4. Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Object Detection | YOLO11n |
| Object Tracking | ByteTrack |
| Computer Vision | OpenCV |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Video Conversion | FFmpeg / imageio-ffmpeg |
| Configuration | JSON |
| Analytics Storage | CSV |

## 5. Core Modules

### Vehicle Detection and Tracking

The system uses a pretrained YOLO11n model to detect:

- Cars
- Motorcycles
- Buses
- Trucks

ByteTrack maintains object IDs across frames.

### Lane-wise Traffic Analytics

Four manually defined lane regions are used to assign detected vehicles based on the bottom-centre point of their bounding boxes.

The system estimates:

- Current vehicle count
- Queue length
- Average waiting time
- Traffic intensity

These measurements are estimates derived from video tracking and are not calibrated ground-truth measurements.

### Hybrid Adaptive Signal Scheduling

The adaptive scheduler uses a priority score:

`Priority Score = (2 × Queue Length) + (0.5 × Waiting Time) + (0.5 × Vehicle Count)`

Waiting time is capped at 30 seconds for priority calculation.

Green time is calculated using:

`Green Time = 10 + (2 × Queue Length) + (0.3 × Waiting Time)`

The calculated green duration is restricted to:

- Minimum green: 10 seconds
- Maximum green: 25 seconds

The scheduler also uses a **Green Denial Counter (GDC)** to prioritize waiting approaches when their counter reaches the configured threshold.

Current configuration:

| Parameter | Value |
|---|---:|
| GDC Threshold | 20 |
| Minimum Green | 10 seconds |
| Maximum Green | 25 seconds |
| Yellow | 3 seconds |
| All-Red | 1 second |

The GDC threshold was selected after observing starvation in an earlier run on the same traffic clip. It is a prototype configuration, not a universally optimized value.

### Fixed-Time Baseline

A round-robin fixed-time scheduler allocates:

- Green: 15 seconds
- Yellow: 3 seconds
- All-Red: 1 second

This serves as a baseline for comparing signal-time allocation.

## 6. Experimental Results

The following results were obtained from a 120-second scheduler simulation using the recorded traffic analytics.

### Green-Time Allocation

| Lane | Adaptive | Fixed-Time |
|---|---:|---:|
| Lane 1 | 22 s | 30 s |
| Lane 2 | 45 s | 30 s |
| Lane 3 | 17 s | 21 s |
| Lane 4 | 12 s | 15 s |
| **Total** | **96 s** | **96 s** |

### Service Count

| Lane | Adaptive | Fixed-Time |
|---|---:|---:|
| Lane 1 | 1 | 2 |
| Lane 2 | 3 | 2 |
| Lane 3 | 1 | 2 |
| Lane 4 | 2 | 1 |

The adaptive scheduler allocated more green time to Lane 2 based on its estimated traffic demand, while the GDC mechanism helped ensure that other waiting lanes also received service.

**Important:** These results demonstrate simulated green-time allocation, not measured reductions in real-world congestion, queue length or waiting time. The recorded vehicles do not respond to the simulated signals.

## 7. Dashboard

The Streamlit dashboard contains four sections:

1. **Overview:** Project summary, system components and annotated traffic video.
2. **Traffic Analytics:** Lane-wise traffic metrics and visualizations.
3. **Signal Simulation:** Simulated signal states and scheduler decisions.
4. **Scheduler Comparison:** Adaptive and fixed-time green allocation and service comparison.

## 8. Project Structure

```text
TRAFFIQ_AI/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- assets/
|   |-- CSS/
|
|-- config/
|   |-- lane_rois.json
|
|-- src/
|   |-- mark_lanes.py
|   |-- lane_detection.py
|   |-- scheduler.py
|   |-- fixed_time_scheduler.py
|   |-- compare_schedulers.py
|
|-- test/
|
|-- dataset/       # Local input data; excluded from Git
|
|-- runs/          # Generated results; excluded from Git
|
|-- venv/          # Python environment; excluded from Git
```

## 9. Installation and Execution

### Prerequisites

- Python 3.11 recommended
- Git
- Internet connection for installing dependencies and obtaining pretrained model weights
- A recorded traffic video for analysis

### Step 1: Clone the Repository

```bash
git clone https://github.com/PAVAZHA-ARASI2024/TRAFFIQ_AI.git
cd TRAFFIQ_AI
```

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare Input Data

Place the recorded traffic video at:

```text
dataset/videos/traffic_video_03.mp4
```

Ensure the lane configuration file is available at:

```text
config/lane_rois.json
```

The pretrained `yolo11n.pt` weights must also be available. Ultralytics may download them automatically when the model is first initialized.

### Step 5: Run Vehicle Detection and Analytics

```bash
python src/lane_detection.py
```

### Step 6: Run the Schedulers

```bash
python src/scheduler.py
python src/fixed_time_scheduler.py
```

### Step 7: Compare Scheduling Results

```bash
python src/compare_schedulers.py
```

### Step 8: Launch the Dashboard

```bash
streamlit run app.py
```

## 10. Current Limitations

- The prototype processes recorded video rather than a live traffic camera feed.
- The detector uses pretrained weights; no custom model training is performed.
- Traffic metrics are estimates and have not been validated against manually annotated ground truth.
- The stationary-vehicle and waiting-time estimates can be affected by detection dropouts and tracking ID changes.
- Lane regions are manually defined for a fixed camera view.
- The scheduler is a simulation; it does not control physical traffic lights.
- Vehicles do not respond to the simulated signal phases.
- The evaluation uses a single recorded traffic clip.
- The comparison measures green-time allocation and service counts, not actual congestion reduction.

## 11. Future Enhancements

- Live traffic camera integration.
- More robust queue estimation and tracking.
- Traffic-demand prediction using machine learning.
- Dynamic green-time extension and early termination.
- Pedestrian-aware signal scheduling.
- Traffic simulation for measuring delay and queue reduction.
- Coordination between multiple intersections.

## 12. Project Goal

TRAFFIQ AI demonstrates how computer vision-based traffic analytics can support demand-aware signal scheduling through a transparent, rule-based adaptive algorithm.

The prototype provides an end-to-end foundation for further experimentation with intelligent traffic management systems.

---

**Project:** TRAFFIQ AI  
**Repository:** https://github.com/PAVAZHA-ARASI2024/TRAFFIQ_AI  
**Status:** Offline prototype and simulated adaptive signal scheduling