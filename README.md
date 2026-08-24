\# EdgeAdapt — Adaptive Edge Inference on NVIDIA Jetson Nano



\*\*EdgeAdapt\*\* is an adaptive real-time object detection and tracking system designed for resource-constrained edge devices. It runs on an \*\*NVIDIA Jetson Nano\*\* using \*\*YOLOv5n and TensorRT\*\*, dynamically adjusting its inference workload according to runtime system conditions.



The system monitors \*\*inference latency, CPU utilization, and device temperature\*\* and switches between predefined operating modes to balance computational workload and real-time performance.



\## Demo



\### Project Walkthrough



A 1.5-minute live demonstration explaining the EdgeAdapt system, its adaptive behavior, and the overall implementation.



\*\*Video:\*\* `Videos/EdgeAdapt-Demo-Explained.mp4`



\### Adaptive Mode Demonstration



A short demonstration showing the system changing operating modes during runtime.



\*\*Video:\*\* `Videos/EdgeAdapt-Mode-Adaptation.mp4`



\### Hardware Setup



A short clip showing the Jetson Nano and camera setup used for the project.



\*\*Video:\*\* `Videos/Jetson-Nano-Camera-Setup.mp4`



\---



\## Overview



Real-time computer vision on edge devices requires balancing detection performance with limited computational resources.



EdgeAdapt explores an adaptive inference strategy for the NVIDIA Jetson Nano. Instead of continuously running the same inference workload, the system monitors the runtime state of the device and dynamically selects an operating mode.



The adaptive controller uses:



\* Inference latency

\* CPU utilization

\* Jetson Nano temperature



The selected operating mode controls:



\* Frame skipping

\* Detection confidence threshold



\*\*FPS is not used as an adaptive control signal.\*\* FPS and latency are instead recorded as performance metrics for evaluating the adaptive system against a non-adaptive baseline.



\---



\## Adaptive Control



EdgeAdapt uses four operating modes:



| Mode        | Frame Skipping | Confidence Threshold |

| ----------- | -------------: | -------------------: |

| QUALITY     |              0 |                 0.25 |

| BALANCED    |              1 |                 0.30 |

| PERFORMANCE |              2 |                 0.35 |

| COOLING     |              3 |                 0.40 |



The controller evaluates runtime conditions and selects a mode according to predefined thresholds.



\### Mode Selection Logic



```text

Temperature > 58°C

&#x20;       │

&#x20;       ▼

&#x20;   COOLING





Latency > 65 ms OR CPU > 80%

&#x20;       │

&#x20;       ▼

&#x20;  PERFORMANCE





Latency > 45 ms OR CPU > 40% OR Temperature > 55°C

&#x20;       │

&#x20;       ▼

&#x20;   BALANCED





Otherwise

&#x20;       │

&#x20;       ▼

&#x20;    QUALITY

```



A minimum mode duration is enforced before another transition can occur, helping prevent rapid switching between modes.



\### Adaptive Workload



As the system moves toward more performance-oriented modes, it increases frame skipping and raises the confidence threshold:



```text

QUALITY

&#x20; ↓

No frame skipping

Confidence = 0.25



BALANCED

&#x20; ↓

Skip 1 frame

Confidence = 0.30



PERFORMANCE

&#x20; ↓

Skip 2 frames

Confidence = 0.35



COOLING

&#x20; ↓

Skip 3 frames

Confidence = 0.40

```



The confidence threshold is smoothed over time rather than changing instantaneously.



\---



\## System Architecture



```text

&#x20;                   Camera Input

&#x20;                        │

&#x20;                        ▼

&#x20;               ┌─────────────────┐

&#x20;               │ YOLOv5n /       │

&#x20;               │ TensorRT        │

&#x20;               └────────┬────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;                Object Detection

&#x20;                        │

&#x20;                        ▼

&#x20;               ┌─────────────────┐

&#x20;               │ Runtime Metrics │

&#x20;               │                 │

&#x20;               │ Latency         │

&#x20;               │ CPU Utilization │

&#x20;               │ Temperature     │

&#x20;               └────────┬────────┘

&#x20;                        │

&#x20;                        ▼

&#x20;               ┌─────────────────┐

&#x20;               │ Adaptive Mode   │

&#x20;               │ Controller      │

&#x20;               └────────┬────────┘

&#x20;                        │

&#x20;             ┌──────────┼──────────┐

&#x20;             ▼          ▼          ▼

&#x20;          QUALITY    BALANCED   PERFORMANCE

&#x20;                                     │

&#x20;                                     ▼

&#x20;                                  COOLING



&#x20;                        │

&#x20;                        ▼

&#x20;                 Object Tracker

&#x20;                        │

&#x20;                        ▼

&#x20;                Detection Output

&#x20;                        │

&#x20;                        ▼

&#x20;                 FPS / Latency

&#x20;                    Logging

```



\---



\## Performance Evaluation



The adaptive implementation was evaluated against a non-adaptive baseline on the NVIDIA Jetson Nano.



The two implementations record FPS and latency measurements independently during execution.



\### Average Results



| Metric          | Non-Adaptive Baseline |    EdgeAdapt |

| --------------- | --------------------: | -----------: |

| Average FPS     |                  7.11 |    \*\*15.01\*\* |

| Average Latency |             142.94 ms | \*\*68.21 ms\*\* |



Based on the collected measurements:



\* EdgeAdapt achieved approximately \*\*2.11× the average FPS\*\* of the non-adaptive implementation.

\* Average latency was reduced by approximately \*\*52.3%\*\*.



These results are based on the measurements stored in the repository and are specific to the tested Jetson Nano hardware and experimental conditions.



\### Performance Comparison



!\[EdgeAdapt Performance Comparison](Images/EdgeAdapt-Performance-Comparison.jpg)



\### Raw Measurements



The original measurements are available in:



```text

Data/

├── adaptive\_fps.txt

├── adaptive\_latency.txt

├── baseline\_fps.txt

└── baseline\_latency.txt

```



\---



\## Implementation



\### EdgeAdapt



`Code/edgeadapt.py`



The adaptive implementation:



\* Captures frames from a camera using OpenCV

\* Runs YOLOv5n inference using TensorRT

\* Uses CUDA/PyCUDA for inference execution

\* Monitors CPU utilization using `psutil`

\* Reads Jetson Nano temperature information

\* Calculates runtime inference latency

\* Dynamically switches between four operating modes

\* Applies frame skipping

\* Adjusts the detection confidence threshold

\* Performs object tracking

\* Records FPS and latency measurements

\* Displays runtime FPS, latency, CPU, temperature, and current mode



\### Non-Adaptive Baseline



`Code/nonadapt.py`



The baseline implementation uses the same general detection and tracking pipeline but keeps the inference configuration fixed.



It continuously performs detection using:



\* Fixed confidence threshold

\* No adaptive mode switching

\* No adaptive frame skipping



This provides a reference point for evaluating the effect of the adaptive controller.



\### Object Tracking



`Code/tracker.py`



The project includes a lightweight centroid-distance-based tracker.



The tracker:



\* Assigns IDs to detected objects

\* Matches detections based on centroid distance

\* Maintains object identities across frames

\* Removes objects after they have been missing for a configurable number of frames



\### Analysis



`Code/adaptgraph.py`



This script contains the adaptive inference/analysis implementation used during development and experimentation.



\---



\## Technologies



| Technology         | Purpose                                            |

| ------------------ | -------------------------------------------------- |

| Python             | Application and control logic                      |

| OpenCV             | Camera capture, image processing, display, and NMS |

| YOLOv5n            | Object detection model                             |

| TensorRT           | Optimized inference execution                      |

| CUDA               | GPU acceleration                                   |

| PyCUDA             | CUDA memory and execution interface                |

| NumPy              | Numerical processing                               |

| psutil             | CPU utilization monitoring                         |

| NVIDIA Jetson Nano | Edge deployment platform                           |



\---



\## Hardware



The system was developed and evaluated on:



\* \*\*NVIDIA Jetson Nano\*\*

\* USB camera



The repository also includes photographs and video documenting the hardware setup.



\### Jetson Nano



!\[Jetson Nano](Images/JetsonNano.jpg)



\---



\## Software Environment



The project was evaluated in the following environment:



| Component | Version |

| --------- | ------- |

| JetPack   | R32.7.6 |

| CUDA      | 10.2    |

| TensorRT  | 8.2.1   |

| OpenCV    | 4.5.5   |

| NumPy     | 1.19.4  |



Additional environment information is available in:



```text

Docs/

├── cuda\_version.txt

├── jetpack\_version.txt

└── tensorrt\_version.txt

```



TensorRT installation information is also documented in:



`Docs/tensorrt\_version.txt`



\---



\## TensorRT Model



The repository includes the TensorRT engine used for deployment:



```text

Models/yolov5n.engine

```



The engine is approximately \*\*5.2 MB\*\*.



The model is based on \*\*YOLOv5n\*\* and is used through TensorRT for inference on the Jetson Nano.



TensorRT engine files can depend on the target hardware and software environment. The provided engine should therefore be considered specific to the documented Jetson Nano/TensorRT configuration.



\---



\## Repository Structure



```text

EdgeAdapt-Jetson-Nano/

│

├── README.md

├── .gitignore

├── LICENSE

│

├── Code/

│   ├── adaptgraph.py

│   ├── edgeadapt.py

│   ├── nonadapt.py

│   ├── requirements.txt

│   └── tracker.py

│

├── Data/

│   ├── adaptive\_fps.txt

│   ├── adaptive\_latency.txt

│   ├── baseline\_fps.txt

│   └── baseline\_latency.txt

│

├── Docs/

│   ├── cuda\_version.txt

│   ├── jetpack\_version.txt

│   └── tensorrt\_version.txt

│

├── Images/

│   ├── EdgeAdapt-Demo-Running.jpg

│   ├── EdgeAdapt-Performance-Comparison.jpg

│   ├── Jetson-Nano.jpg

│   ├── Jetson-Nano-Development-Kit.jpg

│   ├── Jetson-Nano-Development-Kit-Box.jpg

│   └── TensorRT-Version-Info.png

│

├── Misc/

│   └── EdgeAdapt-Project-Presentation.pptx

│

├── Models/

│   └── yolov5n.engine

│

└── Videos/

&#x20;   ├── EdgeAdapt-Demo-Explained.mp4

&#x20;   ├── EdgeAdapt-Mode-Adaptation.mp4

&#x20;   └── Jetson-Nano-Camera-Setup.mp4

```



\---



\## Running the Project



The project requires a compatible NVIDIA Jetson Nano environment with the required CUDA, TensorRT, PyCUDA, OpenCV, and Python dependencies.



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/EdgeAdapt-Jetson-Nano.git

cd EdgeAdapt-Jetson-Nano

```



\### Important



The included TensorRT engine is intended for the documented Jetson Nano environment.



Before running the scripts, ensure that the TensorRT engine is available at the location expected by the code and that the camera is connected.



\### Run EdgeAdapt



```bash

cd Code

python3 edgeadapt.py

```



\### Run the Non-Adaptive Baseline



```bash

cd Code

python3 nonadapt.py

```



Press \*\*Esc\*\* to exit the application.



> \*\*Note:\*\* The current scripts load `yolov5n.engine` from their working directory and write measurement files to the working directory. If running directly from `Code/`, the model path and output paths may need to be adjusted to match the repository structure.



\---



\## Visual Documentation



\### System Running



!\[EdgeAdapt Demo](Images/EdgeAdapt-Demo-Running.jpg)



\### Jetson Nano Development Kit



!\[Jetson Nano Development Kit](Images/Jetson-Nano-Development-Kit.jpg)



\### Development Kit Packaging



!\[Jetson Nano Development Kit Box](Images/Jetson-Nano-Development-Kit-Box.jpg)



\### TensorRT Environment



!\[TensorRT Version Information](Images/TensorRT-Version-Info.png)



Additional demonstration material is available in the `Videos/` directory.



\---



\## Limitations



\* Evaluation was performed on NVIDIA Jetson Nano hardware.

\* Performance depends on the camera input, workload, and system conditions.

\* TensorRT engine compatibility can depend on the target hardware and software environment.

\* The adaptive controller uses predefined thresholds for latency, CPU utilization, and temperature.

\* The current evaluation focuses on the tested workload and hardware configuration.

\* The lightweight tracker uses centroid-distance matching and is not intended to replace more advanced multi-object tracking algorithms.



\---



\## Future Work



Potential extensions include:



\* More sophisticated adaptive control policies

\* Evaluation across additional edge devices

\* Broader workload and scene evaluation

\* Power-consumption measurements

\* More advanced multi-object tracking

\* Dynamic optimization of additional inference parameters

\* Larger-scale performance evaluation under different thermal and computational conditions



\---



\## Project Status



\*\*Completed experimental project / research prototype\*\*



The repository contains the implementation, experimental measurements, deployment environment information, TensorRT model, visual documentation, and demonstrations.



\---



\## Author



\*\*Jhalak Sagar\*\*



EdgeAdapt was developed as an exploration of adaptive real-time computer vision and edge inference on NVIDIA Jetson Nano.



