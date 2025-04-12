<p align="center">
  <img src="https://github.com/user/facial_surveillance_system/raw/main/docs/images/system_preview.png" alt="Facial Recognition System Preview" width="800"/>
</p>

A comprehensive, high-performance facial recognition surveillance system built for real-time identification from multiple video streams. Designed for one-shot learning scenarios and optimized for edge deployment on consumer-grade hardware.

📋 Key Features

One-shot Learning: Recognition from a single reference image per individual
Memory-Augmented Recognition: Maintains identity consistency in challenging conditions
Dynamic Resolution Processing: 2-3× performance boost with background optimization
Region of Interest Management: Focuses computational resources on high-value areas
Embedding Delta Caching: Reduces redundant computation by 60-85%
Lightweight Tracking: Efficiently track faces through video frames
Incremental Learning: System improves over time with collected evidence
FAISS GPU Acceleration: Ultra-fast search across large identity databases
Multi-camera Support: Efficiently process multiple video streams

📊 Performance Metrics
ConfigurationFPSMax FacesCPU UsageGPU Memory720p, Low Density25-301530-40%~1.5GB1080p, Low Density15-251040-50%~2.0GB720p, High Density15-202040-60%~2.5GB1080p, High Density10-151550-70%~3.0GBMulti-camera (4×720p)10-155-8 per camera60-80%~4.0GB
Recognition Accuracy:

Face Detection Rate: 95-98% (in good lighting conditions)
Identity Match Rate: 85-95% (with memory augmentation)
False Positive Rate: <2% (with threshold 0.35)
One-shot Learning: Effective with a single reference photo per individual

Benchmarked on ASUS ROG Strix G513QM with RTX 3060 6GB, AMD Ryzen 7 5800H, 16GB RAM
🏗️ System Architecture
The system employs a sophisticated multi-layer pipeline architecture with feedback loops for optimization:
<p align="center">
  <img src="https://github.com/user/facial_surveillance_system/raw/main/docs/images/architecture.png" alt="System Architecture" width="700"/>
</p>
Key components include:

Face Detection: Based on RetinaFace with InsightFace implementation
Face Recognition: ArcFace embedding comparison with custom enhancements
Memory Systems: Short and medium-term memory for recognition persistence
Tracking: Modified ByteTrack implementation for lightweight identity maintenance
Optimization: Dynamic resolution scaling, ROI management, and embedding caching
Database: FAISS-powered embedding database with incremental learning capability

📚 Literature & Technical Background
This project implements several cutting-edge techniques from recent research:
Key Technologies

RetinaFace (Deng et al., 2019) - Single-stage face detector that simultaneously predicts face locations and landmarks, achieving state-of-the-art performance on WIDER FACE dataset.
ArcFace (Deng et al., 2019) - Additive angular margin loss that enhances the discriminative power of facial recognition. Our implementation leverages the pre-trained models from InsightFace.
FAISS (Johnson et al., 2019) - Facebook AI Similarity Search library for efficient similarity search and clustering of dense vectors, enabling fast identity matching against large databases.
Memory-Augmented Neural Networks - Our approach incorporates principles from memory augmentation to improve recognition in challenging scenarios, similar to work by Santoro et al. (2016).
ByteTrack (Zhang et al., 2022) - Modified for face tracking to associate identities across frames, with optimizations for computational efficiency.

Innovations
Our implementation introduces several enhancements to standard techniques:

Embedding Delta Caching: Reduces redundant embedding computations by 60-85%
Dynamic Resolution Processing: Processes background at lower resolution while maintaining high resolution for faces
Memory Augmentation: Short and medium-term memory systems maintain identity despite challenging conditions

📦 Installation
Prerequisites

Python 3.9+
CUDA 11.2+ and cuDNN 8.1+ (for GPU acceleration)
OpenCV 4.5+

Setup Instructions
bash# Clone the repository
git clone https://github.com/username/facial_surveillance_system.git
cd facial_surveillance_system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Hardware Requirements

Minimum:

CPU: Intel i5-9400 / AMD Ryzen 5 3600 or equivalent
RAM: 8GB
GPU: NVIDIA GTX 1650 or equivalent with CUDA support
Storage: 5GB free space


Recommended:

CPU: Intel i7-10700 / AMD Ryzen 7 5800 or better
RAM: 16GB
GPU: NVIDIA RTX 3060 or better with 6GB+ VRAM
Storage: 10GB SSD



🚀 Usage
Building the Database
Before running the system, you need to organize your face database and build the recognition database:

Organize your face data with the following structure:
face_data/
├── Person1/
│   ├── image1.jpg
│   ├── image2.jpg (optional)
│   └── ...
├── Person2/
│   ├── image1.jpg
│   └── ...
└── ...

Update the database path in config/system_config.py:
pythonDATABASE_CONFIG = {
    'database_dir': "/path/to/your/face_data",
    'use_faiss': True,
    'index_type': 'flat',
}

Build the database:
bashpython build_database.py


Running the System
bash# Running with webcam (camera index 0)
python main.py --source 0

# Running with video file
python main.py --source path/to/video.mp4 --save

# Running with RTSP stream
python main.py --source "rtsp://username:password@192.168.1.64:554/stream"

# Running with multiple cameras defined in config file
python main.py --config config/cameras.json
Keyboard Controls
When the application is running:

q: Quit application
d: Toggle debug overlay
s: Save current frame
r: Force rebuild database

⚙️ Configuration
The system behavior can be customized through the configuration files in the config/ directory:
Key Configuration Options
system_config.py

Feature flags for enabling/disabling components
System-wide parameters like detection thresholds
GPU and memory settings

detection_config.py

Face detection parameters
ROI management configuration
Resolution scaling settings

recognition_config.py

Recognition thresholds and parameters
Database configuration
Embedding cache settings

tracking_config.py

Tracking parameters
Re-identification configuration
Track management settings

📂 Project Structure
facial_surveillance_system/
├── config/
│   ├── system_config.py        # System-wide configuration
│   ├── detection_config.py     # RetinaFace configuration
│   ├── recognition_config.py   # ArcFace configuration
│   └── tracking_config.py      # Tracking parameters
├── models/
│   ├── detection/              # Face detection module
│   │   ├── retinaface.py       # RetinaFace implementation
│   │   └── quantized_models.py # INT8 quantized models
│   ├── recognition/            # Face recognition module
│   │   ├── arcface.py          # ArcFace implementation
│   │   └── quantized_models.py # Optimized recognition models
│   └── optimization/           # Model optimization utilities
│       ├── model_quantizer.py  # INT8 quantization 
│       └── model_pruning.py    # Channel pruning implementation
├── core/
│   ├── engine.py               # Main processing engine
│   ├── pipeline.py             # Processing pipeline coordinator
│   ├── frame_manager.py        # Frame acquisition and management
│   └── alert_system.py         # Alert generation system
├── memory/
│   ├── short_term_memory.py    # Recent recognitions storage
│   ├── medium_term_memory.py   # Identity variations storage
│   ├── embedding_cache.py      # Delta caching implementation
│   └── memory_query.py         # Unified memory query system
├── tracking/
│   ├── byte_tracker.py         # Lightweight tracking implementation
│   ├── reid_manager.py         # Re-identification manager
│   ├── track_manager.py        # Track lifecycle management
│   └── track_priority.py       # Track priority calculation
├── optimization/
│   ├── dynamic_resolution.py   # Dynamic resolution scaling
│   ├── roi_manager.py          # Region of interest management
│   ├── attention_manager.py    # Attention-based prioritization
│   └── frame_priority.py       # Frame prioritization
├── database/
│   ├── face_database.py        # Face database management
│   ├── faiss_index.py          # GPU-accelerated indexing
│   └── incremental_learning.py # Database updating
├── utils/
│   ├── visualization.py        # Visualization utilities
│   ├── performance_monitor.py  # System performance monitoring
│   ├── histogram_utils.py      # Histogram calculation utilities
│   └── warmup.py               # Model warm-up utilities
├── main.py                     # Main entry point
├── camera_manager.py           # Camera input management
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
🔍 Future Work

Multi-modal Recognition: Integrate gait recognition for improved tracking in challenging scenarios
Active Learning: Human-in-the-loop verification for uncertain identifications
Edge-Cloud Hybrid: Distribute workload between edge devices and central server
Anomaly Detection: Unusual behavior identification and crowd pattern analysis
Cross-Camera Tracking: Enhanced identity consistency across multiple camera views
Mobile Interface: Real-time alert system with mobile notification capabilities

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🔗 References

Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4690-4699).
Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I., & Zafeiriou, S. (2019). RetinaFace: Single-stage Dense Face Localisation in the Wild. arXiv preprint arXiv:1905.00641.
Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. IEEE Transactions on Big Data.
Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. In Proceedings of the European Conference on Computer Vision (ECCV).
Santoro, A., Bartunov, S., Botvinick, M., Wierstra, D., & Lillicrap, T. (2016). Meta-learning with memory-augmented neural networks. In International Conference on Machine Learning (ICML) (pp. 1842-1850).
