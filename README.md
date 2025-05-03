# Facial Recognition Surveillance System

<div align="center">
  <img src="docs/logo.png" alt="Face Recognition System Logo" width="200">
  <h3>Advanced Real-time Facial Recognition System with Multi-feature Optimization</h3>
  <p>
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/opencv-4.5.4+-green.svg" alt="OpenCV">
    <img src="https://img.shields.io/badge/insightface-0.6.2+-yellow.svg" alt="InsightFace">
    <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg" alt="License">
  </p>
</div>

## 🚀 Key Features

- **Real-time Face Detection & Recognition**: Powered by RetinaFace and ArcFace models
- **One-shot Learning**: Recognize individuals from a single reference image
- **Memory-Augmented Recognition**: Short and long-term memory systems for improved accuracy
- **Dynamic Resolution Scaling**: Optimizes processing speed without compromising quality
- **ROI Management**: Intelligent region-of-interest focusing for efficient processing
- **Advanced Tracking**: ByteTrack-based face tracking across frames
- **Embedding Caching**: Reduces computational overhead by 60-85%
- **Incremental Learning**: Continuously improves recognition accuracy
- **Multi-camera Support**: Handle multiple video streams simultaneously
- **Real-time Performance**: 25-30 FPS on RTX 3060 with all optimizations

## 📑 Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Options](#configuration-options)
- [Database Management](#database-management)
- [Technical Architecture](#technical-architecture)
- [Recognition Performance](#recognition-performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux Ubuntu 18.04+, macOS 10.15+
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: NVIDIA GPU with 4GB VRAM (optional but recommended for better performance)
- **Python**: 3.8+
- **Storage**: 5GB free space

### Recommended Hardware (for optimal performance)
- **CPU**: AMD Ryzen 7 5800H or equivalent
- **GPU**: NVIDIA RTX 3060 or better
- **RAM**: 16GB DDR4
- **Storage**: SSD with 20GB free space

## 🛠️ Installation

### Windows Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/facial-surveillance-system.git
   cd facial-surveillance-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Mac Installation

1. **Install Homebrew and dependencies**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.9 cmake opencv
   ```

2. **Follow steps 1-3 from Windows installation**

### GPU Support (Optional)

For NVIDIA GPU acceleration:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

## 🚀 Quick Start

### 1. Prepare Face Database

Organize your face data in the following structure:
```
face_data/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
```

### 2. Build Database

```bash
python build_database.py
```

### 3. Run the System

**For webcam:**
```bash
python main.py --source 0
```

**For video file:**
```bash
python main.py --source path/to/video.mp4 --save
```

### 4. Keyboard Controls

- `q`: Quit application
- `d`: Toggle debug mode
- `s`: Save current frame

## ⚙️ Configuration Options

### System Configuration (`config/system_config.py`)

```python
SYSTEM_CONFIG = {
    'det_size': (640, 640),        # Detection input size
    'recognition_threshold': 0.35,  # Recognition confidence threshold
    'detection_interval': 3,        # Process every N frames
    'max_faces_per_frame': 20,     # Maximum faces to process
    'log_level': 'INFO',
}
```

### Feature Flags

```python
FEATURE_FLAGS = {
    'use_dynamic_resolution': True,
    'use_embedding_cache': True,
    'use_memory_system': True,
    'use_roi_management': True,
    'use_tracking': True,
    'use_incremental_learning': True,
}
```

### Recognition Optimization

```python
RECOGNITION_CONFIG = {
    'threshold': 0.35,     # Lower for one-shot learning
    'embedding_size': 512,
}
```

## 📊 Database Management

### Building Database

```bash
# Build from directory
python build_database.py

# Rebuild existing database
python main.py --build-db
```

### Managing Identities

```python
from database.face_database import FaceDatabase

# Add new person
db = FaceDatabase(config, app)
db.add_face("person_name", embedding)

# Remove person
db.remove_face("person_name")
```

### Database Validation

```bash
python validate_database.py
```

## 🏗️ Technical Architecture

### Pipeline Overview

```mermaid
graph LR
    A[Video Input] --> B[Frame Manager]
    B --> C[Dynamic Resolution]
    C --> D[ROI Manager]
    D --> E[Face Detection]
    E --> F[Face Tracking]
    F --> G[Embedding Cache]
    G --> H[Face Recognition]
    H --> I[Memory Systems]
    I --> J[Visualization]
    J --> K[Output]
```

### Component Descriptions

| Component | Purpose | Performance Impact |
|-----------|---------|-------------------|
| **Dynamic Resolution** | Reduces processing load by scaling background | 50-70% faster |
| **ROI Management** | Focuses processing on active regions | 40-60% less computation |
| **Embedding Cache** | Reuses face embeddings | 60-85% fewer computations |
| **Memory Systems** | Improves recognition consistency | 15-25% better accuracy |
| **Tracking** | Maintains identity across frames | Enables feature integration |

## 📈 Recognition Performance

### Benchmark Results (RTX 3060)

| Scenario | FPS | Accuracy | Latency |
|----------|-----|----------|---------|
| Single face, optimal lighting | 30 FPS | 92-97% | 25ms |
| Multiple faces (3-5) | 20-25 FPS | 85-92% | 40ms |
| Poor lighting | 15-20 FPS | 75-85% | 50ms |
| Crowded scene (10+ faces) | 10-15 FPS | 70-80% | 67ms |

### Accuracy Metrics

- **One-shot recognition**: 60-70% (initial)
- **After incremental learning**: 75-85%
- **Cross-camera recognition**: 65-75%
- **Partial occlusion handling**: 70-80%

## 🔧 Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check camera permissions
# Windows: Settings > Privacy > Camera
# Mac: System Preferences > Security & Privacy > Camera
```

#### 2. Low Recognition Accuracy
- Verify database images are high quality
- Check lighting conditions match training data
- Lower recognition threshold in config

#### 3. Performance Issues
```bash
# Check system resources
python utils/performance_monitor.py

# Enable only essential features
# Edit FEATURE_FLAGS in system_config.py
```

#### 4. GPU Not Detected
```bash
# Verify GPU installation
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Reinstall GPU runtime
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### Debugging Tools

```bash
# Run with debug logging
python main.py --source 0 --debug

# Component testing
python test_detection_visualization.py
python test_recognition_accuracy.py
python test_memory.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Install development dependencies (`pip install -r requirements-dev.txt`)
4. Run tests (`python -m pytest tests/`)
5. Commit changes (`git commit -m 'Add AmazingFeature'`)
6. Push to branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Code Standards

- Follow PEP 8 guidelines
- Add type hints for new functions
- Include docstrings for all classes and methods
- Write unit tests for new features

## 🙏 Acknowledgments

- **InsightFace Team**: For providing excellent face recognition models
- **OpenCV Community**: For the comprehensive computer vision library
- **ByteTrack Authors**: For the efficient tracking algorithm
- **FAISS Developers**: For the fast similarity search library

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support, please:
- Open an issue on GitHub
- Contact: support@yourproject.com
- Documentation: [Wiki](https://github.com/yourusername/facial-surveillance-system/wiki)

## 🔄 Version History

- **v1.0.0** - Initial release
  - Core face detection and recognition
  - Basic tracking implementation
- **v1.1.0** - Performance optimizations
  - Added dynamic resolution scaling
  - Implemented ROI management
- **v1.2.0** - Enhanced features
  - Memory-augmented recognition
  - Incremental learning support

---

<div align="center">
  Made with ❤️ for advancing facial recognition technology
</div>
