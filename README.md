# Real-Time Lip Reading: Translating Visual/Speech Cues to Text Using Deep Learning

An enhanced real-time multimodal speech recognition system that combines visual lip-reading with audio speech recognition to achieve higher accuracy than either modality alone.

## 🎯 Overview

This project implements an **Audio-Visual Speech Recognition (AVSR)** system that:

- **Visual Speech Recognition (VSR)**: Deep learning-based lip-reading using webcam video
- **Automatic Speech Recognition (ASR)**: Audio-based speech recognition using microphone
- **Intelligent Fusion**: Weighted confidence-based algorithm to combine both modalities
- **LLM Post-Processing**: Language model correction for final output

### Key Features

- ✅ Real-time lip-reading from webcam input
- ✅ Audio speech recognition integration
- ✅ Multimodal fusion algorithm for improved accuracy
- ✅ Handles viseme confusion (e.g., "bat" vs "pat" vs "mat")
- ✅ LLM-based text correction and formatting
- ✅ Support for MediaPipe and RetinaFace face detection

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Webcam and microphone
- 8GB+ RAM
- 3GB+ free disk space for models

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd LipReadEnhance
   ```

2. **Install system dependencies** (Linux/Ubuntu)
   ```bash
   sudo apt-get install portaudio19-dev ffmpeg libsndfile1
   ```

3. **Install Python packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (for LLM post-processing)
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.2
   ```

### Running the Application

#### Full Audio-Visual Mode
```bash
python main_with_audio.py
```

#### Visual-Only Mode
```bash
python main.py
```

#### Fusion Algorithm Demo (No Hardware Required)
```bash
python test_fusion.py
```

### Controls

- **Alt**: Start/Stop recording
- **Q**: Quit application

## 📁 Project Structure

```
RealTimeLipRead/
├── pipelines/              # Visual speech recognition pipeline
│   ├── pipeline.py        # Main inference pipeline
│   ├── model.py           # AVSR Transformer model
│   ├── data/              # Data loading and preprocessing
│   ├── detectors/         # Face detection (MediaPipe/RetinaFace)
│   └── metrics/           # Evaluation metrics
│
├── espnet/                # ESPnet framework components
│   ├── nets/              # Neural network architectures
│   ├── asr/               # ASR utilities
│   └── utils/             # Helper utilities
│
├── benchmarks/            # Pre-trained models
│   └── LRS3/             # LRS3 dataset models
│
├── fusion.py              # Audio-visual fusion algorithm
├── audio_module.py        # Audio capture and ASR
├── main_with_audio.py     # Enhanced main application
├── main.py                # Original visual-only system
├── test_fusion.py         # Fusion demo
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

### Model Configuration

Model configurations are specified in `.ini` files. Example structure:

```ini
[input]
modality=video
v_fps=25

[model]
v_fps=25
model_path=benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
model_conf=benchmarks/LRS3/models/LRS3_V_WER19.1/model.json
rnnlm=benchmarks/LRS3/language_models/lm_en_subword/model.pth
rnnlm_conf=benchmarks/LRS3/language_models/lm_en_subword/model.json

[decode]
beam_size=40
penalty=0.0
ctc_weight=0.1
lm_weight=0.3
```

### Fusion Parameters

The fusion algorithm can be configured in `fusion.py`:

```python
fusion = AudioVisualFusion(
    audio_weight=0.6,    # 60% weight to audio
    visual_weight=0.4    # 40% weight to visual
)
```

## 🧠 How It Works

### System Architecture

```
┌─────────────┐    ┌─────────────┐
│   Webcam    │    │ Microphone  │
│  (Video)    │    │   (Audio)   │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│ Visual SR   │    │  Audio SR   │
│  (ESPnet)   │    │ (Google API)│
└──────┬──────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                ▼
         ┌─────────────┐
         │   Fusion    │
         │  Algorithm  │
         └──────┬──────┘
                ▼
         ┌─────────────┐
         │  LLM Post-  │
         │  Processing │
         └─────────────┘
```

### Fusion Algorithm

The system uses a weighted confidence-based fusion algorithm:

1. **Similarity Calculation**: Computes Jaccard similarity between visual and audio predictions
2. **Weighted Scoring**: Applies modality-specific weights (audio: 60%, visual: 40%)
3. **Decision Logic**:
   - **High agreement** (>0.8): Choose prediction with higher weighted score
   - **Medium agreement** (0.3-0.8): Word-level fusion
   - **Low agreement** (<0.3): Fallback to more reliable modality

## 📊 Performance

### Model Performance

- **Visual Model**: 19.1% WER on LRS3 test set
- **Expected Fused Performance**: 3-4% WER in clean conditions
- **Model Size**: ~1.77 GB
- **Inference Speed**: ~0.5s per utterance (GPU), ~2s (CPU)

### Fusion Benefits

| Condition | Visual WER | Audio WER | Fused WER |
|-----------|-----------|-----------|-----------|
| Clean | 19.1% | 5% | **3-4%** |
| Moderate noise | 19.1% | 15% | **12-14%** |
| High noise | 19.1% | 40% | **18-20%** |
| Low light | 35% | 5% | **6-8%** |

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, ESPnet
- **Computer Vision**: OpenCV, MediaPipe, RetinaFace
- **Audio Processing**: sounddevice, SpeechRecognition
- **LLM**: Ollama (LLaMA 3.2)
- **Configuration**: Hydra
- **Other**: NumPy, SciPy, scikit-image

## 📝 Usage Example

```python
from pipelines.pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    config_filename="configs/LRS3_V_WER19.1.ini",
    detector="retinaface",
    device="cuda:0"
)

# Run inference
result = pipeline(video_path="path/to/video.mp4")
print(f"Transcription: {result}")
```

## 🐛 Troubleshooting

### Common Issues

1. **No webcam/microphone detected**
   - Check device permissions
   - Verify hardware connections
   - Try different device indices in code

2. **CUDA out of memory**
   - Reduce batch size
   - Use CPU mode: `device="cpu"`
   - Close other GPU applications

3. **Model files not found**
   - Ensure model files are in `benchmarks/LRS3/`
   - Check config file paths

4. **Audio recognition fails**
   - Check microphone permissions
   - Verify internet connection (for Google Speech API)
   - Test microphone with system tools

## 📚 Documentation

For detailed technical documentation, see:
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Comprehensive technical overview

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes. Pre-trained models are subject to their original licenses (ESPnet, LRS3).

## 🙏 Acknowledgments

- **ESPnet**: Open-source speech processing toolkit
- **LRS3 Dataset**: University of Oxford lip-reading dataset
- **Ollama**: Local LLM inference engine
- **MediaPipe**: Google's ML solutions for face detection

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Version**: 1.0  
**Last Updated**: November 2025

