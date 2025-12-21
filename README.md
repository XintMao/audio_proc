**Armenian Speaker Diarization & Alignment Pipeline**

This project provides a high-precision pipeline for processing Armenian interview audio. It specializes in resolving complex conversational challenges such as overlapping speech, rapid speaker turns, and semantic repetitions through a customized heuristic alignment engine.

**Key Components**

- **Forced Alignment (FA):** Utilizes **WhisperX (large-v3)** and **Wav2Vec2.0** Armenian models to generate word-level timestamps from raw transcripts.
- **Advanced Speaker Diarization (SD):** Leverages **Pyannote.audio 3.1** to identify speaker turns with high temporal resolution.
- **V77 Heuristic Post-Processing:** A custom-engineered logic layer designed to:
  - Resolve speaker overlaps using intensity-ratio triggers.
  - Handle "Prompt-Repeat" patterns (e.g., interviewer reminds, guest repeats).
  - Apply "Hard-coded Anchoring" for micro-interjections like "ու" or "հա՞".
- **Refinement:** Ensures output follows a logical interview flow, maintaining speaker identity consistency over long durations.

**Setup and Installation**

Due to specific dependency requirements between WhisperX and PyTorch, we recommend a dedicated environment.

1. Recommended Environment
- **Python:** 3.10
- **PyTorch:** 2.1.2 (Optimized for CUDA 11.8 stability)
- **GPU:** NVIDIA GPU with CUDA 11.8+ recommended.

2. Environment Setup
```bash
# Create and activate a dedicated Conda environment
conda create -n armenian_diarization python=3.10 -y
conda activate armenian_diarization

# Install core dependencies with specific CUDA index
pip install -r requirements.txt --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Setup Hugging Face (Required for Pyannote model access)
pip install -U "huggingface-hub[cli]"
huggingface-cli login

###Repository Structure
.
├── data/                    # Input directory
│   ├── 01_kapital_29_09_25.mp3   # Source Armenian audio
│   └── 01_kapital_29_09_25.txt   # Complete transcript for alignment
├── output/                  # Generated results
│   └── 01_kapital_29_09_25_V77_FINAL_GOLD.txt # Final formatted output
├── process_pipeline.py      # Standard execution script
├── process_V77_GOLD.py      # Ultimate heuristic logic version (Recommended)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
