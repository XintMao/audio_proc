# Automated Multi-Speaker Audio-Text Alignment Pipeline

This project provides a high-precision pipeline designed for processing complex multi-speaker radio broadcasts and interview audio, specifically optimized for Nigerian languages such as Yoruba, Igbo, and Hausa. It specializes in resolving complex conversational challenges including heavy background music (BGM), overlapping speech, rapid speaker turns, and temporal drift through a customized heuristic alignment engine.

## Key Components

* **Automatic Speech Recognition (ASR):** Utilizes `faster-whisper` (Large-v3) to generate precise word-level timestamps from raw recordings, exposing crucial Voice Activity Detection (VAD) hyperparameters to handle long pauses.
* **Advanced Speaker Diarization (SD):** Leverages `pyannote.audio` 3.1 to identify speaker turns and detect conversational boundaries with high temporal resolution.
* **Speaker Verification:** Integrates `WeSpeaker` (voxceleb backend) to extract unique vocal fingerprints, successfully mapping anonymous AI-detected Speaker IDs to the actual character identities in the text transcripts.
* **V77 Heuristic Post-Processing:** A custom-engineered logic layer designed to execute a 30-second elastic sliding search window, handle systemic background noise, and eliminate logical deadlocks during complex radio transitions.

---

## Setup and Installation

Due to specific dependency requirements between system modules and PyTorch, we recommend utilizing a dedicated virtual environment.

### Recommended Environment
* **Python:** 3.10
* **PyTorch:** 2.1.2 (Optimized for CUDA 11.8 stability)
* **GPU:** NVIDIA GPU with CUDA 11.8+ recommended

### Environment Setup

```bash
# Create and activate a dedicated Conda environment
conda create -n audio_alignment python=3.10 -y
conda activate audio_alignment

# Install dependencies using the requirements configuration file
pip install -r requirement.txt

# Execute the local initialization script to pull required model weights
bash download.bash
