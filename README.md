### Key Components
- **Forced Alignment (FA):** Uses WhisperX (large-v2) and Wav2Vec2.0 to generate word-level timestamps from the input transcript.
- **Segmentation:** Re-segments alignment results by a maximum duration threshold (30.0s), ensuring cuts only occur at word boundaries.
- **Speaker Diarization (SD):** Leverages Pyannote.audio to identify and label speaker turns across the full audio file.
- **Post-Processing:** Merges aligned text segments with speaker diarization results to produce the final formatted output.

## Setup and Installation
Due to known **dependency conflicts** between WhisperX and recent PyTorch versions, a dedicated, stable environment is required for reliable execution.

### 1. Recommended Environment
- Python Version: 3.10 (tested and verified for compatibility)
- PyTorch Version: 2.7.1 (CUDA 11.8 base, optimized for stability)

### 2. Environment Setup
```bash
# Create a dedicated Conda environment
conda create -n stable_armenian_env python=3.10 -y

# Activate the environment
conda activate stable_armenian_env

### Install Dependencies
Use the provided requirements.txt with the specified PyTorch index URL to avoid version conflicts:
# Install core dependencies (PyTorch, Torchaudio, WhisperX, etc.)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Install and upgrade Hugging Face Hub CLI (for Pyannote model access)
pip install -U 'huggingface-hub[cli]'

# Log in to Hugging Face (required for Pyannote model download)
huggingface-cli login

###Repository Structure
.
├── data/                  # Input directory (place your files here)
│   ├── 01281_ARM.mp3      # Example source audio file (Armenian)
│   └── 01281_ARM.txt      # Example complete transcript (matching the audio)
├── output/                # Auto-created output directory
│   ├── [FILENAME]_alignment.json        # Raw word-level alignment results
│   ├── [FILENAME]_resegmented_alignment.json  # Duration-based segmented results
│   ├── [FILENAME]_diarization.rttm      # Speaker diarization results
│   └── [FILENAME]_final_transcript_formatted.txt  # Final output (time-coded + speaker labels)
├── process_pipeline.py    # Main execution script
└── requirements.txt       # Full list of project dependencies
