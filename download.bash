echo "--- 1/8: Activating existing environment 'audio_clean' ---"
source activate audio_clean

rm -rf ~/.cache/huggingface/hub/

echo "Installing git-lfs..."
conda install -c conda-forge git-lfs -y
git lfs install

echo "--- 2/8: Installing Torch 2.8.0 / Torchaudio 2.8.0 (via --user) ---"
pip install torch==2.8.0 torchaudio==2.8.0 --user

echo "--- 3/8: Installing compatible TorchVision 0.23.0 (via --user) ---"
pip install torchvision==0.23.0 --user

echo "--- 4/8: Installing missing common Python dependencies (via --user) ---"
pip install packaging requests tqdm PyYAML rich joblib regex sentencepiece cycler fonttools kiwisolver pyparsing python-dateutil --user

echo "--- 5/8: Installing Pyannote audio (via --user) ---"
pip install pyannote-audio==3.4.0 --user

echo "--- 6/8: Installing omegaconf and pytorch-lightning (via --user) ---"
pip install omegaconf pytorch-lightning --user

echo "--- 7/8: Installing WhisperX from GitHub (via --user) ---"
pip install git+https://github.com/m-bain/whisperx.git --user

echo "--- 8/8: Running the processing pipeline ---"
echo "✅ Dependencies installed and synchronized. Starting script execution."
python process_pipeline.py
