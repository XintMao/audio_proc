ENV_NAME="stable_armenian_env"
PYTHON_VERSION="3.10"

if conda info --envs | grep -q "${ENV_NAME}"; then
    echo "Deleting existing environment ${ENV_NAME}..."
    conda remove -n ${ENV_NAME} --all -y
fi

echo "--- 1/3: Creating stable Conda environment (${PYTHON_VERSION}) ---"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
source activate ${ENV_NAME}

PT_INDEX_URL="https://download.pytorch.org/whl/cu118"
WHISPERX_VERSION="3.7.4"
PYANNOTE_VERSION="3.4.0"

echo "--- 2/3: Installing PyTorch Core Dependencies (Stable 2.7.1 + cu118) ---"
pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 --index-url ${PT_INDEX_URL}

echo "--- 3/3: Installing WhisperX, Pyannote, and Utilities ---"
pip install whisperx==${WHISPERX_VERSION} \
            pyannote.audio==${PYANNOTE_VERSION} \
            transformers \
            librosa \
            soundfile \
            -i https://pypi.org/simple

echo "--- Final Step: Run Hugging Face Login ---"
pip install -U 'huggingface-hub[cli]'
huggingface-cli login

echo "Environment ${ENV_NAME} setup complete. Run 'python process_pipeline.py' after successful login."
