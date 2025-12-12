#!/bin/bash
# 脚本名称: install_and_run_final.sh
# 目标: 激活已创建的环境，解决 H100 权限/依赖冲突，并运行 process_pipeline.py

# =======================================================
# 步骤 0: 激活已创建的环境
# =======================================================

echo "--- 1/8: Activating existing environment 'audio_clean' ---"
# 激活环境
source activate audio_clean

# 移除旧的 Hugging Face 缓存
rm -rf ~/.cache/huggingface/hub/

# 2. 安装 Git LFS (如果之前在 base 环境中安装过，这一步是必要的环境隔离)
echo "Installing git-lfs..."
conda install -c conda-forge git-lfs -y
git lfs install

# =======================================================
# 步骤 1: 强制安装 PyTorch 核心栈 (解决版本和权限问题)
# =======================================================

# 3. 安装 PyTorch 2.8.0 (WhisperX 的目标版本) 和 Torchaudio
echo "--- 2/8: Installing Torch 2.8.0 / Torchaudio 2.8.0 (via --user) ---"
# 所有 pip 命令都添加 --user 来绕过系统权限问题
pip install torch==2.8.0 torchaudio==2.8.0 --user

# 4. 安装兼容的 TorchVision 0.23.0
echo "--- 3/8: Installing compatible TorchVision 0.23.0 (via --user) ---"
pip install torchvision==0.23.0 --user

# =======================================================
# 步骤 2: 安装所有 AI/对齐库 (解决所有依赖冲突)
# =======================================================

# 5. 强制安装所有缺失的通用依赖
echo "--- 4/8: Installing missing common Python dependencies (via --user) ---"
# 确保安装了所有的通用库，避免依赖解析器冲突
pip install packaging requests tqdm PyYAML rich joblib regex sentencepiece cycler fonttools kiwisolver pyparsing python-dateutil --user

# 6. 安装 Pyannote 依赖核心
echo "--- 5/8: Installing Pyannote audio (via --user) ---"
pip install pyannote-audio==3.4.0 --user

# 7. 安装 Lightning/Conf 所需的依赖
echo "--- 6/8: Installing omegaconf and pytorch-lightning (via --user) ---"
pip install omegaconf pytorch-lightning --user

# 8. 安装 WhisperX
echo "--- 7/8: Installing WhisperX from GitHub (via --user) ---"
pip install git+https://github.com/m-bain/whisperx.git --user

# =======================================================
# 步骤 3: 运行你的主脚本
# =======================================================
echo "--- 8/8: Running the processing pipeline ---"
echo "✅ Dependencies installed and synchronized. Starting script execution."
python process_pipeline.py