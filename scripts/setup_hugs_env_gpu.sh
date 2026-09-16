#!/bin/bash
# ============================================================
# HUGS 环境一键部署脚本 (GPU 服务器用)
# 在 GPU 服务器上执行：
#   cd "$SCRIPT_DIR/.."
#   bash scripts/setup_hugs_env_gpu.sh
# ============================================================

set -e

# ---------- 路径配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$SCRIPT_DIR"/miniconda3"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="hugs"

echo "=========================================="
echo " HUGS Environment Setup"
echo " Conda: ${CONDA_DIR}"
echo " Project: ${PROJECT_DIR}"
echo "=========================================="

# ---------- 初始化 conda ----------
eval "$(${CONDA_DIR}/bin/conda shell.bash hook)"

# ---------- 检查是否已存在 hugs 环境 ----------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] 环境 '${ENV_NAME}' 已存在，跳过创建。"
else
    echo "[STEP 1/6] 创建 conda 环境: ${ENV_NAME} (Python 3.8)..."
    conda create -n ${ENV_NAME} python=3.8 -y
fi

conda activate ${ENV_NAME}
echo "[INFO] 当前 Python: $(which python) ($(python --version))"

# ---------- 安装 PyTorch ----------
echo "[STEP 2/6] 安装 PyTorch 1.13.1 + CUDA 11.7..."
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# ---------- 验证 PyTorch + CUDA ----------
echo "[CHECK] PyTorch CUDA 验证..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ---------- 安装 pytorch3d ----------
echo "[STEP 3/6] 安装 pytorch3d..."
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# ---------- 编译安装 diff-gaussian-rasterization 和 simple-knn ----------
echo "[STEP 4/6] 编译安装 diff-gaussian-rasterization..."
cd "${PROJECT_DIR}"
pip install submodules/diff-gaussian-rasterization

echo "[STEP 5/6] 编译安装 simple-knn..."
pip install submodules/simple-knn

# ---------- 安装 requirements.txt 和 chumpy ----------
echo "[STEP 6/6] 安装 requirements.txt 和 chumpy..."
pip install -r requirements.txt
pip install git+https://github.com/mattloper/chumpy.git

# ---------- 最终验证 ----------
echo ""
echo "=========================================="
echo " 环境验证"
echo "=========================================="
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import pytorch3d; print('  pytorch3d: OK')"
python -c "import diff_gaussian_rasterization; print('  diff_gaussian_rasterization: OK')"
python -c "import simple_knn; print('  simple_knn: OK')"
python -c "import lpips; print('  lpips: OK')"
echo ""
echo "=========================================="
echo " 部署完成！"
echo " 使用方式："
echo "   eval \"\$(${CONDA_DIR}/bin/conda shell.bash hook)\""
echo "   conda activate ${ENV_NAME}"
echo "   cd ${PROJECT_DIR}"
echo "   python scripts/evaluate.py -o output/pretrained_models/lab"
echo "=========================================="
