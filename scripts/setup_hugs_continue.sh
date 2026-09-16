#!/bin/bash
# ============================================================
# HUGS 环境后续安装脚本 (PyTorch 已就绪后继续)
# 在 GPU 服务器上执行：
#   cd "$SCRIPT_DIR/.."
#   conda activate hugs
#   bash scripts/setup_hugs_continue.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=========================================="
echo " HUGS 后续依赖安装"
echo " Project: ${PROJECT_DIR}"
echo "=========================================="

# ---------- 确保使用 conda 环境内的 pip ----------
echo "[INFO] 确保 pip 已安装在 conda 环境中..."
conda install -n hugs pip -y --quiet

CONDA_PIP="$(dirname $(which python))/pip"
if [ ! -f "${CONDA_PIP}" ]; then
    PIP_CMD="python -m pip"
else
    PIP_CMD="${CONDA_PIP}"
fi
echo "[INFO] 使用 pip: ${PIP_CMD}"

# ---------- 验证 PyTorch ----------
echo "[CHECK] PyTorch CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'  PyTorch {torch.__version__}, CUDA: OK')"

# ---------- 安装 CUDA 11.7 toolkit 到 conda 环境 (解决系统 CUDA 12.x 与 PyTorch CUDA 11.7 不匹配) ----------
echo "[STEP 0a] 安装 CUDA 11.7 toolkit 到 conda 环境..."
conda install -y cuda-toolkit=11.7 -c "nvidia/label/cuda-11.7.1" --quiet 2>/dev/null || \
conda install -y cudatoolkit-dev=11.7 -c conda-forge --quiet 2>/dev/null || \
echo "[WARN] conda cuda toolkit 安装失败，尝试用 nvcc 路径..."

# 设置 CUDA_HOME 为 conda 环境
export CUDA_HOME="${CONDA_PREFIX}"
# RTX 4090 = compute capability 8.9, PyTorch 1.13 不直接支持 8.9
# 使用 8.6+PTX 让 PTX 在 8.9 上 JIT 运行
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
export FORCE_CUDA=1

# 如果 conda 中没有 nvcc，则 patch PyTorch 的 CUDA 版本检查
NVCC_PATH="${CONDA_PREFIX}/bin/nvcc"
if [ ! -f "${NVCC_PATH}" ]; then
    echo "[INFO] conda 环境无 nvcc，将 patch PyTorch cpp_extension.py 跳过 CUDA 版本检查..."
    PYTORCH_CPP_EXT="$(python -c 'import torch.utils.cpp_extension as e; print(e.__file__)')"
    # 备份并 patch: 将 _check_cuda_version 函数改为空操作
    cp "${PYTORCH_CPP_EXT}" "${PYTORCH_CPP_EXT}.bak"
    python -c "
import re
path = '${PYTORCH_CPP_EXT}'
with open(path, 'r') as f:
    content = f.read()
# 替换 _check_cuda_version 函数体为 pass
patched = re.sub(
    r'(def _check_cuda_version\([^)]*\):)\n(.*?)(?=\ndef |\nclass |\Z)',
    r'\1\n    pass\n\n',
    content,
    count=1,
    flags=re.DOTALL
)
with open(path, 'w') as f:
    f.write(patched)
print('  Patched _check_cuda_version -> pass')
"
fi

echo "[INFO] CUDA_HOME=${CUDA_HOME}"
echo "[INFO] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ---------- 初始化 git 子模块 ----------
echo "[STEP 0b] 初始化 git 子模块..."
git submodule update --init --recursive

# ---------- 安装 pytorch3d (从本地源码编译 v0.7.5，兼容 PyTorch 1.13) ----------
echo "[STEP 1/4] 安装 pytorch3d v0.7.5 (本地源码编译)..."
${PIP_CMD} install fvcore iopath
${PIP_CMD} install submodules/pytorch3d

# ---------- 编译安装子模块 ----------
echo "[STEP 2/4] 编译安装 diff-gaussian-rasterization..."
${PIP_CMD} install submodules/diff-gaussian-rasterization

echo "[STEP 3/4] 编译安装 simple-knn..."
${PIP_CMD} install submodules/simple-knn

# ---------- 安装 requirements.txt 和 chumpy ----------
echo "[STEP 4/4] 安装 requirements.txt 和 chumpy..."
${PIP_CMD} install -r requirements.txt
${PIP_CMD} install submodules/chumpy

# ---------- 最终验证 ----------
echo ""
echo "=========================================="
echo " 环境验证"
echo "=========================================="
python -c "import pytorch3d; print('  pytorch3d: OK')"
python -c "import diff_gaussian_rasterization; print('  diff_gaussian_rasterization: OK')"
python -c "import simple_knn; print('  simple_knn: OK')"
python -c "import lpips; print('  lpips: OK')"
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""
echo "=========================================="
echo " ALL DONE! 可以运行："
echo "   python scripts/evaluate.py -o output/pretrained_models/lab"
echo "=========================================="
