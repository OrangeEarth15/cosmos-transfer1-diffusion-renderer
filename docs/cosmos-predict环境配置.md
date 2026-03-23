# Cosmos-Predict1 环境配置
---

检查当前环境:

```bash
nvidia-smi              # GPU和CUDA驱动
nvcc --version          # CUDA版本
df -h                   # 可用磁盘
```

---

## 安装步骤

### 1. 安装Miniconda (如未安装)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda init bash
source ~/.bashrc
```

### 2. 克隆仓库

```bash
mkdir -p ~/robotics && cd ~/robotics
git clone https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer.git
cd cosmos-transfer1-diffusion-renderer
```

### 3. 创建Conda环境

**方式1: YAML文件 (推荐)**

```bash
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
```

**方式2: 手动创建**

```bash
conda create -n cosmos-predict1 python=3.10 -y
conda activate cosmos-predict1
python --version  # 验证3.10.x
```

### 4. 安装Python依赖

```bash
conda activate cosmos-predict1
pip install -r requirements.txt
```

关键依赖:
- torch 2.6.0
- diffusers 0.32.2
- transformers 4.49.0
- megatron-core 0.10.0
- imageio[ffmpeg]

耗时: 5-10分钟

### 5. 配置Transformer Engine

创建符号链接解决头文件问题:

```bash
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* \
       $CONDA_PREFIX/include/

ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* \
       $CONDA_PREFIX/include/python3.10/

pip install 'transformer-engine[pytorch]==1.12.0'
```

验证:

```bash
python -c "import transformer_engine; print(transformer_engine.__version__)"
# 输出: 1.12.0
```

### 6. 安装nvdiffrast

标准安装:

```bash
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/triton/backends/nvidia/include/crt \
       $CONDA_PREFIX/include/

pip install git+https://github.com/NVlabs/nvdiffrast.git
```

如果编译失败 (Unknown CUDA arch):

```bash
# 查看GPU架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 手动指定架构 (示例: A100=8.0)
TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git
```

验证:

```bash
python -c "import nvdiffrast; print('OK')"
```

### 7. 修复OpenCV (必须)

问题: `opencv-python` 需要GUI库 (libGL.so.1), headless服务器没有。

错误:

```
ImportError: libGL.so.1: cannot open shared object file
```

解决:

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless==4.10.0.84
```

验证:

```bash
python -c "import cv2; print(cv2.__version__)"
```

`opencv-python-headless` 提供完整图像处理，无GUI模块。

### 8. 验证安装

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

成功输出:

```
[SUCCESS] torch found
[SUCCESS] diffusers found
[SUCCESS] transformers found
[SUCCESS] megatron.core found
[SUCCESS] transformer_engine found
[SUCCESS] nvdiffrast found
[SUCCESS] Cosmos environment setup is successful!
```

---

## 下载模型

### 配置Hugging Face

获取Token:
1. 访问 https://huggingface.co/settings/tokens
2. 创建新token (Read权限)
3. 复制token

登录:

```bash
pip install huggingface_hub
huggingface-cli login
# 粘贴token

# 验证
huggingface-cli whoami
```

### 下载权重 (~58GB)

```bash
cd ~/robotics/cosmos-transfer1-diffusion-renderer

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python \
    scripts/download_diffusion_renderer_checkpoints.py \
    --checkpoint_dir checkpoints
```

下载内容:

| 模型 | 大小 | 用途 |
|------|------|------|
| Diffusion_Renderer_Inverse_Cosmos_7B | 28GB | 逆向渲染 |
| Diffusion_Renderer_Forward_Cosmos_7B | 28GB | 正向渲染 |
| Cosmos-Tokenize1-CV8x8x8-720p | 1.8GB | VAE Tokenizer |

耗时: ~10分钟 (1Gbps网络)

### 处理Gated Repository

如果遇到403错误:

```
GatedRepoError: 403 Client Error
Access to model nvidia/Cosmos-Tokenize1-CV8x8x8-720p is restricted
```

解决:
1. 访问 https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p
2. 点击 "Agree and access repository"
3. 接受NVIDIA Open Model License
4. 重新运行下载命令

验证下载:

```bash
cd checkpoints
ls -lh

# 应包含:
# Diffusion_Renderer_Inverse_Cosmos_7B/
# Diffusion_Renderer_Forward_Cosmos_7B/
# Cosmos-Tokenize1-CV8x8x8-720p/

# 检查Tokenizer
ls Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt  # 必须存在
```

---

## 验证脚本

创建 `check_env.sh`:

```bash
#!/bin/bash
echo "=== 环境验证 ==="

# Python
python --version

# CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, {torch.version.cuda}')"

# GPU
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# 依赖
python -c "import torch, diffusers, transformers, transformer_engine, nvdiffrast, cv2"
echo "所有依赖已安装"

# 模型
[ -d checkpoints/Diffusion_Renderer_Inverse_Cosmos_7B ] && echo "✓ Inverse" || echo "✗ Inverse"
[ -d checkpoints/Diffusion_Renderer_Forward_Cosmos_7B ] && echo "✓ Forward" || echo "✗ Forward"
[ -d checkpoints/Cosmos-Tokenize1-CV8x8x8-720p ] && echo "✓ Tokenizer" || echo "✗ Tokenizer"
```

运行:

```bash
chmod +x check_env.sh
./check_env.sh
```

---

## 常见问题

### OpenCV libGL.so.1错误

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless==4.10.0.84
```

### nvdiffrast编译失败

```bash
# 查看架构
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 手动指定 (示例: A100)
TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git
```

### Transformer Engine导入失败

```bash
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* \
       $CONDA_PREFIX/include/

pip install --force-reinstall 'transformer-engine[pytorch]==1.12.0'
```

### Tokenizer 403错误

访问模型页面接受协议:
https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p

### CUDA_HOME未设置

```bash
export CUDA_HOME=$CONDA_PREFIX

# 永久设置
echo 'export CUDA_HOME=$CONDA_PREFIX' >> ~/.bashrc
```

---

## 实际配置记录


### 遇到的问题

**1. OpenCV GUI依赖**
- 错误: `libGL.so.1` 找不到
- 原因: Docker/headless环境无GUI库
- 解决: 安装headless版本

**2. 路径Bug** (已在上游修复)
- 文件: `inference_inverse_renderer.py:191-203`
- 问题: 路径前导斜杠导致 `os.path.join` 失效
- 错误: `PermissionError: '/0000.0000.basecolor.jpg'`

**3. Hugging Face认证**
- Token存储: `~/.cache/huggingface/token`
- 登录: `huggingface-cli login`
