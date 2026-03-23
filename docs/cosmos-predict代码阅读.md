# Cosmos DiffusionRenderer 代码架构解读

---

## 目录

- [核心概念](#核心概念)
- [整体架构](#整体架构)
- [核心组件](#核心组件)
- [数据流分析](#数据流分析)
- [模型加载机制](#模型加载机制)
- [代码实现细节](#代码实现细节)

---

## 核心概念

### 什么是DiffusionRenderer？

DiffusionRenderer = **VAE Encoder + Diffusion Transformer + VAE Decoder**

```
Input Video (RGB)
    ↓
┌──────────────────┐
│  VAE Encoder     │  压缩到latent空间 (8×8×8压缩)
└──────────────────┘
    ↓
Latent [B, 16, T/8, H/8, W/8]
    ↓
┌──────────────────┐
│ Diffusion DiT    │  扩散采样（50步）
│ (7B参数)         │  条件：context_index或G-buffers+HDRI
└──────────────────┘
    ↓
Generated Latent
    ↓
┌──────────────────┐
│  VAE Decoder     │  解压到像素空间
└──────────────────┘
    ↓
Output (G-buffers或Relit Video)
```

### 两阶段渲染

**阶段1: Inverse Rendering (逆向渲染 / 去光照)**
```
Input: RGB视频
Output: G-buffers (basecolor, normal, depth, roughness, metallic)
目的: 估计与光照无关的材质属性
```

**阶段2: Forward Rendering (正向渲染 / 重光照)**
```
Input: G-buffers + HDRI环境光
Output: 重光照的RGB视频
目的: 在新光照下合成视频
```

### G-Buffer详解

| G-Buffer | 物理含义 | 范围 | 通道数 | 用途 |
|----------|----------|------|--------|------|
| **basecolor** | 基色/反照率 | [0, 1]³ | 3 (RGB) | 材质固有颜色 |
| **normal** | 表面法线 | [-1, 1]³ | 3 (XYZ) | 表面朝向 |
| **depth** | 深度 | [0, 1] | 1 | 相机距离 |
| **roughness** | 粗糙度 | [0, 1] | 1 | 表面光滑度 |
| **metallic** | 金属度 | [0, 1] | 1 | 金属vs电介质 |

**为什么是5个？**
- 基于PBR (Physically Based Rendering)
- 符合Disney BRDF模型
- 足够重建真实材质

---

## 整体架构

### 目录结构

```
cosmos-transfer1-diffusion-renderer/
├── cosmos_predict1/                    # 核心Python包
│   ├── diffusion/                      # 扩散模型
│   │   ├── inference/                  # 推理脚本⭐
│   │   │   ├── inference_inverse_renderer.py     # 逆向渲染
│   │   │   ├── inference_forward_renderer.py     # 正向渲染
│   │   │   └── diffusion_renderer_pipeline.py    # 管线封装
│   │   ├── model/                      # 模型定义⭐
│   │   │   └── model_diffusion_renderer.py       # 主模型类
│   │   ├── networks/                   # 网络架构⭐
│   │   │   └── general_dit_diffusion_renderer.py # DiT网络
│   │   └── config/                     # 配置文件
│   │
│   ├── tokenizer/                      # VAE Tokenizer⭐
│   │   ├── cosmos_cv_tokenizer.py      # Tokenizer类
│   │   └── pretrained_ckpt_paths.py    # 预训练路径
│   │
│   ├── autoregressive/                 # 自回归模型（未使用）
│   ├── auxiliary/                      # 辅助工具
│   └── utils/                          # 工具函数
│
├── checkpoints/                        # 模型权重 (57.8GB)
│   ├── Diffusion_Renderer_Inverse_Cosmos_7B/     # 逆向模型
│   ├── Diffusion_Renderer_Forward_Cosmos_7B/     # 正向模型
│   └── Cosmos-Tokenize1-CV8x8x8-720p/            # VAE
│
├── scripts/                            # 辅助脚本
│   ├── download_diffusion_renderer_checkpoints.py
│   └── dataproc_extract_frames_from_video.py
│
└── asset/examples/                     # 测试数据
    ├── hdri_examples/                  # HDRI环境光
    └── video_examples/                 # 测试视频
```

---

## 核心组件

### 1. Tokenizer (VAE)

**作用**: 视频压缩/解压，减少计算量

**架构**: 3D Convolutional VAE

**压缩比**: 8×8×8 (时间×高度×宽度)

**代码位置**: `cosmos_predict1/tokenizer/cosmos_cv_tokenizer.py`

**关键方法**:

```python
class CosmosTokenizer:
    def encode(self, x):
        """
        Input:  [B, 3, 57, 704, 1280]  # RGB视频
        Output: [B, 16, 8, 88, 160]    # Latent

        压缩比:
        - 时间: 57 → 8 (约7倍)
        - 空间: 704×1280 → 88×160 (8倍)
        - 通道: 3 → 16 (扩展)
        """
        h = self.encoder(x)         # 3D Conv encoder
        z = self.quant_conv(h)      # 量化
        return z

    def decode(self, z):
        """
        Input:  [B, 16, 8, 88, 160]    # Latent
        Output: [B, 3, 57, 704, 1280]  # RGB视频
        """
        h = self.post_quant_conv(z)  # 反量化
        x = self.decoder(h)          # 3D Transposed Conv decoder
        return x
```

**为什么需要Tokenizer？**
- 直接在像素空间: [B, 3, 57, 704, 1280] = 122M参数 → 太大！
- 在latent空间: [B, 16, 8, 88, 160] = 0.27M参数 → 节省450倍

### 2. DiT网络 (Diffusion Transformer)

**vs 传统UNet**:

| 架构 | UNet | DiT |
|------|------|-----|
| **设计** | 编码器-解码器 | 纯Transformer |
| **可扩展性** | ~2B参数上限 | 7B+参数 |
| **长程依赖** | 局部感受野 | 全局attention |
| **视频建模** | 困难 | 自然 |

**代码位置**: `cosmos_predict1/diffusion/networks/general_dit_diffusion_renderer.py`

**核心组件**:

1. **Patch Embedding (3D)**:
```python
# 将3D latent切成patches
patches = self.x_embedder(x)
# Input:  [B, 16, 8, 88, 160]
# Output: [B, 28160, 1024]
# num_patches = 8 × 44 × 80 = 28,160
```

2. **RoPE位置编码（3D）**:
```python
# Rotary Position Embedding
# 支持外推：训练57帧 → 推理121帧
class RoPE3D:
    def forward(self, x):
        t_idx, h_idx, w_idx = get_patch_indices(x)
        angles = t_idx*freq_t + h_idx*freq_h + w_idx*freq_w
        return rotate(x, angles)
```

3. **DiT Block with AdaLN**:
```python
class DiTBlock(nn.Module):
    def forward(self, x, t_emb):
        # 时间步调制参数
        shift_attn, scale_attn, gate_attn, \
        shift_mlp, scale_mlp, gate_mlp = self.adaln(t_emb).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = layer_norm(x) * (1 + scale_attn) + shift_attn
        x = x + gate_attn * self.attn(x_norm)

        # MLP with AdaLN
        x_norm = layer_norm(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_norm)

        return x
```

**AdaLN作用**: 不同时间步需要不同处理（高噪声vs低噪声），AdaLN让网络自适应。

### 3. Inverse Renderer（逆向渲染器）

**代码位置**: `cosmos_predict1/diffusion/inference/inference_inverse_renderer.py`

**Context Embedding机制**:

```python
# 模型定义: general_dit_diffusion_renderer.py
class DiffusionRendererGeneralDIT(nn.Module):
    def __init__(self, ...):
        # 初始化context embedding
        self.context_embedding = nn.Embedding(
            num_embeddings=7,       # 7种G-buffer类型
            embedding_dim=1024,     # 嵌入维度
        )

    def forward(self, x, context_index, ...):
        # 根据context_index获取embedding
        context_emb = self.context_embedding(context_index)  # [B, D]

        # 注入到交叉注意力
        crossattn_emb = context_emb.unsqueeze(1).expand(-1, seq_len, -1)
```

**G-buffer索引映射**:

```python
GBUFFER_INDEX_MAPPING = {
    'basecolor': 0,
    'roughness': 1,
    'metallic': 2,
    'normal': 3,
    'depth': 4,
}
```

**为什么同一个模型能生成5种G-buffer？**
- 使用可学习的context embedding
- 不同context_index → 不同embedding → 网络输出不同属性
- 参数共享，避免训练5个独立模型

**核心循环**:

```python
for gbuffer_pass in ['basecolor', 'normal', 'depth', 'roughness', 'metallic']:
    # 设置context
    data_batch["context_index"] = GBUFFER_INDEX_MAPPING[gbuffer_pass]

    # 推理
    output = pipeline.generate_video(
        data_batch=data_batch,
        normalize_normal=(gbuffer_pass == 'normal'),
    )

    # 保存
    save_image_or_video(output, f"{name}.{gbuffer_pass}.jpg")
```

### 4. Forward Renderer（正向渲染器）

**代码位置**: `cosmos_predict1/diffusion/inference/inference_forward_renderer.py`

**环境光表示** (为什么需要3种？):

| 表示 | 物理意义 | 范围 | 作用 |
|------|----------|------|------|
| **env_ldr** | 色调映射的LDR | [0, 1] | 视觉语义 |
| **env_log** | 对数尺度HDR | [-∞, ∞] | 物理准确性 |
| **env_nrm** | 球面法线向量 | [-1, 1]³ | 空间方向 |

**条件注入**:

```python
# 加载HDRI
envlight_dict = process_environment_map(
    hdr_path="asset/examples/hdri_examples/sunny_vondelpark_2k.hdr",
    resolution=(704, 1280),
    num_frames=57,
)

# 3种环境光表示
data_batch['env_ldr'] = envlight_dict['env_ldr'] * 2 - 1  # [-1, 1]
data_batch['env_log'] = envlight_dict['env_log'] * 2 - 1
data_batch['env_nrm'] = envmap_vec([H, W])  # 球面坐标 → 笛卡尔

# 8个条件 = 5个G-buffer + 3个env
# 每个条件16通道 → 总共8×16=128通道
```

**条件拼接**:

```python
def prepare_diffusion_renderer_latent_conditions(self, data_batch):
    latent_conditions = []

    # 遍历所有条件
    for key in ['basecolor', 'normal', 'metallic', 'roughness', 'depth',
                'env_ldr', 'env_log', 'env_nrm']:
        # VAE编码
        condition_latent = self.tokenizer.encode(data_batch[key])
        # [B, C, T, H, W] → [B, 16, T_l, H_l, W_l]

        latent_conditions.append(condition_latent)

    # 通道拼接
    concat_latent = torch.cat(latent_conditions, dim=1)  # [B, 128, T_l, H_l, W_l]

    # 添加mask通道
    mask = torch.ones_like(concat_latent[:, :8, ...])
    concat_latent = torch.cat([concat_latent, mask], dim=1)  # [B, 136, T_l, H_l, W_l]

    return concat_latent
```

---

## 数据流分析

### Inverse Renderer完整数据流

```
[Input] video.mp4
    ↓ Load & Preprocess
[Tensor] [1, 3, 57, 704, 1280] RGB in [-1, 1]
    ↓ Tokenizer.encode()
[Latent] [1, 16, 8, 88, 160]
    ↓ Set context_index (e.g., 0 for basecolor)
[Condition] [1, 16, 8, 88, 160]
    ↓ Diffusion sampling (50 steps)
[Latent] [1, 16, 8, 88, 160] (G-buffer latent)
    ↓ Tokenizer.decode()
[Output] [1, 3, 57, 704, 1280] G-buffer in [-1, 1]
    ↓ Denormalize & Save
[Files] frame_0000.basecolor.jpg, ...
```

### Forward Renderer完整数据流

```
[Input] G-buffers (5个) + HDRI (1个)
    ↓ Load & Preprocess
[Tensors] basecolor, normal, depth, roughness, metallic, env_ldr, env_log, env_nrm
          每个: [1, C, 57, 704, 1280]
    ↓ Tokenizer.encode() for each
[Latents] 8个latents，每个: [1, 16, 8, 88, 160]
    ↓ Concat
[Condition] [1, 128, 8, 88, 160]
    ↓ Add mask
[Condition] [1, 136, 8, 88, 160]
    ↓ Diffusion sampling (50 steps)
[Latent] [1, 16, 8, 88, 160] (Relit RGB latent)
    ↓ Tokenizer.decode()
[Output] [1, 3, 57, 704, 1280] Relit RGB
    ↓ Save as video
[File] output_relit.mp4
```

---

## 模型加载机制

### 模型文件结构

```
checkpoints/
├── Diffusion_Renderer_Inverse_Cosmos_7B/
│   ├── model.pt                    # DiT权重 (28GB)
│   └── config.json                 # 模型配置
│
├── Diffusion_Renderer_Forward_Cosmos_7B/
│   ├── model.pt                    # DiT权重 (28GB)
│   └── config.json
│
└── Cosmos-Tokenize1-CV8x8x8-720p/
    ├── autoencoder.jit             # 完整VAE (203MB)
    ├── encoder.jit                 # 仅Encoder (83MB)
    ├── decoder.jit                 # 仅Decoder (121MB)
    ├── model.pt                    # PyTorch权重 (1.4GB)
    ├── mean_std.pt                 # 归一化参数
    └── config.json
```

### 加载流程

**代码位置**: `cosmos_predict1/diffusion/model/model_diffusion_renderer.py`

```python
class DiffusionRendererModel:
    def load_model(self, checkpoint_dir, diffusion_transformer_dir):
        # 1. 加载Tokenizer
        self.tokenizer = load_tokenizer(
            checkpoint_dir=checkpoint_dir / "Cosmos-Tokenize1-CV8x8x8-720p",
        )

        # 2. 加载DiT配置
        config_path = checkpoint_dir / diffusion_transformer_dir / "config.json"
        config = json.load(open(config_path))

        # 3. 创建DiT网络
        self.dit_net = DiffusionRendererGeneralDIT(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_attention_heads=config['num_attention_heads'],
            ...
        )

        # 4. 加载DiT权重
        model_path = checkpoint_dir / diffusion_transformer_dir / "model.pt"
        state_dict = torch.load(model_path, map_location='cpu')
        self.dit_net.load_state_dict(state_dict)

        # 5. 移到GPU
        self.dit_net = self.dit_net.cuda()
```

### Offload机制

**作用**: 降低显存占用（牺牲速度）

```python
if args.offload_diffusion_transformer:
    # DiT移到CPU，推理时再移到GPU
    self.dit_net = self.dit_net.cpu()

if args.offload_tokenizer:
    # Tokenizer移到CPU
    self.tokenizer = self.tokenizer.cpu()

# 推理时临时移到GPU
def forward(self, x):
    if offload:
        self.dit_net = self.dit_net.cuda()
    output = self.dit_net(x)
    if offload:
        self.dit_net = self.dit_net.cpu()
    return output
```

**Tradeoff**:
- 显存节省
- 速度损失

---

## 扩散采样流程

### EDM Scheduler

**vs DDPM/DDIM**:

| Scheduler | 时间步表示 | 调度方式 | 采样步数 | 质量 |
|-----------|-----------|---------|---------|------|
| DDPM | t ∈ [0, 1000] | 线性 | 1000 | 中等 |
| DDIM | t ∈ [0, 1000] | 余弦 | 50-100 | 好 |
| **EDM** | σ ∈ [0.002, 80] | **对数** | **50** | **最好** |

### 采样循环

**代码位置**: `cosmos_predict1/diffusion/model/model_diffusion_renderer.py`

```python
def generate_samples_from_batch(self, data_batch, num_steps=50):
    # 1. 准备条件
    conditions = self._get_conditions(data_batch)

    # 2. 初始化噪声
    xt = torch.randn(B, 16, T_l, H_l, W_l) * sigma_max

    # 3. 设置时间步（EDM调度）
    self.scheduler.set_timesteps(num_steps)
    # timesteps = [80, 78.5, ..., 0.5, 0.002]

    # 4. 去噪循环
    for t in timesteps:
        # 4.1 Concat条件
        model_input = torch.cat([xt, conditions], dim=1)
        model_input = self.concat_head(model_input)

        # 4.2 DiT预测噪声
        noise_pred = self.dit_net(
            model_input,
            t,
            rope_emb,       # 位置编码
            crossattn_emb   # 交叉注意力（inverse的context）
        )

        # 4.3 EDM更新
        xt = self.scheduler.step(noise_pred, t, xt).prev_sample

    # 5. 返回去噪后的latent
    return xt
```

---

## 关键技术点

### 1. 时序一致性

**如何保证视频连贯？**
- 3D卷积和3D attention处理时间维度
- RoPE 3D位置编码保持时空关系
- VAE在时间维度上的平滑

### 2. 物理准确性

**如何保证物理正确？**
- G-buffer基于PBR模型
- HDRI提供真实光照
- 扩散模型学习渲染方程

### 3. 可扩展性

**如何扩展到更长视频？**
- RoPE支持外推：训练57帧 → 推理121帧
- 滑动窗口处理超长视频
- 时间分块 + 重叠融合

---

## 性能优化

### 显存占用

| 配置 | 分辨率 | 帧数 | 显存峰值 |
|------|--------|------|---------|
| 最小 | 640×360 | 33 | ~16GB |
| 推荐 | 1280×704 | 57 | ~27GB |
| 高质量 | 1920×1080 | 121 | ~48GB |

### 优化策略

1. **降低分辨率**: ~60%显存节省，~4x加速
2. **减少帧数**: ~40%显存节省
3. **启用Offload**: ~35%显存节省，~45%变慢
4. **减少采样步数**: 从50→30步，~40%加速，轻微质量损失

---

## 总结

### 核心设计

1. **Latent Diffusion**: 在压缩空间操作，提升效率
2. **DiT架构**: 替代UNet，提升可扩展性
3. **两阶段渲染**: 分离材质和光照，灵活控制
4. **Context Embedding**: 一个模型生成多种输出

### 优势

- ✅ 无需3D模型
- ✅ 物理准确
- ✅ 时序一致
- ✅ 高效推理

### 局限

- 逐帧处理仍较慢（~180秒/视频）
- 需要大显存（≥16GB）
- 依赖高质量输入
