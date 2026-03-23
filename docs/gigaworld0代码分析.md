# GigaWorld-0 vs Cosmos-Transfer1 对比分析

---

## 基本信息

### GigaWorld-0

- 论文: arXiv 2511.19861 (2025-11)
- 定位: VLA数据引擎
- 模型: 2B参数 IT2V (Image-Text-to-Video)
- 许可: Apache 2.0
- 开源状态: 部分 (仅IT2V)

### Cosmos-Transfer1

- 论文: CVPR'25 Oral
- 定位: 视频重光照
- 模型: 7B Inverse + 7B Forward
- 许可: Apache 2.0 + NVIDIA Open Model
- 开源状态: 完全

---

## 开源范围

### GigaWorld-0 已开源内容

```
giga-world-0/
├── giga_world_0/
│   ├── giga_world_0_trainer.py      # 扩散模型训练
│   └── giga_world_0_transforms.py   # 数据预处理
├── scripts/
│   ├── inference.py                 # IT2V推理
│   ├── train.py                     # 训练
│   └── pack_data.py                 # 数据打包
└── configs/
    └── giga_world_0_video.py        # 训练配置
```

功能: IT2V生成 (从图像+文本生成视频)

### GigaWorld-0 未开源内容

论文提及但代码不存在:

| 功能 | 论文描述 | 代码验证 |
|------|----------|----------|
| AppearanceTransfer | 光照/材质编辑 | `grep -r "relighting\|appearance"` 无匹配 |
| 3D模块 | Gaussian Splatting | `grep -r "gaussian\|3d"` 无匹配 |
| 物理仿真 | 可微物理系统 | `grep -r "physics"` 无匹配 |

---

## 技术架构

### GigaWorld-0 管线

```
图像 + 文本
  ↓ T5编码 + VAE编码
条件latent
  ↓ 3D DiT (2B参数, 30步)
生成latent
  ↓ VAE解码
全新视频
```

### Cosmos-Transfer1 管线

```
RGB视频
  ↓ Inverse渲染 (7B DiT, 50步)
G-buffers (5种)
  ↓ + HDRI (3种表示)
条件latent (8×16=128通道)
  ↓ Forward渲染 (7B DiT, 50步)
重光照视频
```

### 条件注入对比

| 维度 | GigaWorld-0 | Cosmos-Transfer1 |
|------|-------------|------------------|
| 条件类型 | Reference frames + Text | G-buffers + HDRI |
| 条件通道 | 16 (latent) + text | 136 (8×16+8) |
| 注入方式 | Concat + Cross-attn | Concat |
| 控制精度 | 文本描述 | 物理参数 |

### 模型规格

| 特性 | GigaWorld-0 | Cosmos-Transfer1 |
|------|-------------|------------------|
| 参数量 | 2B | 14B (7B×2) |
| 显存 | 16GB | 27GB |
| 采样步数 | 30 | 50 |
| Scheduler | EDM | EDM |
| 位置编码 | 未知 | RoPE 3D |
| VAE压缩比 | 未知 | 8×8×8 |

---

## 代码实现

### GigaWorld-0 训练器

`giga_world_0/giga_world_0_trainer.py`:

```python
class GigaWorld0Trainer(Trainer):
    def forward_step(self, batch_dict):
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']

        # VAE编码
        latents = self.forward_vae(images)

        # 添加噪声
        input_latents, timesteps = self.edm_loss.add_noise(latents)

        # Reference frame masking
        ref_latents = self.forward_vae(batch_dict['ref_images'])
        ref_masks = batch_dict['ref_masks']
        input_latents = ref_masks * ref_latents + (1 - ref_masks) * input_latents
        input_latents = torch.cat([input_latents, ref_masks], dim=1)

        # Transformer预测
        model_pred = transformer(
            x=input_latents,
            timesteps=timesteps,
            crossattn_emb=prompt_embeds,
        )

        # 损失计算
        loss = self.edm_loss.compute_loss(model_pred)
        return loss
```

特点: 标准扩散训练, Reference frame作为条件, 文本通过cross-attention注入, 无光照控制。

### GigaWorld-0 推理

`scripts/inference.py`:

```python
def _inference(device, data_path, ...):
    pipe = GigaWorld0Pipeline.from_pretrained(...)

    for data_dict in data_list:
        prompt = data_dict['prompt']
        image = Image.open(data_dict['image'])

        output_images = pipe(
            prompt=[prompt],
            image=image,
            num_frames=61,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        save_video(output_images, save_path)
```

功能: IT2V生成, 不支持后处理。

---

## 功能对比

| 维度 | GigaWorld-0 | Cosmos-Transfer1 |
|------|-------------|------------------|
| 任务 | 生成新视频 | 后处理已有视频 |
| 输入 | 图像+文本 | RGB视频 |
| 输出 | 全新视频 | 重光照视频 |
| 几何保持 | 否 | 是 |
| 光照控制 | 文本描述 | HDRI |
| 物理准确性 | 学习驱动 | PBR渲染 |
| 数据增强 | 未开源 | 已开源 |

---

## 应用场景

### GigaWorld-0

- 生成合成场景
- VLA预训练数据生成
- 快速原型验证

### Cosmos-Transfer1

- 真实数据光照增强
- VLA fine-tuning
- 电影后期重光照

---

## 任务匹配

| 任务 | 方案 | 理由 |
|------|------|------|
| 已有视频重光照 | Cosmos | 保持几何, 精确控制 |
| 机器人数据增强 | Cosmos | 物理准确 |
| 生成全新场景 | GigaWorld-0 | 无需真实数据 |
| 电影后期 | Cosmos | 物理准确 |
| VLA预训练 | GigaWorld-0 | 大规模生成 |
| VLA fine-tuning | Cosmos | 保真度高 |

---

## GigaWorld-0 使用

### 环境配置

```bash
conda create -n giga_world_0 python=3.11.10 -y
conda activate giga_world_0

pip install giga-train giga-datasets natten

git clone https://github.com/open-gigaai/giga-models.git
cd giga-models && pip install -e .

git clone https://github.com/open-gigaai/giga-world-0.git
cd giga-world-0
```

### 下载模型

```bash
python scripts/download.py \
    --model-name video_pretrain \
    --save-dir /path/to/model/
```

模型大小: 2GB

### 推理

```bash
python scripts/inference.py \
    --data-path input.json \
    --save-dir ./output \
    --transformer-model-path /path/to/model \
    --text-encoder-model-path /path/to/text_encoder \
    --vae-model-path /path/to/vae \
    --gpu_ids 0
```

输入格式 (input.json):

```json
[
  {
    "prompt": "Use the right hand to pick up pink peach",
    "image": "init_frame.png"
  }
]
```

### 训练

```bash
# 打包数据
python scripts/pack_data.py \
    --video-dir /path/to/videos \
    --save-dir /path/to/packed

# 训练
python scripts/train.py --config configs.giga_world_0_video.config
```

LoRA fine-tuning:

```python
# configs/giga_world_0_video.py
config.train_mode = 'lora'
config.lora_rank = 64
```

LoRA训练: 0.1%参数, 显存降低50%, 速度提升2倍。

---

## 代码验证

### 搜索relighting功能

```bash
cd giga-world-0
grep -r "relighting\|AppearanceTransfer\|lighting" . --include="*.py" -i
```

结果: 仅找到 `augment_sigma` (噪声增强), 与光照无关。

### 搜索3D功能

```bash
grep -r "gaussian\|3d\|physics\|splatting" . --include="*.py" -i
```

结果: 无匹配。

论文声称的AppearanceTransfer和3D模块在代码中不存在。

---

## 组合使用

### VLA训练管线

```
Stage 1: 预训练
  数据: GigaWorld-0生成大规模合成数据
  要求: 量大, 质量中等

Stage 2: Fine-tuning
  数据: Cosmos增强真实数据
  要求: 质量高, 保真度高
```

### 机器人数据集

```
Step 1: 采集真实轨迹 (少量)
Step 2: Cosmos重光照 (3-4倍扩充)
Step 3: GigaWorld-0生成新场景 (可选)
```

---

## 总结

### 核心发现

1. GigaWorld-0仅开源IT2V基础功能
2. AppearanceTransfer (重光照/数据增强) 未开源
3. 3D模块 (Gaussian Splatting, 物理仿真) 未开源
4. 两者适用场景不同: GigaWorld-0生成新数据, Cosmos后处理增强

### 数据增强任务建议

使用Cosmos-Transfer1:
- 核心功能完全开源
- 物理准确光照控制
- 保持几何一致性
- 适合VLA训练

GigaWorld-0不适用于数据增强:
- 核心功能未开源
- 无精确光照控制
- 可能改变几何

GigaWorld-0参考价值:
- IT2V生成框架
- 3D DiT架构
- VLA训练pipeline设计

---

## 资源

GigaWorld-0:
- 论文: https://arxiv.org/abs/2511.19861
- GitHub: https://github.com/open-gigaai/giga-world-0
- HuggingFace: https://huggingface.co/open-gigaai

Cosmos-Transfer1:
- 论文: https://arxiv.org/abs/2501.18590
- GitHub: https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer
- 项目主页: https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/
