# gen_trajs.py 使用手册

视频重光照数据增强脚本，基于DiffusionRenderer实现批量处理。

---

## 工作原理

```
输入视频 → 提取帧(ffmpeg) → Inverse渲染(估计G-buffer) → Forward渲染(注入HDRI) → 输出视频
```

G-buffer包含:
- basecolor: 漫反射颜色(去除光照)
- normal: 表面法线
- depth: 深度图
- roughness: 粗糙度
- metallic: 金属度

---

## 基础用法

```bash
python gen_trajs.py \
    --input_video_dir ./videos \
    --output_dir ./output
```

默认配置:
- 帧数: 57 (2.4秒 @24fps)
- 分辨率: 1280x704
- HDRI: sunny_vondelpark, pink_sunrise, street_lamp (3个)
- Offload: 启用

输出:
- 每个输入视频 → 3个重光照版本
- 数据扩充3倍

---

## 关键限制

### 1. 帧数只能是1或57

```bash
--num_frames 1     # 图像模式
--num_frames 57    # 视频模式(默认)
```

其他值会报错，这是模型训练设置。

### 2. example数据

**视频** (`asset/examples/video_examples/`):
- video1.mp4 ~ video4.mp4
- 1280x704, ~2.4秒

**图像** (`asset/examples/image_examples/`):
- 16个.jpg
- 1280x704

---

## 参数说明

### HDRI控制

**预定义HDRI** (默认):

```bash
--hdri_indices 0 1 2        # 3个HDRI(默认)
--hdri_indices 0            # 1个
--hdri_indices 0 1 2 3      # 全部4个
```

索引对应 (`asset/examples/hdri_examples/`):
- 0: sunny_vondelpark_2k.hdr
- 1: pink_sunrise_2k.hdr
- 2: street_lamp_2k.hdr
- 3: rosendal_plains_1_2k.hdr

**随机HDRI**:

```bash
--use_random_hdri --hdri_indices 0 1 2
```

生成3个随机光照版本，不可复现。

输出命名:
- 预定义: `video1_sunny_vondelpark/`
- 随机: `video1_random_0/`

性能: 随机HDRI无额外开销(差异<1%)。

### 分辨率

```bash
--resolution 640x360      # 降低分辨率
--resolution 1280x704     # 默认
--resolution 3840x2160    # 提高分辨率
```

实验结果: 分辨率对性能影响<1%，瓶颈在DiT推理而非分辨率。
建议: 保持1280x704或更高。

### Offload模式

```bash
# 默认: 启用offload
python gen_trajs.py ...

# 禁用offload
python gen_trajs.py ... --disable_offload
```

对比 (A100 80GB):

| 配置 | 速度 | 显存 |
|------|------|------|
| Offload启用 | 26.9min/视频 | 24.9GB |
| Offload禁用 | 24.2min/视频(快10%) | 27.7GB |

选择:
- 显存≤24GB: 必须offload
- 显存≥28GB: 禁用offload更快

原理: Offload将DiT模型在CPU/GPU间移动，节省显存但增加PCIe传输开销。

---

## 典型用例

### 快速测试

```bash
python gen_trajs.py \
    --input_video_dir asset/examples/video_examples \
    --output_dir ./test \
    --num_frames 1 \
    --hdri_indices 0
```

预计耗时: ~3分钟(4个视频, 1帧, 1HDRI)

### 标准数据增强

```bash
python gen_trajs.py \
    --input_video_dir ./raw_videos \
    --output_dir ./augmented \
    --num_frames 57 \
    --hdri_indices 0 1 2
```

预计耗时: ~27分钟/视频 (3个HDRI)

### 高性能配置

```bash
python gen_trajs.py \
    --input_video_dir ./videos \
    --output_dir ./output \
    --disable_offload \
    --hdri_indices 0 1 2 3
```

要求: GPU显存≥28GB
预计耗时: ~24分钟/视频 (4个HDRI)

### 最大多样性

```bash
python gen_trajs.py \
    --input_video_dir ./videos \
    --output_dir ./output \
    --use_random_hdri \
    --hdri_indices 0 1 2 3 4 5 6 7
```

生成8个随机光照版本，适合VLA训练。

---

## 输出结构

```
output_dir/
├── extracted_frames/              # 阶段1: PNG序列
│   └── video1/
│       ├── 00000.png
│       └── ...
│
├── gbuffer_frames/                # 阶段2: G-buffer估计
│   └── video1/
│       └── gbuffer_frames/
│           ├── basecolor/         # 5种材质×57帧
│           ├── normal/
│           ├── depth/
│           ├── roughness/
│           └── metallic/
│
├── relit_videos/                  # 阶段3: 重光照视频
│   ├── video1_sunny_vondelpark/
│   │   └── video.mp4
│   ├── video1_pink_sunrise/
│   │   └── video.mp4
│   └── ...
│
├── logs/
│   └── pipeline_*.log
│
└── augmentation_report.json       # 性能统计
```

---

## 性能报告

`augmentation_report.json` 包含:

```json
{
  "summary": {
    "total_videos": 4,
    "processed_videos": 4,
    "total_augmented_videos": 12,
    "expansion_ratio": "3.0x",
    "total_time_formatted": "107.8 minutes"
  },
  "timing": {
    "video1_extract": 2.3,
    "video1_inverse": 703.7,
    "video1_sunny_vondelpark_forward": 227.0,
    ...
  },
  "vram_usage": {
    "inverse_rendering": {
      "max_reserved_gb": 20.59
    },
    "forward_rendering": {
      "max_reserved_gb": 24.94
    }
  },
  "failed_videos": []
}
```

关键指标:
- `expansion_ratio`: 数据扩充倍数
- `timing.*_inverse`: Inverse渲染耗时(秒)
- `timing.*_forward`: Forward渲染耗时(秒)
- `vram_usage.*.max_reserved_gb`: 峰值显存(GB)

---

## 完整参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_video_dir` | str | 必需 | 输入目录(.mp4或.jpg) |
| `--output_dir` | str | 必需 | 输出目录 |
| `--checkpoint_dir` | str | `checkpoints` | 模型路径 |
| `--num_frames` | int | `57` | 帧数(1或57) |
| `--resolution` | str | `1280x704` | 分辨率(WxH) |
| `--frame_rate` | int | `24` | 帧率(fps) |
| `--hdri_indices` | int[] | `[0,1,2]` | HDRI索引 |
| `--use_random_hdri` | flag | False | 随机光照 |
| `--disable_offload` | flag | False | 禁用offload |
| `--no_vram_monitor` | flag | False | 禁用显存监控 |
| `--quiet` | flag | False | 静默模式 |

---

## 常见问题

**Q: 为什么只支持1或57帧?**

A: 模型训练时固定这两个设置，其他值未训练。

**Q: 降低分辨率能加速吗?**

A: 不能。实验显示分辨率影响<1%，瓶颈在DiT推理。

**Q: 应该用多少个HDRI?**

A:
- 测试: 1个
- 生产: 3-4个(边际成本低)
- 大规模: 先Inverse缓存, 按需Forward

**Q: 随机HDRI vs 预定义HDRI?**

A:
- 预定义: 可控, 可复现
- 随机: 多样性高, 不可复现, 无性能损失

**Q: 如何继续中断的任务?**

A: 直接重新运行相同命令，脚本会跳过已完成部分。

**Q: 处理时间多久?**

A: 基于A100 80GB:
- 1帧: 3.6min/视频
- 57帧(offload): 26.9min/视频
- 57帧(无offload): 24.2min/视频

时间 = Inverse(固定) + Forward×HDRI数

**Q: 显存不够?**

A: 按优先级:
1. 确保offload启用(默认)
2. 改用1帧模式: `--num_frames 1`
3. 减少HDRI: `--hdri_indices 0`

---

## 性能数据 (A100 80GB)

### 显存峰值

| 配置 | Inverse | Forward | 总峰值 |
|------|---------|---------|--------|
| 1帧 | 14.7GB | 14.8GB | 14.8GB |
| 57帧+offload | 20.6GB | 24.9GB | 24.9GB |
| 57帧+无offload | 23.4GB | 27.7GB | 27.7GB |

### 处理时间 (4视频, 4HDRI)

| 阶段 | Offload | 无Offload |
|------|---------|-----------|
| 帧提取 | 19s | 20s |
| Inverse | 2815s | 2235s |
| Forward | 3632s | 3549s |
| 总计 | 107.8min | 96.8min |

### 成本分析

单视频, 57帧:
- Inverse(固定): 703s (16.3min)
- Forward/HDRI: 227s (3.8min)

HDRI扩充成本:
- 1→2个HDRI: +3.8min
- 1→4个HDRI: +11.4min
- 边际成本: 3.8min/HDRI

---

## 环境要求

- Python 3.10
- GPU ≥16GB (推荐≥24GB)
- 模型已下载 (~58GB)
- Conda环境 `cosmos-predict1`

详见: [cosmos-predict环境配置.md](cosmos-predict环境配置.md)

---

## 技术细节

### 显存监控实现

子进程内部监控:

```python
# inference_inverse_renderer.py
torch.cuda.reset_peak_memory_stats()
# ... 推理 ...
peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"[VRAM_STATS] Peak GPU Memory: {peak_gb:.2f}GB")
```

主进程解析输出:

```python
# gen_trajs.py
def _parse_vram_stats(output):
    match = re.search(r'\[VRAM_STATS\].*?(\d+\.\d+)GB', output)
    return float(match.group(1)) if match else None
```

优势: 100%准确，零性能开销。

### 断点续传逻辑

检查已完成的阶段:

```python
def _check_stage_completed(stage, video_name, hdri_idx=None):
    if stage == "extract":
        return os.path.exists(f"{output_dir}/extracted_frames/{video_name}/00000.png")
    elif stage == "inverse":
        return os.path.exists(f"{output_dir}/gbuffer_frames/{video_name}/gbuffer_frames/")
    elif stage == "forward":
        return os.path.exists(f"{output_dir}/relit_videos/{video_name}_{hdri_name}/video.mp4")
```

跳过已完成阶段，从中断处继续。
