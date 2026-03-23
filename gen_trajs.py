#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_trajs.py - Data Augmentation Pipeline Entry Script

功能：遍历输入视频，通过DiffusionRenderer生成3种不同光照条件的重光照视频，
     实现3倍数据扩充用于VLA模型训练。

工作流程：
    1. 提取视频帧（如果输入是视频文件）
    2. 逆向渲染：估计G-buffers (albedo, normal, depth, roughness, metallic)
    3. 正向渲染：使用3种HDRI环境光重光照
    4. 保存增强后的视频

Updated: 2026-03-21
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil
from datetime import datetime
import logging

# GPU 显存监控
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, GPU memory monitoring disabled")

# 添加项目路径
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

# 默认HDRI环境光索引（对应3种不同光照条件）
DEFAULT_HDRI_INDICES = [0, 1, 2]  # sunny, pink_sunrise, street_lamp

# HDRI环境光名称映射
HDRI_NAMES = {
    0: "sunny_vondelpark",
    1: "pink_sunrise",
    2: "street_lamp",
    3: "rosendal_plains",
}


class GPUMonitor:
    """GPU显存监控类（使用nvidia-smi，可跨进程监控）"""

    def __init__(self, gpu_id: int = 0):
        """初始化GPU监控器"""
        self.gpu_id = gpu_id
        self.available = self._check_nvidia_smi()

        if not self.available:
            print(f"GPU monitoring disabled (GPU {gpu_id}) - nvidia-smi not available")

    def _check_nvidia_smi(self) -> bool:
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(
                "nvidia-smi",
                shell=True,
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_memory_info(self) -> Optional[Dict[str, float]]:
        """
        获取当前显存使用情况（使用nvidia-smi，可监控子进程）

        Returns:
            包含显存信息的字典:
            {
                'used_gb': float,        # 已使用显存(GB)
                'total_gb': float,       # 总显存(GB)
                'free_gb': float,        # 空闲显存(GB)
                'utilization': float     # 使用率(%)
            }
        """
        if not self.available:
            return None

        try:
            # 使用nvidia-smi查询显存
            cmd = f"nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i {self.gpu_id}"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                used_mb, total_mb = map(float, result.stdout.strip().split(','))
                used_gb = used_mb / 1024
                total_gb = total_mb / 1024
                free_gb = total_gb - used_gb
                utilization = (used_gb / total_gb) * 100

                return {
                    'used_gb': used_gb,
                    'total_gb': total_gb,
                    'free_gb': free_gb,
                    'utilization': utilization
                }
        except Exception as e:
            print(f"Failed to get GPU memory info: {e}")
            return None

    def format_memory_info(self, mem_info: Optional[Dict[str, float]]) -> str:
        """格式化显存信息为可读字符串"""
        if mem_info is None:
            return "GPU memory info unavailable"

        return (f"GPU Memory: {mem_info['used_gb']:.2f}GB used / {mem_info['total_gb']:.2f}GB total "
                f"({mem_info['utilization']:.1f}%), {mem_info['free_gb']:.2f}GB free")

    def clear_cache(self):
        """清空GPU缓存（仅当torch可用时）"""
        if TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


class DataAugmentationPipeline:
    """数据增强管线类"""

    def __init__(
        self,
        input_video_dir: str,
        output_dir: str,
        checkpoint_dir: str = "checkpoints",
        hdri_indices: List[int] = None,
        num_video_frames: int = 57,
        frame_rate: int = 24,
        resolution: str = "1280x704",
        conda_env: str = "cosmos-predict1",
        verbose: bool = True,
        monitor_vram: bool = True,
        log_to_file: bool = True,
        enable_offload: bool = True,
        use_random_hdri: bool = False,
        max_videos: Optional[int] = None,
    ):
        """
        初始化数据增强管线

        Args:
            input_video_dir: 输入视频目录
            output_dir: 输出目录
            checkpoint_dir: 模型权重目录（相对于repo根目录）
            hdri_indices: HDRI环境光索引列表
            num_video_frames: 每个视频处理的帧数
            frame_rate: 视频帧率
            resolution: 视频分辨率 (宽x高)
            conda_env: conda环境名称
            verbose: 是否打印详细信息
            monitor_vram: 是否监控显存使用
            log_to_file: 是否输出日志到文件
            enable_offload: 是否启用CPU offload（降低显存占用，默认True）
            use_random_hdri: 是否使用随机HDRI（默认False，使用预定义HDRI）
            max_videos: 限制处理的视频数量（默认None，处理所有视频）
        """
        self.input_video_dir = Path(input_video_dir).absolute()
        self.output_dir = Path(output_dir).absolute()

        # 处理checkpoint_dir路径 - 如果是相对路径，则相对于repo根目录
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.is_absolute():
            self.checkpoint_dir = (REPO_ROOT / checkpoint_path).absolute()
        else:
            self.checkpoint_dir = checkpoint_path

        self.hdri_indices = hdri_indices or DEFAULT_HDRI_INDICES
        self.num_video_frames = num_video_frames
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.conda_env = conda_env
        self.verbose = verbose
        self.monitor_vram = monitor_vram
        self.log_to_file = log_to_file
        self.enable_offload = enable_offload
        self.use_random_hdri = use_random_hdri
        self.max_videos = max_videos

        # 创建输出目录结构
        self.frames_dir = self.output_dir / "extracted_frames"
        self.gbuffer_dir = self.output_dir / "gbuffer_frames"
        self.relit_dir = self.output_dir / "relit_videos"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.frames_dir, self.gbuffer_dir, self.relit_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 设置日志系统
        self.logger = None
        self.log_file = None
        if self.log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.logs_dir / f"pipeline_{timestamp}.log"

            # 配置 logger
            self.logger = logging.getLogger(f"pipeline_{timestamp}")
            self.logger.setLevel(logging.INFO)

            # 文件处理器
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)

            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            # 格式化
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s',
                                         datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            ch.setFormatter(logging.Formatter('[Pipeline] %(message)s'))

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # 初始化GPU监控器
        self.gpu_monitor = GPUMonitor() if monitor_vram else None

        # 启动时间
        self.start_time = time.time()

        # 性能统计
        self.stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "failed_videos": [],
            "timing": {},
            "stage_timing": {},  # 各阶段总耗时
            "vram_usage": {},
        }

        # 阶段计时器
        self.stage_timers = {}

        self._log(f"Initialized data augmentation pipeline")
        self._log(f"Input directory: {self.input_video_dir}")
        self._log(f"Output directory: {self.output_dir}")
        self._log(f"Checkpoint directory: {self.checkpoint_dir}")
        self._log(f"HDRI indices: {self.hdri_indices}")
        if self.log_to_file:
            self._log(f"Log file: {self.log_file}")

    def _log(self, message: str):
        """打印日志"""
        if self.logger:
            self.logger.info(message)
        elif self.verbose:
            print(f"[Pipeline] {message}")

    def _log_vram(self, prefix: str = "") -> Optional[Dict[str, float]]:
        """记录当前显存使用情况"""
        if not self.gpu_monitor:
            return None

        mem_info = self.gpu_monitor.get_memory_info()
        if mem_info:
            msg = self.gpu_monitor.format_memory_info(mem_info)
            self._log(f"{prefix}{msg}")
            return mem_info
        return None

    def _start_stage_timer(self, stage_name: str):
        """开始阶段计时"""
        self.stage_timers[stage_name] = time.time()
        self._log(f"\n⏱️  Stage '{stage_name}' started")

    def _end_stage_timer(self, stage_name: str):
        """结束阶段计时"""
        if stage_name in self.stage_timers:
            elapsed = time.time() - self.stage_timers[stage_name]
            self.stats["stage_timing"][stage_name] = elapsed
            self._log(f"⏱️  Stage '{stage_name}' completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
            return elapsed
        return 0

    def _get_python_cmd(self) -> str:
        """
        获取 Python 命令前缀

        检查当前是否在目标 conda 环境中：
        - 如果是，直接使用 python（子进程继承环境）
        - 如果不是，返回带环境激活的命令
        """
        import os
        current_env = os.environ.get('CONDA_DEFAULT_ENV', '')

        if current_env == self.conda_env:
            # 已经在正确的环境中，直接使用 python
            return "python"
        else:
            # 需要激活环境 - 尝试检测 conda 路径
            conda_exe = os.environ.get('CONDA_EXE', '')
            if conda_exe:
                # 使用 conda run 命令（更可靠）
                return f"conda run -n {self.conda_env} python"
            else:
                # 回退到传统方式（可能失败）
                self._log(f"Warning: Not in target conda env '{self.conda_env}'. Current: '{current_env}'")
                self._log(f"Please run this script from within the '{self.conda_env}' environment")
                return "python"  # 尝试使用当前环境的 python

    def _run_command(
        self,
        cmd: str,
        description: str,
        capture_output: bool = False,
        track_stage: str = None
    ) -> Tuple[bool, str, float]:
        """
        执行shell命令

        Args:
            cmd: 命令字符串
            description: 命令描述
            capture_output: 是否捕获输出
            track_stage: 跟踪阶段名称（用于显存统计）

        Returns:
            (success, output, elapsed_time)
        """
        self._log(f"Executing: {description}")
        if self.verbose:
            self._log(f"Command: {cmd.strip()}")

        # 记录开始前的显存
        mem_before = self._log_vram("  Before execution - ")

        start_time = time.time()

        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    executable="/bin/bash"
                )
                elapsed = time.time() - start_time

                # 记录结束后的显存
                mem_after = self._log_vram("  After execution - ")

                # 解析子进程输出的显存峰值（来自推理脚本内部）
                peak_vram_gb = self._parse_vram_stats(result.stdout)
                if peak_vram_gb and track_stage:
                    if track_stage not in self.stats["vram_usage"]:
                        self.stats["vram_usage"][track_stage] = []
                    self.stats["vram_usage"][track_stage].append({
                        'peak_used_gb': peak_vram_gb,
                        'total_gb': mem_after['total_gb'] if mem_after else 0
                    })
                    self._log(f"  ✅ Captured peak VRAM from subprocess: {peak_vram_gb:.2f}GB")

                if result.returncode == 0:
                    self._log(f"✓ Completed ({elapsed:.2f}s)")
                    return True, result.stdout, elapsed
                else:
                    self._log(f"✗ Failed: {result.stderr}")
                    return False, result.stderr, elapsed
            else:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    executable="/bin/bash"
                )
                elapsed = time.time() - start_time

                # 记录结束后的显存
                mem_after = self._log_vram("  After execution - ")

                if result.returncode == 0:
                    self._log(f"✓ Completed ({elapsed:.2f}s)")
                    return True, "", elapsed
                else:
                    self._log(f"✗ Failed (return code: {result.returncode})")
                    return False, "", elapsed

        except Exception as e:
            elapsed = time.time() - start_time
            self._log(f"✗ Exception: {str(e)}")
            return False, str(e), elapsed

    def _parse_vram_stats(self, output: str) -> Optional[float]:
        """
        解析子进程输出中的显存统计信息

        查找形如 "[VRAM_STATS] Peak GPU Memory: 12.34GB / 80.00GB" 的行

        Returns:
            峰值显存使用量（GB），如果未找到则返回None
        """
        import re
        if not output:
            return None

        # 匹配 [VRAM_STATS] Peak GPU Memory: 12.34GB / 80.00GB
        pattern = r'\[VRAM_STATS\]\s+Peak GPU Memory:\s+([\d.]+)GB'
        match = re.search(pattern, output)

        if match:
            return float(match.group(1))
        return None

    def extract_frames_from_videos(self) -> List[Path]:
        """
        从输入视频中提取帧

        Returns:
            提取的视频帧目录列表
        """
        self._log("\n=== Stage 1: Extract Video Frames ===")
        self._start_stage_timer("Stage 1: Frame Extraction")

        # 查找所有视频文件
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(self.input_video_dir.glob(f"*{ext}")))
            video_files.extend(list(self.input_video_dir.glob(f"*{ext.upper()}")))

        # 限制处理的视频数量（如果指定）
        if self.max_videos is not None and len(video_files) > self.max_videos:
            self._log(f"Limiting to first {self.max_videos} video(s) (found {len(video_files)} total)")
            video_files = video_files[:self.max_videos]

        self.stats["total_videos"] = len(video_files)
        self._log(f"Processing {len(video_files)} video file(s)")

        if len(video_files) == 0:
            self._log("Warning: No video files found, trying to use frame directories")
            # 假设输入目录已经包含帧
            return [self.input_video_dir]

        # 提取每个视频的帧
        extracted_dirs = []
        for video_path in video_files:
            video_name = video_path.stem
            output_frame_dir = self.frames_dir / video_name

            # 如果已经提取过，跳过
            if output_frame_dir.exists() and len(list(output_frame_dir.glob("*.png"))) > 0:
                self._log(f"Skipping already extracted video: {video_name}")
                extracted_dirs.append(output_frame_dir)
                continue

            python_cmd = self._get_python_cmd()
            cmd = f"""cd {REPO_ROOT} && \\
            {python_cmd} scripts/dataproc_extract_frames_from_video.py \\
                --input_folder {video_path.parent} \\
                --output_folder {self.frames_dir} \\
                --frame_rate {self.frame_rate} \\
                --resize {self.resolution} \\
                --max_frames {self.num_video_frames}
            """

            success, output, elapsed = self._run_command(
                cmd,
                f"Extracting frames from: {video_name}",
                track_stage="frame_extraction"
            )

            if success and output_frame_dir.exists():
                extracted_dirs.append(output_frame_dir)
                self.stats["timing"][f"{video_name}_extract"] = elapsed
            else:
                self.stats["failed_videos"].append((video_name, "frame_extraction"))

        self._log(f"Successfully extracted frames from {len(extracted_dirs)} video(s)")
        self._end_stage_timer("Stage 1: Frame Extraction")
        return extracted_dirs

    def run_inverse_rendering(self, frame_dirs: List[Path]) -> List[Path]:
        """
        运行逆向渲染，估计G-buffers

        Args:
            frame_dirs: 视频帧目录列表

        Returns:
            G-buffer输出目录列表
        """
        self._log("\n=== Stage 2: Inverse Rendering (Estimate G-buffers) ===")
        self._start_stage_timer("Stage 2: Inverse Rendering")

        gbuffer_output_dirs = []

        for frame_dir in frame_dirs:
            video_name = frame_dir.name
            self._log(f"\nProcessing video: {video_name}")

            # G-buffer输出路径
            gbuffer_output = self.gbuffer_dir / video_name

            # 如果已经处理过，跳过
            gbuffer_frames_dir = gbuffer_output / "gbuffer_frames"
            if gbuffer_frames_dir.exists() and len(list(gbuffer_frames_dir.glob("*"))) > 0:
                self._log(f"Skipping already processed G-buffers: {video_name}")
                gbuffer_output_dirs.append(gbuffer_frames_dir)
                continue

            # 清空GPU缓存
            if self.gpu_monitor:
                self.gpu_monitor.clear_cache()

            python_cmd = self._get_python_cmd()
            offload_flags = "--offload_diffusion_transformer --offload_tokenizer" if self.enable_offload else ""
            cmd = f"""cd {REPO_ROOT} && \\
            CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) {python_cmd} cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \\
                --checkpoint_dir {self.checkpoint_dir} \\
                --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \\
                --dataset_path {frame_dir} \\
                --num_video_frames {self.num_video_frames} \\
                --group_mode folder \\
                --video_save_folder {gbuffer_output} \\
                --save_video True \\
                --save_image True \\
                {offload_flags}
            """

            success, output, elapsed = self._run_command(
                cmd,
                f"Inverse rendering: {video_name}",
                capture_output=True,
                track_stage="inverse_rendering"
            )

            if success and gbuffer_frames_dir.exists():
                gbuffer_output_dirs.append(gbuffer_frames_dir)
                self.stats["timing"][f"{video_name}_inverse"] = elapsed
                self.stats["processed_videos"] += 1
            else:
                self.stats["failed_videos"].append((video_name, "inverse_rendering"))

        self._log(f"\nSuccessfully processed G-buffers for {len(gbuffer_output_dirs)} video(s)")
        self._end_stage_timer("Stage 2: Inverse Rendering")
        return gbuffer_output_dirs

    def run_forward_rendering(self, gbuffer_dirs: List[Path]) -> Dict[str, List[Path]]:
        """
        运行正向渲染，使用不同HDRI重光照

        Args:
            gbuffer_dirs: G-buffer目录列表

        Returns:
            重光照视频路径字典 {video_name: [relit_path1, relit_path2, ...]}
        """
        self._log("\n=== Stage 3: Forward Rendering (Relighting) ===")
        self._start_stage_timer("Stage 3: Forward Rendering")

        relit_videos = {}

        for gbuffer_dir in gbuffer_dirs:
            # 从路径获取视频名称
            # gbuffer_dir 通常是 .../video_name/gbuffer_frames
            video_name = gbuffer_dir.parent.name
            self._log(f"\nRelighting video: {video_name}")

            relit_videos[video_name] = []

            for idx, hdri_idx in enumerate(self.hdri_indices):
                # 输出命名：预定义HDRI用名称，随机HDRI用序号
                if self.use_random_hdri:
                    hdri_name = f"random_{idx}"
                else:
                    hdri_name = HDRI_NAMES.get(hdri_idx, f"hdri_{hdri_idx}")

                output_name = f"{video_name}_{hdri_name}"
                relit_output = self.relit_dir / output_name

                # 如果已经处理过，跳过
                if relit_output.exists() and len(list(relit_output.glob("*.mp4"))) > 0:
                    self._log(f"Skipping already generated relit video: {output_name}")
                    relit_videos[video_name].append(relit_output)
                    continue

                # 清空GPU缓存
                if self.gpu_monitor:
                    self.gpu_monitor.clear_cache()

                python_cmd = self._get_python_cmd()
                offload_flags = "--offload_diffusion_transformer --offload_tokenizer" if self.enable_offload else ""

                # 随机HDRI: use_custom_envmap=False; 预定义HDRI: use_custom_envmap=True
                use_custom_envmap = "False" if self.use_random_hdri else "True"

                cmd = f"""cd {REPO_ROOT} && \\
                CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) {python_cmd} cosmos_predict1/diffusion/inference/inference_forward_renderer.py \\
                    --checkpoint_dir {self.checkpoint_dir} \\
                    --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \\
                    --dataset_path {gbuffer_dir} \\
                    --num_video_frames {self.num_video_frames} \\
                    --envlight_ind {hdri_idx} \\
                    --use_custom_envmap {use_custom_envmap} \\
                    --video_save_folder {relit_output} \\
                    {offload_flags}
                """

                success, output, elapsed = self._run_command(
                    cmd,
                    f"Relighting: {output_name} (HDRI {hdri_idx})",
                    capture_output=True,
                    track_stage="forward_rendering"
                )

                if success and relit_output.exists():
                    relit_videos[video_name].append(relit_output)
                    self.stats["timing"][f"{output_name}_forward"] = elapsed
                else:
                    self.stats["failed_videos"].append((output_name, "forward_rendering"))

        # 统计总生成数
        total_relit = sum(len(v) for v in relit_videos.values())
        self._log(f"\nSuccessfully generated {total_relit} relit video(s)")
        self._end_stage_timer("Stage 3: Forward Rendering")

        return relit_videos

    def run_full_pipeline(self):
        """运行完整的数据增强管线"""
        self._log("\n" + "="*60)
        self._log("Starting Data Augmentation Pipeline")
        self._log("="*60)

        pipeline_start = time.time()

        # 阶段1: 提取帧
        frame_dirs = self.extract_frames_from_videos()

        if len(frame_dirs) == 0:
            self._log("Error: No video frames to process")
            return

        # 阶段2: 逆向渲染
        gbuffer_dirs = self.run_inverse_rendering(frame_dirs)

        if len(gbuffer_dirs) == 0:
            self._log("Error: Inverse rendering failed")
            return

        # 阶段3: 正向渲染
        relit_videos = self.run_forward_rendering(gbuffer_dirs)

        pipeline_elapsed = time.time() - pipeline_start

        # 生成报告
        self.generate_report(pipeline_elapsed, relit_videos)

    def generate_report(self, total_time: float, relit_videos: Dict[str, List[Path]]):
        """生成处理报告"""
        self._log("\n" + "="*60)
        self._log("Data Augmentation Pipeline Completed")
        self._log("="*60)

        # 计算总耗时（从初始化开始）
        total_elapsed = time.time() - self.start_time

        # 统计信息
        total_original = len(relit_videos)
        total_augmented = sum(len(v) for v in relit_videos.values())
        expansion_ratio = total_augmented / total_original if total_original > 0 else 0

        # 计算平均显存使用
        vram_summary = {}
        for stage, measurements in self.stats["vram_usage"].items():
            if measurements:
                avg_used = sum(m['peak_used_gb'] for m in measurements) / len(measurements)
                max_used = max(m['peak_used_gb'] for m in measurements)
                total = measurements[0]['total_gb']
                vram_summary[stage] = {
                    'avg_used_gb': round(avg_used, 2),
                    'max_used_gb': round(max_used, 2),
                    'total_gb': round(total, 2)
                }

        report = {
            "summary": {
                "total_videos": self.stats["total_videos"],
                "processed_videos": self.stats["processed_videos"],
                "failed_videos": len(self.stats["failed_videos"]),
                "total_augmented_videos": total_augmented,
                "expansion_ratio": f"{expansion_ratio:.1f}x",
                "total_time_seconds": round(total_time, 2),
                "total_time_formatted": f"{total_time/60:.1f} minutes",
                "total_elapsed_seconds": round(total_elapsed, 2),
                "total_elapsed_formatted": f"{total_elapsed/60:.1f} minutes",
            },
            "timing": {k: round(v, 2) for k, v in self.stats["timing"].items()},
            "stage_timing": {k: round(v, 2) for k, v in self.stats["stage_timing"].items()},
            "vram_usage": vram_summary,
            "failed_videos": self.stats["failed_videos"],
            "output_structure": {
                "frames": str(self.frames_dir),
                "gbuffers": str(self.gbuffer_dir),
                "relit_videos": str(self.relit_dir),
                "logs": str(self.logs_dir),
            },
            "log_file": str(self.log_file) if self.log_file else None,
            "relit_videos": {
                name: [str(p) for p in paths]
                for name, paths in relit_videos.items()
            }
        }

        # 保存JSON报告
        report_path = self.output_dir / "augmentation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 打印摘要
        self._log("\n📊 Processing Summary:")
        self._log(f"  Original videos: {total_original}")
        self._log(f"  Generated videos: {total_augmented}")
        self._log(f"  Data expansion ratio: {expansion_ratio:.1f}x")
        self._log(f"  Pipeline time: {total_time/60:.1f} minutes")
        self._log(f"  Total elapsed: {total_elapsed/60:.1f} minutes")
        self._log(f"  Failed tasks: {len(self.stats['failed_videos'])}")

        if self.stats["stage_timing"]:
            self._log("\n⏱️  Stage Timing:")
            for stage, duration in self.stats["stage_timing"].items():
                self._log(f"  {stage}: {duration:.2f}s ({duration/60:.2f} min)")

        if vram_summary:
            self._log("\n🎮 GPU Memory Usage:")
            for stage, info in vram_summary.items():
                self._log(f"  {stage}:")
                self._log(f"    Avg used: {info['avg_used_gb']:.2f}GB")
                self._log(f"    Peak used: {info['max_used_gb']:.2f}GB")
                self._log(f"    Total: {info['total_gb']:.2f}GB")

        if len(self.stats["failed_videos"]) > 0:
            self._log("\n⚠️  Failed Tasks:")
            for video_name, stage in self.stats["failed_videos"]:
                self._log(f"  - {video_name} (stage: {stage})")

        self._log(f"\n📄 Detailed report saved to: {report_path}")
        if self.log_file:
            self._log(f"📋 Execution log saved to: {self.log_file}")
        self._log("\n✅ Pipeline execution completed!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Data Augmentation Pipeline - Generate multi-lighting videos using DiffusionRenderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Basic usage (with CPU offload enabled by default)
  python gen_trajs.py --input_video_dir ./videos --output_dir ./output

  # Specify HDRI indices for 3x data augmentation
  python gen_trajs.py --input_video_dir ./videos --output_dir ./output --hdri_indices 0 1 2

  # Custom parameters with lower resolution
  python gen_trajs.py --input_video_dir ./videos --output_dir ./output \\
      --num_frames 33 --frame_rate 16 --resolution 640x360

  # Disable offload for performance testing (uses more GPU memory)
  python gen_trajs.py --input_video_dir ./videos --output_dir ./output --disable_offload

Performance Notes:
  - Offload is ENABLED by default (saves ~2.8GB VRAM for 57-frame mode)
  - Use --disable_offload only for testing or if you have plenty of GPU memory
  - 1-frame mode: offload has minimal impact (~0.1GB)
  - 57-frame mode: offload is recommended (reduces VRAM from 27.7GB to 24.9GB)
        """
    )

    parser.add_argument(
        "--input_video_dir",
        type=str,
        required=True,
        help="Input video directory path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory path"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Model checkpoint directory (default: checkpoints, relative to repo root)"
    )
    parser.add_argument(
        "--hdri_indices",
        type=int,
        nargs="+",
        default=DEFAULT_HDRI_INDICES,
        help=f"HDRI environment light indices (default: {DEFAULT_HDRI_INDICES})"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=57,
        help="Number of frames to process per video (default: 57)"
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=24,
        help="Video frame rate (default: 24)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1280x704",
        help="Video resolution WxH (default: 1280x704)"
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default="cosmos-predict1",
        help="Conda environment name (default: cosmos-predict1)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode, reduce output"
    )
    parser.add_argument(
        "--no_vram_monitor",
        action="store_true",
        help="Disable GPU memory monitoring"
    )
    parser.add_argument(
        "--no_log_file",
        action="store_true",
        help="Disable logging to file (output to console only)"
    )
    parser.add_argument(
        "--disable_offload",
        action="store_true",
        help="Disable CPU offload (use more GPU memory but may be faster for testing)"
    )
    parser.add_argument(
        "--use_random_hdri",
        action="store_true",
        help="Use random HDRI lighting instead of predefined HDRI files"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Limit the number of videos to process (default: None, process all videos)"
    )

    args = parser.parse_args()

    # 创建并运行管线
    pipeline = DataAugmentationPipeline(
        input_video_dir=args.input_video_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        hdri_indices=args.hdri_indices,
        num_video_frames=args.num_frames,
        frame_rate=args.frame_rate,
        resolution=args.resolution,
        conda_env=args.conda_env,
        verbose=not args.quiet,
        monitor_vram=not args.no_vram_monitor,
        log_to_file=not args.no_log_file,
        enable_offload=not args.disable_offload,
        use_random_hdri=args.use_random_hdri,
        max_videos=args.max_videos,
    )

    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
