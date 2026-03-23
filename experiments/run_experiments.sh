#!/bin/bash

# 用法: ./run_experiments.sh [GPU_ID]
# 对应文档: experiments/实验计划.md

# 获取GPU ID (默认使用GPU 0)
GPU_ID=${1:-0}

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 创建实验输出目录
mkdir -p experiments

echo "========================================="
echo "实验时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "使用GPU: $GPU_ID"
echo "========================================="
echo ""

# 定义实验数组
# 格式: 输入目录|输出目录|帧数|分辨率|HDRI索引|额外参数|实验描述
declare -a EXPERIMENTS=(
    "asset/examples/video_examples|experiments/exp_01_57f_offload_4hdri|57|1280x704|0 1 2 3||4视频,57帧,4HDRI,offload"
    "asset/examples/video_examples|experiments/exp_02_57f_1video_4hdri|57|1280x704|0 1 2 3|--max_videos 1|1视频,57帧,4HDRI,offload"
    "asset/examples/video_examples|experiments/exp_03_57f_1video_1hdri|57|1280x704|0|--max_videos 1|1视频,57帧,1HDRI,offload"
    "asset/examples/video_examples|experiments/exp_04_57f_lowres_4hdri|57|640x360|0 1 2 3||4视频,57帧,4HDRI,低分辨率"
    "asset/examples/video_examples|experiments/exp_05_57f_no_offload_4hdri|57|1280x704|0 1 2 3|--disable_offload|4视频,57帧,4HDRI,无offload"
    "asset/examples/video_examples|experiments/exp_06_57f_random_hdri|57|1280x704|0 1 2 3|--use_random_hdri --disable_offload|4视频,57帧,随机HDRI"
    "asset/examples/image_examples|experiments/exp_07_img_4hdri|1|1280x704|0 1 2 3|--disable_offload|16图像,4HDRI,无offload"
    "asset/examples/image_examples|experiments/exp_08_img_random_hdri|1|1280x704|0 1 2 3|--use_random_hdri --disable_offload|16图像,随机HDRI"
)

# 记录开始时间
START_TIME=$(date +%s)

# 顺序执行所有实验
exp_num=1
for exp in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r input output frames resolution hdri flags description <<< "$exp"

    exp_name=$(basename "$output")
    log_file="experiments/${exp_name}.out"

    echo "========================================="
    echo "实验 ${exp_num}/8: ${exp_name}"
    echo "  描述: ${description}"
    echo "  输入: ${input}"
    echo "  帧数: ${frames}"
    echo "  分辨率: ${resolution}"
    echo "  HDRI: ${hdri}"
    [ -n "$flags" ] && echo "  参数: ${flags}"
    echo "========================================="

    # 记录单个实验开始时间
    exp_start_time=$(date +%s)

    # 运行实验
    CUDA_VISIBLE_DEVICES=$GPU_ID python gen_trajs.py \
        --input_video_dir "$input" \
        --output_dir "$output" \
        --num_frames "$frames" \
        --resolution "$resolution" \
        --hdri_indices $hdri \
        $flags \
        > "$log_file" 2>&1

    # 检查实验是否成功
    exit_code=$?
    exp_end_time=$(date +%s)
    exp_duration=$((exp_end_time - exp_start_time))

    if [ $exit_code -eq 0 ]; then
        echo "  ✓ 实验完成: ${exp_name}"
        echo "  耗时: $((exp_duration / 60))分钟$((exp_duration % 60))秒"
        echo "  日志: ${log_file}"
    else
        echo "  ✗ 实验失败: ${exp_name} (退出码: $exit_code)"
        echo "  日志: ${log_file}"
        echo "  是否继续下一个实验? [y/n]"
        read -r continue_choice
        if [ "$continue_choice" != "y" ]; then
            echo "实验序列已中断"
            exit 1
        fi
    fi

    echo ""
    ((exp_num++))
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "========================================="
echo "所有实验已完成！"
echo "========================================="
echo ""
echo "总耗时: $((TOTAL_DURATION / 3600))小时$((TOTAL_DURATION % 3600 / 60))分钟"
echo "开始时间: $(date -d @$START_TIME '+%Y-%m-%d %H:%M:%S')"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "结果检查:"
echo "  列出报告: ls -lh experiments/*/augmentation_report.json"
echo "  查看报告: cat experiments/exp_01_57f_offload_4hdri/augmentation_report.json"
echo "  统计视频: find experiments/*/relit_videos -name 'video.mp4' | wc -l"
echo ""

# 保存实验信息
cat > experiments/experiment_info.txt <<EOF
实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')
使用GPU: $GPU_ID
主机名: $(hostname)
工作目录: $(pwd)
总耗时: $((TOTAL_DURATION / 3600))小时$((TOTAL_DURATION % 3600 / 60))分钟

已完成的实验:
EOF

exp_num=1
for exp in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r input output frames resolution hdri flags description <<< "$exp"
    exp_name=$(basename "$output")
    echo "EXP-0${exp_num}: ${exp_name} - ${description}" >> experiments/experiment_info.txt
    ((exp_num++))
done

echo "实验信息已保存到: experiments/experiment_info.txt"
echo ""
