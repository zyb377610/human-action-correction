"""
离线视频处理脚本

输入视频文件 → 提取关键点 → 保存 PoseSequence → 输出标注视频
用法: python scripts/process_video.py <input_video> [--output_dir outputs]
"""
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.pose_estimation import (
    PoseEstimator, FileSource, FeatureExtractor,
    draw_skeleton, draw_angles, draw_fps,
)


def main():
    parser = argparse.ArgumentParser(description="离线视频姿态估计处理")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("--output_dir", default="outputs", help="输出目录")
    parser.add_argument("--no_video", action="store_true", help="不输出标注视频")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input))[0]

    print(f"输入视频: {args.input}")
    print(f"输出目录: {args.output_dir}")

    # 1. 姿态估计
    with PoseEstimator() as estimator:
        with FileSource(args.input) as source:
            print(f"视频信息: {source.width}x{source.height}@{source.fps:.1f}fps, "
                  f"{source.total_frames} 帧")

            sequence = estimator.estimate_video(source)

    print(f"有效姿态帧: {sequence.num_frames}")

    # 2. 保存关键点数据
    npy_path = os.path.join(args.output_dir, f"{basename}_pose.npy")
    json_path = os.path.join(args.output_dir, f"{basename}_pose.json")
    sequence.save(npy_path)
    sequence.save(json_path)
    print(f"关键点数据已保存: {npy_path}, {json_path}")

    # 3. 提取特征
    extractor = FeatureExtractor()
    features = extractor.extract_sequence_features(sequence)
    print(f"特征矩阵: {features.shape} (帧数 x 特征维度)")

    # 4. 输出标注视频
    if not args.no_video:
        out_path = os.path.join(args.output_dir, f"{basename}_annotated.mp4")
        source2 = FileSource(args.input)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, source2.fps,
                                 (source2.width, source2.height))
        pose_idx = 0
        while True:
            success, frame = source2.read()
            if not success:
                break
            if pose_idx < sequence.num_frames:
                pf = sequence.frames[pose_idx]
                draw_skeleton(frame, pf)
                draw_angles(frame, pf)
                pose_idx += 1
            writer.write(frame)
        writer.release()
        source2.release()
        print(f"标注视频已保存: {out_path}")

    print("处理完成!")


if __name__ == "__main__":
    main()
