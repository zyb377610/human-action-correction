"""
命令行标准动作录制脚本

用法:
    python scripts/record_action.py --action squat --name standard_01
    python scripts/record_action.py --action arm_raise --name standard_01 --camera 0
    python scripts/record_action.py --list   # 列出所有已有动作和模板
"""

import argparse
import logging
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.recorder import ActionRecorder
from src.data.template_library import TemplateLibrary
from src.utils.config import get_config


def main():
    parser = argparse.ArgumentParser(
        description="标准动作录制工具 — 交互式录制标准动作模板"
    )
    parser.add_argument(
        "--action", type=str, default=None,
        help="动作类别名称 (如 squat, arm_raise, side_bend)"
    )
    parser.add_argument(
        "--name", type=str, default="standard_01",
        help="模板名称 (默认: standard_01)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="摄像头设备 ID (默认: 0)"
    )
    parser.add_argument(
        "--countdown", type=int, default=3,
        help="倒计时秒数 (默认: 3)"
    )
    parser.add_argument(
        "--max-seconds", type=float, default=30.0,
        help="最大录制时长/秒 (默认: 30)"
    )
    parser.add_argument(
        "--target-frames", type=int, default=60,
        help="归一化帧数 (默认: 60)"
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="保存时不执行预处理"
    )
    parser.add_argument(
        "--templates-dir", type=str, default=None,
        help="模板存储目录 (默认: data/templates/)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有动作类别和模板"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 列出模式
    if args.list:
        library = TemplateLibrary(args.templates_dir)
        actions = library.list_actions()
        if not actions:
            print("模板库为空。使用 --action <name> 开始录制。")
            return

        print(f"\n{'='*50}")
        print(f"  标准动作模板库 ({len(actions)} 个类别)")
        print(f"{'='*50}")
        for action in actions:
            info = library.get_action_info(action)
            templates = library.list_templates(action)
            print(f"\n  [{action}] {info.get('display_name', action)}")
            print(f"    描述: {info.get('description', '-')}")
            print(f"    模板: {len(templates)} 个")
            for t in templates:
                print(f"      - {t}")
        print()
        return

    # 录制模式
    if not args.action:
        # 显示可用动作类别
        config = get_config()
        categories = config.get("actions", {}).get("categories", {})
        if categories:
            print("\n可用动作类别:")
            for name, info in categories.items():
                print(f"  {name}: {info.get('display_name', name)}")
            print(f"\n用法: python {sys.argv[0]} --action <名称> --name <模板名>")
        else:
            print("请指定 --action 参数")
        return

    print(f"\n{'='*50}")
    print(f"  标准动作录制")
    print(f"  动作: {args.action}")
    print(f"  模板: {args.name}")
    print(f"  摄像头: {args.camera}")
    print(f"  倒计时: {args.countdown}s")
    print(f"  最大时长: {args.max_seconds}s")
    print(f"  归一化帧数: {args.target_frames}")
    print(f"  预处理: {'否' if args.no_preprocess else '是'}")
    print(f"{'='*50}\n")

    recorder = ActionRecorder(
        camera_id=args.camera,
        countdown_seconds=args.countdown,
        max_record_seconds=args.max_seconds,
        preprocess=not args.no_preprocess,
        target_frames=args.target_frames,
        templates_dir=args.templates_dir,
    )

    result = recorder.record(args.action, args.name)

    if result:
        print(f"\n✓ 模板保存成功: {args.action}/{args.name} ({result.num_frames} 帧)")
    else:
        print("\n✗ 录制未保存")


if __name__ == "__main__":
    main()
