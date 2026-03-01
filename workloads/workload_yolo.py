#!/usr/bin/env python3
"""workload_yolo.py - Mixed workload (YOLOv8 Nano repeated inference)
Usage: python3 workload_yolo.py -t 120
Path: Run from ~/optimus/yolo/ on the GPU server, or activate venv first
Prereq: pip install ultralytics (already installed in ~/optimus/yolo/venv)
"""

import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Nano workload (mixed CPU+GPU)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--source', type=str, default='test_video.mp4',
                        help='Video or image source (default: test_video.mp4)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='(unused, for compatibility with run scripts)')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed")
        print("Run: pip install ultralytics")
        print("Or activate venv: source ~/optimus/yolo/venv/bin/activate")
        return

    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Duration: {args.timeout}s")
    print("Loading model...")

    model = YOLO(args.model)

    print("Starting YOLO workload...\n")

    start_time = time.time()
    iteration = 0
    total_frames = 0

    try:
        while time.time() - start_time < args.timeout:
            results = model.predict(
                source=args.source,
                device=args.device,
                verbose=False,
                stream=True,
                save=True,
                exist_ok=True,
            )
            for r in results:
                total_frames += 1
                if time.time() - start_time >= args.timeout:
                    break

            iteration += 1
            elapsed = time.time() - start_time
            fps = total_frames / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.1f}s] round: {iteration} | "
                  f"frames: {total_frames} ({fps:.1f} fps)")

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.1f} fps)")


if __name__ == '__main__':
    main()
