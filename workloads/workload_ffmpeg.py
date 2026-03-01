#!/usr/bin/env python3
"""workload_ffmpeg.py - CPU Compute-heavy workload (x264 Video Encoding)
Usage: python3 workload_ffmpeg.py -t 120
Purpose: Obtain energy/IPC patterns for a realistic CPU compute-heavy workload.
         Validate reproducibility of CPU compute-bound classification against ResNet-CPU (synthetic).

- Encode synthetic video (testsrc2) with libx264 via ffmpeg
- Pure CPU workload (no GPU usage)
- Synthetic source -> no input file needed, 100% reproducible

Prereq: sudo apt install ffmpeg
PID pattern: "libx264" (matches the actual ffmpeg process, not the Python wrapper)
"""

import subprocess
import time
import argparse
import os
import signal


def main():
    parser = argparse.ArgumentParser(description='CPU compute-heavy ffmpeg x264 workload')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--preset', type=str, default='medium',
                        help='x264 preset: ultrafast~veryslow (default: medium)')
    parser.add_argument('--resolution', type=str, default='1920x1080',
                        help='Video resolution (default: 1920x1080)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save encoding stats (default: None)')
    args = parser.parse_args()

    # Check ffmpeg availability
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        version_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
        print(f"ffmpeg: {version_line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("ERROR: ffmpeg not installed")
        print("Run: sudo apt install ffmpeg")
        return

    print(f"Preset: {args.preset}")
    print(f"Resolution: {args.resolution}")
    print(f"Duration: {args.timeout}s")

    # Prepare result saving
    save_results = args.output_dir is not None
    stats_lines = []
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)

    # Synthetic video source -> x264 encoding -> /dev/null
    # testsrc2: sufficiently complex test pattern (color bars, counters, etc.)
    # duration=9999: very long source -> controlled by timeout
    ffmpeg_cmd = [
        "ffmpeg",
        "-nostdin",                # Prevent stdin reads
        "-loglevel", "error",      # Suppress unnecessary output
        "-f", "lavfi",
        "-i", f"testsrc2=duration=9999:size={args.resolution}:rate=30",
        "-c:v", "libx264",
        "-preset", args.preset,
        "-f", "null",
        "-y",
        "/dev/null",
    ]

    print(f"Starting ffmpeg x264 encoding...\n")
    print(f"  cmd: {' '.join(ffmpeg_cmd)}")

    start_time = time.time()
    round_count = 0

    # Start ffmpeg
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    round_count += 1

    try:
        while time.time() - start_time < args.timeout:
            # Restart ffmpeg if it exits unexpectedly
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] ffmpeg exited (code={ret}), restarting...")
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                round_count += 1

            time.sleep(5)
            elapsed = time.time() - start_time
            msg = f"  [{elapsed:.1f}s] encoding round {round_count} in progress..."
            print(msg)
            if save_results:
                stats_lines.append(msg)

    except KeyboardInterrupt:
        pass

    # Terminate ffmpeg
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass

    if save_results:
        stats_path = os.path.join(args.output_dir, "ffmpeg_stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"preset: {args.preset}\n")
            f.write(f"resolution: {args.resolution}\n")
            f.write(f"rounds: {round_count}\n")
            f.write(f"total_time: {time.time() - start_time:.1f}s\n\n")
            for line in stats_lines:
                f.write(line + "\n")
        print(f"Stats saved to: {stats_path}")

    elapsed = time.time() - start_time
    print(f"\nDone. {round_count} round(s) in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
