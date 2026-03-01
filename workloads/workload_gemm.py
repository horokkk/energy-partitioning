#!/usr/bin/env python3
"""workload_gemm.py - GPU-dominant workload (Large Matrix Multiplication)
Usage: python3 workload_gemm.py -t 120
Purpose: Maximize GPU load to generate a workload with high GPU power ratio
"""

import torch
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='GPU-dominant GEMM workload')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('-s', '--size', type=int, default=4096,
                        help='Matrix size NxN (default: 4096)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save computation results (default: None)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = torch.device(args.device)
    dev_idx = device.index if device.index is not None else 0
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}")
    print(f"Matrix size: {args.size}x{args.size}")
    print(f"Duration: {args.timeout}s")
    # Prepare result saving
    save_results = args.output_dir is not None
    result_file = None
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "gemm_results.csv")
        result_file = open(result_path, "w")
        result_file.write("iteration,frobenius_norm,trace,max_val\n")
        print(f"Saving results to: {result_path}")

    print("Starting GEMM workload...\n")

    # GPU warm-up
    A = torch.randn(args.size, args.size, device=device, dtype=torch.float32)
    B = torch.randn(args.size, args.size, device=device, dtype=torch.float32)
    torch.cuda.synchronize()

    start_time = time.time()
    iteration = 0

    try:
        while time.time() - start_time < args.timeout:
            C = torch.mm(A, B)
            torch.cuda.synchronize()
            iteration += 1

            # Save results: matrix multiplication stats (every 50 iterations)
            if save_results and iteration % 50 == 0:
                norm = torch.norm(C).item()
                tr = torch.trace(C).item()
                mx = C.max().item()
                result_file.write(f"{iteration},{norm:.2f},{tr:.2f},{mx:.2f}\n")

            if iteration % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.1f}s] iterations: {iteration}")

    except KeyboardInterrupt:
        pass

    if result_file:
        result_file.close()

    elapsed = time.time() - start_time
    print(f"\nDone. {iteration} iterations in {elapsed:.1f}s "
          f"({iteration/elapsed:.1f} iter/s)")


if __name__ == '__main__':
    main()
