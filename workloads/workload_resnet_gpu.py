#!/usr/bin/env python3
"""workload_resnet_gpu.py - GPU inference workload (ResNet18 GPU Inference)
Usage: python3 workload_resnet_gpu.py -t 120
Purpose: GPU inference to compare energy profile with workload_resnet.py (CPU)
"""

import torch
import torchvision.models as models
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='GPU ResNet18 inference workload')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save classification results (default: None)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device(args.device)
    dev_idx = device.index if device.index is not None else 0
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}")
    print(f"Batch size: {args.batch}")
    print(f"Duration: {args.timeout}s")

    # Prepare result saving
    save_results = args.output_dir is not None
    result_file = None
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "resnet_gpu_results.csv")
        result_file = open(result_path, "w")
        result_file.write("iteration,batch_idx,class_id,confidence\n")
        print(f"Saving results to: {result_path}")

    # Load model
    print("Loading ResNet18 to GPU...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    # Dummy input (ImageNet size: 3x224x224)
    dummy_input = torch.randn(args.batch, 3, 224, 224, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    torch.cuda.synchronize()

    print("Starting ResNet18 GPU inference...\n")

    start_time = time.time()
    iteration = 0
    total_images = 0

    try:
        with torch.no_grad():
            while time.time() - start_time < args.timeout:
                output = model(dummy_input)
                iteration += 1
                total_images += args.batch

                # Save results: top-1 class id + confidence per image
                if save_results:
                    probs = torch.softmax(output, dim=1)
                    confs, preds = torch.max(probs, dim=1)
                    for i in range(args.batch):
                        result_file.write(
                            f"{iteration},{i},{preds[i].item()},{confs[i].item():.4f}\n")

                if iteration % 500 == 0:
                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    ips = total_images / elapsed
                    print(f"  [{elapsed:.1f}s] images: {total_images} ({ips:.1f} img/s)")

    except KeyboardInterrupt:
        pass

    torch.cuda.synchronize()

    if result_file:
        result_file.close()

    elapsed = time.time() - start_time
    print(f"\nDone. {total_images} images in {elapsed:.1f}s "
          f"({total_images/elapsed:.1f} img/s)")


if __name__ == '__main__':
    main()
