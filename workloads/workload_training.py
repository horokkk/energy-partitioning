#!/usr/bin/env python3
"""workload_training.py - GPU Training workload (ResNet-18 CIFAR-10 Fine-tuning)
Usage: python3 workload_training.py -t 120 --device cuda:0
Purpose: Demonstrate the unique energy structure of training (backward pass, increased
         GPU memory footprint) compared to inference.

- forward + backward + optimizer step -> 2-3x GPU memory vs inference
- DataLoader(num_workers=2) -> concurrent CPU preprocessing + GPU training
- CIFAR-10 (32x32 -> 224x224 resize) -> ResNet-18 transfer learning

Prereq: pip install torchvision (already installed in venv)
Note: CIFAR-10 auto-downloads on first run (~170MB)
Note: DataLoader workers spawn separate PIDs -> per-PID CPU% tracks main only (supplement with system RAPL)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='GPU Training workload (ResNet-18 CIFAR-10)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('-b', '--batch', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--workers', type=int, default=2,
                        help='DataLoader num_workers (default: 2)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save training logs (default: None)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    device = torch.device(args.device)
    dev_idx = device.index if device.index is not None else 0
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}")
    print(f"Batch size: {args.batch}")
    print(f"DataLoader workers: {args.workers}")
    print(f"Duration: {args.timeout}s")

    # Prepare result saving
    save_results = args.output_dir is not None
    result_file = None
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "training_log.csv")
        result_file = open(result_path, "w")
        result_file.write("epoch,step,loss,lr,images_per_sec\n")
        print(f"Saving results to: {result_path}")

    # CIFAR-10 dataset (32x32 -> 224x224 resize to fit ResNet18)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    print("Loading CIFAR-10 dataset...")
    data_dir = os.path.join(os.path.expanduser("~"), ".cache", "cifar10")
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # ResNet-18 pretrained -> CIFAR-10 fine-tune (replace final FC layer only)
    print("Loading ResNet-18 (pretrained, fine-tune mode)...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10: 10 classes
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Warm-up (1 batch)
    warmup_input = torch.randn(args.batch, 3, 224, 224, device=device)
    warmup_target = torch.zeros(args.batch, dtype=torch.long, device=device)
    output = model(warmup_input)
    loss = criterion(output, warmup_target)
    loss.backward()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    mem_alloc = torch.cuda.memory_allocated(dev_idx) / (1024**3)
    print(f"GPU memory after warmup: {mem_alloc:.2f} GB")
    print("Starting training...\n")

    start_time = time.time()
    epoch = 0
    total_steps = 0
    total_images = 0

    try:
        while time.time() - start_time < args.timeout:
            epoch += 1
            for inputs, targets in trainloader:
                if time.time() - start_time >= args.timeout:
                    break

                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_steps += 1
                total_images += inputs.size(0)

                if total_steps % 20 == 0:
                    elapsed = time.time() - start_time
                    ips = total_images / elapsed
                    print(f"  [{elapsed:.1f}s] epoch {epoch} step {total_steps} | "
                          f"loss={loss.item():.4f} | {ips:.1f} img/s")

                    if save_results:
                        lr = optimizer.param_groups[0]['lr']
                        result_file.write(
                            f"{epoch},{total_steps},{loss.item():.4f},{lr},{ips:.1f}\n")

    except KeyboardInterrupt:
        pass

    torch.cuda.synchronize()

    if result_file:
        result_file.close()

    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated(dev_idx) / (1024**3)
    print(f"\nDone. {total_images} images, {total_steps} steps, {epoch} epochs "
          f"in {elapsed:.1f}s ({total_images/elapsed:.1f} img/s)")
    print(f"GPU peak memory: {mem_peak:.2f} GB")


if __name__ == '__main__':
    main()
