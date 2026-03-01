#!/usr/bin/env python3
"""workload_gpu_llm.py - GPU Memory-heavy workload (GPU LLM Inference)
Usage: python3 workload_gpu_llm.py -t 120 --device cuda:0
Purpose: Complete the SM-heavy (GEMM) vs Memory-heavy (GPU-LLM) spectrum within GPU-dominant workloads.

- KV-cache + weight reads -> HBM bandwidth intensive
- SM utilization lower than GEMM (95%), but high GPU memory bandwidth usage
- Text generation loop maintains steady-state

Prereq: pip install transformers (already installed in venv)
Note: Model auto-downloads on first run (OPT-1.3B: ~2.6GB)
Note: Titan V (12GB HBM2) -> float16 required
"""

import torch
import time
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='GPU LLM inference workload (memory-heavy)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--model', type=str, default='facebook/opt-1.3b',
                        help='HuggingFace model (default: facebook/opt-1.3b)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                        help='Max tokens per generation (default: 128)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save generated texts (default: None)')
    parser.add_argument('--cache-dir', type=str,
                        default='/data/home/optimus/huggingface_cache',
                        help='HuggingFace cache directory (default: /data/home/optimus/huggingface_cache)')
    args = parser.parse_args()

    # Set HF cache path (SSD space insufficient -> use HDD /data)
    os.environ['HF_HOME'] = args.cache_dir

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed")
        print("Run: pip install transformers")
        return

    device = torch.device(args.device)
    dev_idx = device.index if device.index is not None else 0
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}")
    print(f"Model: {args.model}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Duration: {args.timeout}s")

    # Prepare result saving
    save_results = args.output_dir is not None
    result_file = None
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "gpu_llm_results.txt")
        result_file = open(result_path, "w")
        print(f"Saving results to: {result_path}")

    # Load model (float16 for Titan V 12GB)
    print("Loading model (float16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,  # torch <2.6 compatibility (bypass torch.load security issue)
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Long prompt (maximize KV-cache utilization)
    prompt = (
        "The field of artificial intelligence has seen remarkable progress in recent years. "
        "Large language models have demonstrated capabilities in reasoning, coding, and creative writing. "
        "The energy consumption of these models during inference is a critical consideration for "
        "sustainable deployment in cloud data centers. In this paper, we analyze the energy "
        "characteristics of various AI workloads running on shared GPU infrastructure. "
    ) * 3  # Repeat to build long context

    # Check GPU memory usage
    mem_alloc = torch.cuda.memory_allocated(dev_idx) / (1024**3)
    print(f"GPU memory (model loaded): {mem_alloc:.2f} GB")

    # Warm-up (1 generation)
    print("Warming up...")
    warmup_input = tokenizer("Hello", return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        model.generate(warmup_input, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()

    print("Starting GPU LLM inference...\n")

    start_time = time.time()
    iteration = 0
    total_tokens = 0

    try:
        with torch.no_grad():
            while time.time() - start_time < args.timeout:
                # Tokenize
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=512)
                input_ids = inputs["input_ids"].to(device)

                # Text generation (autoregressive -> KV-cache read/write per token)
                output = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

                generated_tokens = output.shape[1] - input_ids.shape[1]
                total_tokens += generated_tokens
                iteration += 1

                # Save results
                if save_results:
                    text = tokenizer.decode(output[0], skip_special_tokens=True)
                    result_file.write(f"--- iteration {iteration} ---\n")
                    result_file.write(text[:500] + "\n\n")

                if iteration % 2 == 0:
                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed
                    mem = torch.cuda.memory_allocated(dev_idx) / (1024**3)
                    print(f"  [{elapsed:.1f}s] iter {iteration} | "
                          f"tokens: {total_tokens} ({tps:.1f} tok/s) | "
                          f"GPU mem: {mem:.2f} GB")

    except KeyboardInterrupt:
        pass

    torch.cuda.synchronize()

    if result_file:
        result_file.close()

    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated(dev_idx) / (1024**3)
    print(f"\nDone. {iteration} iterations, {total_tokens} tokens "
          f"in {elapsed:.1f}s ({total_tokens/elapsed:.1f} tok/s)")
    print(f"GPU peak memory: {mem_peak:.2f} GB")


if __name__ == '__main__':
    main()
