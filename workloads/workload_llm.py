#!/usr/bin/env python3
"""workload_llm.py - CPU+Memory-dominant workload (LLM Tokenization + Small Inference)
Usage: python3 workload_llm.py -t 120
Purpose: Load CPU and memory to generate a workload with high DRAM power ratio

Prereq: pip install transformers
"""

import torch
import time
import argparse
import os


def check_transformers():
    try:
        import transformers
        return True
    except ImportError:
        print("ERROR: transformers not installed")
        print("Run: pip install transformers")
        return False


def main():
    parser = argparse.ArgumentParser(description='CPU+Memory dominant LLM workload')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Model name (default: gpt2)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save generated texts (default: None)')
    args = parser.parse_args()

    if not check_transformers():
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device('cpu')
    print(f"Device: CPU only")
    print(f"Model: {args.model}")
    print(f"Duration: {args.timeout}s")

    # Load model & tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate long text (to load the tokenizer)
    base_text = (
        "Artificial intelligence is transforming every aspect of modern society. "
        "From healthcare to transportation, machine learning algorithms are being deployed "
        "to solve complex problems that were previously considered intractable. "
        "Deep neural networks have shown remarkable capabilities in understanding natural language, "
        "recognizing images, and generating creative content. "
    )
    long_text = base_text * 50  # Long text to stress the tokenizer

    # Prepare result saving
    save_results = args.output_dir is not None
    result_file = None
    if save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "llm_results.txt")
        result_file = open(result_path, "w")
        print(f"Saving results to: {result_path}")

    print("Starting LLM workload (tokenization + inference)...\n")

    start_time = time.time()
    iteration = 0
    total_tokens_processed = 0

    try:
        with torch.no_grad():
            while time.time() - start_time < args.timeout:
                # Phase 1: Bulk tokenization (CPU + memory intensive)
                tokens = tokenizer(long_text, return_tensors='pt',
                                   truncation=True, max_length=512,
                                   padding='max_length')
                total_tokens_processed += tokens['input_ids'].shape[1]

                # Phase 2: Model inference (CPU + memory intensive)
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)

                # Phase 3: Text generation (sequential decoding - memory intensive)
                generated = model.generate(
                    input_ids[:, :20],  # Generate from short prompt
                    max_new_tokens=50,
                    do_sample=False
                )
                total_tokens_processed += generated.shape[1]

                # Save results: generated text
                if save_results:
                    text = tokenizer.decode(generated[0], skip_special_tokens=True)
                    result_file.write(f"--- iteration {iteration + 1} ---\n")
                    result_file.write(text + "\n\n")

                iteration += 1

                if iteration % 5 == 0:
                    elapsed = time.time() - start_time
                    tps = total_tokens_processed / elapsed
                    print(f"  [{elapsed:.1f}s] iterations: {iteration} | "
                          f"tokens: {total_tokens_processed} ({tps:.0f} tok/s)")

    except KeyboardInterrupt:
        pass

    if result_file:
        result_file.close()

    elapsed = time.time() - start_time
    print(f"\nDone. {iteration} iterations, {total_tokens_processed} tokens "
          f"in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
