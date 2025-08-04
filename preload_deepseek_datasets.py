#!/usr/bin/env python3
"""
Script to download and preprocess datasets using DeepSeek tokenizers for Megatron-LM training.

This script:
1. Downloads datasets (SlimPajama, Wikipedia, etc.) from HuggingFace
2. Uses DeepSeek tokenizers for preprocessing
3. Converts to JSONL format
4. Preprocesses using Megatron's preprocess_data.py tool
5. Creates the binary indexed datasets required for training

Usage:
    python preload_deepseek_datasets.py --dataset slimpajama --output-dir /path/to/data --deepseek-model deepseek-ai/DeepSeek-V2-Lite
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Download and preprocess datasets using DeepSeek tokenizers for Megatron-LM"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["slimpajama", "wikipedia", "custom"],
        help="Dataset to download and preprocess",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save processed data"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help="Directory to save raw JSONL files (defaults to output-dir/raw)",
    )

    # DeepSeek model arguments
    parser.add_argument(
        "--deepseek-model",
        type=str,
        required=True,
        choices=[
            "deepseek-ai/DeepSeek-V2-Lite",
            "deepseek-ai/DeepSeek-V2",
            "deepseek-ai/DeepSeek-V3",
        ],
        help="DeepSeek model to use for tokenizer",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded tokenizer",
    )

    # Dataset-specific arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for Wikipedia (en, es, fr, de, etc.)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20231101",
        help="Wikipedia dump date (YYYYMMDD format)",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to download"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    # Processing arguments
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(), help="Number of worker processes"
    )
    parser.add_argument(
        "--min-length", type=int, default=100, help="Minimum text length in characters"
    )
    parser.add_argument(
        "--max-length", type=int, default=None, help="Maximum text length in characters"
    )
    parser.add_argument(
        "--append-eod", action="store_true", help="Append end-of-document token"
    )
    parser.add_argument(
        "--split-sentences", action="store_true", help="Split documents into sentences"
    )
    parser.add_argument(
        "--keep-newlines",
        action="store_true",
        help="Keep newlines when splitting sentences",
    )
    parser.add_argument(
        "--clean-text",
        action="store_true",
        default=True,
        help="Clean markup and formatting (for Wikipedia)",
    )

    # Custom dataset arguments
    parser.add_argument(
        "--custom-dataset-name",
        type=str,
        default=None,
        help="Custom HuggingFace dataset name",
    )
    parser.add_argument(
        "--custom-text-field",
        type=str,
        default="text",
        help="Field name containing text in custom dataset",
    )

    # Megatron paths
    parser.add_argument(
        "--megatron-path",
        type=str,
        default="/home/less/Megatron-LM",
        help="Path to Megatron-LM repository",
    )

    return parser.parse_args()


def setup_deepseek_tokenizer(args):
    """Setup DeepSeek tokenizer and extract tokenizer files."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers library not found. Please install with:")
        print("pip install transformers")
        sys.exit(1)

    print(f"Loading DeepSeek tokenizer: {args.deepseek_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.deepseek_model, cache_dir=args.tokenizer_cache_dir, trust_remote_code=True
    )

    # Create tokenizer directory
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Save tokenizer files
    tokenizer.save_pretrained(tokenizer_dir)

    print(f"Tokenizer saved to: {tokenizer_dir}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # Determine tokenizer type and files for Megatron
    tokenizer_files = os.listdir(tokenizer_dir)

    if "tokenizer.model" in tokenizer_files:
        # SentencePiece tokenizer
        tokenizer_type = "GPTSentencePieceTokenizer"
        tokenizer_model = os.path.join(tokenizer_dir, "tokenizer.model")
        vocab_file = None
        merge_file = None
    elif "vocab.json" in tokenizer_files and "merges.txt" in tokenizer_files:
        # BPE tokenizer
        tokenizer_type = "GPT2BPETokenizer"
        tokenizer_model = None
        vocab_file = os.path.join(tokenizer_dir, "vocab.json")
        merge_file = os.path.join(tokenizer_dir, "merges.txt")
    else:
        # HuggingFace tokenizer (fallback)
        tokenizer_type = "HuggingFaceTokenizer"
        tokenizer_model = tokenizer_dir
        vocab_file = None
        merge_file = None

    print(f"Detected tokenizer type: {tokenizer_type}")

    return {
        "type": tokenizer_type,
        "model": tokenizer_model,
        "vocab_file": vocab_file,
        "merge_file": merge_file,
        "tokenizer_dir": tokenizer_dir,
        "tokenizer": tokenizer,
    }


def clean_wikipedia_text(text):
    """Clean Wikipedia text by removing markup and formatting."""
    if not text:
        return ""

    # Remove common Wikipedia markup patterns
    text = re.sub(r"\[\d+\]", "", text)  # References like [1]
    text = re.sub(r"\[citation needed\]", "", text)
    text = re.sub(r"\[clarification needed\]", "", text)

    # Remove file references
    text = re.sub(r"\[\[File:.*?\]\]", "", text)
    text = re.sub(r"\[\[Image:.*?\]\]", "", text)

    # Clean up wiki links - keep the display text
    text = re.sub(
        r"\[\[([^|\]]+)\|([^|\]]+)\]\]", r"\2", text
    )  # [[link|display]] -> display
    text = re.sub(r"\[\[([^|\]]+)\]\]", r"\1", text)  # [[link]] -> link

    # Remove category links
    text = re.sub(r"\[\[Category:.*?\]\]", "", text)

    # Remove templates (basic cleanup)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)

    # Clean up multiple spaces and newlines
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Multiple newlines -> double newline
    text = re.sub(r" +", " ", text)  # Multiple spaces -> single space

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def download_and_convert_dataset(args):
    """Download dataset and convert to JSONL format."""

    raw_data_dir = args.raw_data_dir or os.path.join(args.output_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    # Configure dataset based on type
    if args.dataset == "slimpajama":
        dataset_name = args.dataset_name or "cerebras/SlimPajama-627B"
        dataset_config = None
        text_field = "text"
        jsonl_file = os.path.join(raw_data_dir, f"slimpajama_{args.split}.jsonl")

    elif args.dataset == "wikipedia":
        dataset_name = args.dataset_name or "wikimedia/wikipedia"
        dataset_config = f"{args.date}.{args.language}"
        text_field = "text"
        jsonl_file = os.path.join(
            raw_data_dir, f"wikipedia_{args.language}_{args.date}.jsonl"
        )

    elif args.dataset == "custom":
        if not args.custom_dataset_name:
            raise ValueError("--custom-dataset-name is required for custom datasets")
        dataset_name = args.custom_dataset_name
        dataset_config = None
        text_field = args.custom_text_field
        safe_name = args.custom_dataset_name.replace("/", "_")
        jsonl_file = os.path.join(raw_data_dir, f"{safe_name}_{args.split}.jsonl")

    print(f"Downloading {dataset_name}")
    if dataset_config:
        print(f"Configuration: {dataset_config}")

    try:
        # Load dataset - always use streaming for large datasets like SlimPajama
        # and handle max_samples in the processing loop
        if dataset_config:
            dataset = load_dataset(
                dataset_name, dataset_config, split=args.split, streaming=True
            )
        else:
            dataset = load_dataset(dataset_name, split=args.split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if args.dataset == "wikipedia":
            print(f"Try checking available configurations for Wikipedia")
        sys.exit(1)

    print(f"Converting to JSONL format: {jsonl_file}")
    print(f"Text field: {text_field}")
    if args.min_length:
        print(f"Minimum length: {args.min_length} characters")
    if args.max_length:
        print(f"Maximum length: {args.max_length} characters")

    processed_count = 0
    filtered_count = 0

    with open(jsonl_file, "w", encoding="utf-8") as f:
        # Use streaming dataset and handle max_samples in the loop
        for sample in dataset:
            text = sample.get(text_field, "")

            # Clean text if requested (mainly for Wikipedia)
            if args.clean_text and args.dataset == "wikipedia":
                text = clean_wikipedia_text(text)

            # Filter by length
            if len(text) < args.min_length:
                filtered_count += 1
                continue

            if args.max_length and len(text) > args.max_length:
                text = text[: args.max_length]

            # Create JSON object
            json_obj = {"text": text}

            # Add additional fields for some datasets
            if args.dataset == "wikipedia":
                json_obj.update(
                    {
                        "title": sample.get("title", ""),
                        "url": sample.get("url", ""),
                        "id": sample.get("id", ""),
                    }
                )

            f.write(json.dumps(json_obj) + "\n")
            processed_count += 1

            # Check if we've reached max_samples
            if args.max_samples and processed_count >= args.max_samples:
                print(f"Reached max_samples limit: {args.max_samples}")
                break

            if processed_count % 10000 == 0:
                print(
                    f"Processed {processed_count} samples, filtered {filtered_count}..."
                )

    print(f"JSONL conversion complete: {jsonl_file}")
    print(f"Total samples processed: {processed_count}")
    print(f"Samples filtered (too short): {filtered_count}")
    return jsonl_file


def preprocess_with_megatron(args, jsonl_file, tokenizer_info):
    """Preprocess JSONL file using Megatron's preprocess_data.py."""

    preprocess_script = os.path.join(args.megatron_path, "tools", "preprocess_data.py")

    if not os.path.exists(preprocess_script):
        raise FileNotFoundError(
            f"Megatron preprocess script not found: {preprocess_script}"
        )

    # Build output prefix
    if args.dataset == "wikipedia":
        output_prefix = os.path.join(args.output_dir, f"wikipedia_{args.language}")
    elif args.dataset == "slimpajama":
        output_prefix = os.path.join(args.output_dir, "slimpajama")
    else:
        safe_name = (
            args.custom_dataset_name.replace("/", "_")
            if args.custom_dataset_name
            else "custom"
        )
        output_prefix = os.path.join(args.output_dir, safe_name)

    # Build command
    cmd = [
        "python",
        preprocess_script,
        "--input",
        jsonl_file,
        "--output-prefix",
        output_prefix,
        "--tokenizer-type",
        tokenizer_info["type"],
        "--workers",
        str(args.workers),
        "--json-keys",
        "text",
    ]

    # Add tokenizer-specific arguments
    if tokenizer_info["model"]:
        cmd.extend(["--tokenizer-model", tokenizer_info["model"]])
    if tokenizer_info["vocab_file"]:
        cmd.extend(["--vocab-file", tokenizer_info["vocab_file"]])
    if tokenizer_info["merge_file"]:
        cmd.extend(["--merge-file", tokenizer_info["merge_file"]])

    # Add optional flags
    if args.append_eod:
        cmd.append("--append-eod")
    if args.split_sentences:
        cmd.append("--split-sentences")
    if args.keep_newlines:
        cmd.append("--keep-newlines")

    print(f"Running Megatron preprocessing...")
    print(f"Command: {' '.join(cmd)}")

    # Run preprocessing
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Preprocessing failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)

    print("Preprocessing completed successfully!")

    # List generated files
    output_files = []
    for ext in [".bin", ".idx"]:
        file_path = f"{output_prefix}_text_document{ext}"
        if os.path.exists(file_path):
            output_files.append(file_path)
            print(f"Generated: {file_path}")

    return output_files, output_prefix


def create_data_config(args, output_files, output_prefix, tokenizer_info):
    """Create a configuration file for easy reference."""

    config = {
        "dataset_name": args.dataset,
        "deepseek_model": args.deepseek_model,
        "tokenizer_type": tokenizer_info["type"],
        "tokenizer_model": tokenizer_info["model"],
        "vocab_file": tokenizer_info["vocab_file"],
        "merge_file": tokenizer_info["merge_file"],
        "tokenizer_dir": tokenizer_info["tokenizer_dir"],
        "data_path": f"{output_prefix}_text_document",
        "output_files": output_files,
        "dataset_config": {
            "dataset_name": (
                args.dataset_name if hasattr(args, "dataset_name") else None
            ),
            "language": args.language if args.dataset == "wikipedia" else None,
            "date": args.date if args.dataset == "wikipedia" else None,
            "split": args.split,
            "custom_dataset_name": (
                args.custom_dataset_name if args.dataset == "custom" else None
            ),
            "custom_text_field": (
                args.custom_text_field if args.dataset == "custom" else None
            ),
        },
        "preprocessing_args": {
            "workers": args.workers,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "append_eod": args.append_eod,
            "split_sentences": args.split_sentences,
            "keep_newlines": args.keep_newlines,
            "clean_text": args.clean_text,
        },
    }

    config_file = os.path.join(args.output_dir, f"{args.dataset}_deepseek_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_file}")
    return config_file


def main():
    args = get_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Dataset Preprocessing with DeepSeek Tokenizers for Megatron-LM")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"DeepSeek model: {args.deepseek_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Workers: {args.workers}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 70)

    # Step 1: Setup DeepSeek tokenizer
    tokenizer_info = setup_deepseek_tokenizer(args)

    # Step 2: Download and convert dataset to JSONL
    jsonl_file = download_and_convert_dataset(args)

    # Step 3: Preprocess with Megatron
    output_files, output_prefix = preprocess_with_megatron(
        args, jsonl_file, tokenizer_info
    )

    # Step 4: Create configuration file
    config_file = create_data_config(args, output_files, output_prefix, tokenizer_info)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Data path for training: {output_prefix}_text_document")
    print(f"Tokenizer directory: {tokenizer_info['tokenizer_dir']}")
    print(f"Configuration file: {config_file}")
    print("\nTo use in Megatron training, set:")
    print(f"  --data-path {output_prefix}_text_document")
    print(f"  --tokenizer-type {tokenizer_info['type']}")
    if tokenizer_info["model"]:
        print(f"  --tokenizer-model {tokenizer_info['model']}")
    if tokenizer_info["vocab_file"]:
        print(f"  --vocab-file {tokenizer_info['vocab_file']}")
        print(f"  --merge-file {tokenizer_info['merge_file']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
