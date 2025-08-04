# DeepSeek Dataset Preprocessing for Megatron-LM

This guide explains how to download and preprocess datasets using DeepSeek tokenizers for Megatron-LM training, specifically for DeepSeek V2 Lite, V2, and V3 models.

## Overview

DeepSeek models use specific tokenizers that are optimized for their architecture. This script automatically downloads the appropriate DeepSeek tokenizer and uses it to preprocess datasets for training with Megatron-LM.

## Supported DeepSeek Models

- **DeepSeek-V2-Lite**: `deepseek-ai/DeepSeek-V2-Lite`
- **DeepSeek-V2**: `deepseek-ai/DeepSeek-V2`
- **DeepSeek-V3**: `deepseek-ai/DeepSeek-V3`

## Supported Datasets

- **SlimPajama**: High-quality web text (627B tokens)
- **Wikipedia**: Multi-language encyclopedic content
- **Custom**: Any HuggingFace dataset

## Prerequisites

1. **Python Dependencies**: Install required packages
   ```bash
   pip install datasets tqdm huggingface_hub transformers
   ```

2. **Storage**: Ensure sufficient disk space:
   - SlimPajama: ~1.2TB raw, ~300-500GB processed
   - Wikipedia (English): ~20GB raw, ~6-8GB processed
   - Custom datasets: varies

## Quick Start

### 1. SlimPajama with DeepSeek-V2-Lite

```bash
python preload_deepseek_datasets.py \
    --dataset slimpajama \
    --deepseek-model deepseek-ai/DeepSeek-V2-Lite \
    --output-dir /path/to/processed/data \
    --workers 32 \
    --append-eod
```

### 2. Wikipedia with DeepSeek-V3

```bash
python preload_deepseek_datasets.py \
    --dataset wikipedia \
    --deepseek-model deepseek-ai/DeepSeek-V3 \
    --output-dir /path/to/processed/data \
    --language en \
    --workers 16 \
    --append-eod
```

### 3. Custom Dataset with DeepSeek-V2

```bash
python preload_deepseek_datasets.py \
    --dataset custom \
    --deepseek-model deepseek-ai/DeepSeek-V2 \
    --custom-dataset-name your/dataset-name \
    --custom-text-field content \
    --output-dir /path/to/processed/data \
    --workers 16 \
    --append-eod
```

### 4. Test with Small Sample

```bash
python preload_deepseek_datasets.py \
    --dataset slimpajama \
    --deepseek-model deepseek-ai/DeepSeek-V2-Lite \
    --output-dir /tmp/deepseek_test \
    --max-samples 1000 \
    --workers 4
```

## Command Line Options

### Required Arguments
- `--dataset`: Dataset type (`slimpajama`, `wikipedia`, `custom`)
- `--deepseek-model`: DeepSeek model for tokenizer
- `--output-dir`: Directory to save processed data

### DeepSeek Model Arguments
- `--deepseek-model`: DeepSeek model to use for tokenizer
- `--tokenizer-cache-dir`: Directory to cache downloaded tokenizer

### Dataset Arguments
- `--dataset-name`: Override default HuggingFace dataset name
- `--language`: Wikipedia language code (en, es, fr, de, etc.)
- `--date`: Wikipedia dump date (YYYYMMDD format)
- `--split`: Dataset split to download (default: train)
- `--max-samples`: Limit number of samples (for testing)

### Custom Dataset Arguments
- `--custom-dataset-name`: HuggingFace dataset name for custom datasets
- `--custom-text-field`: Field name containing text (default: text)

### Processing Arguments
- `--workers`: Number of parallel workers
- `--min-length`: Minimum text length in characters (default: 100)
- `--max-length`: Maximum text length in characters
- `--append-eod`: Add end-of-document tokens
- `--split-sentences`: Split documents into sentences
- `--keep-newlines`: Preserve newlines when splitting
- `--clean-text`: Clean markup and formatting (default: True)

### Advanced Options
- `--raw-data-dir`: Directory for intermediate JSONL files
- `--megatron-path`: Path to Megatron-LM repository

## Tokenizer Detection

The script automatically detects the tokenizer type used by each DeepSeek model:

- **SentencePiece**: Uses `GPTSentencePieceTokenizer` with `.model` file
- **BPE**: Uses `GPT2BPETokenizer` with `vocab.json` and `merges.txt`
- **HuggingFace**: Uses `HuggingFaceTokenizer` as fallback

## Output Files

The script generates several files in your output directory:

```
output-dir/
├── tokenizer/                           # DeepSeek tokenizer files
│   ├── tokenizer.model                  # (SentencePiece)
│   ├── vocab.json                       # (BPE)
│   ├── merges.txt                       # (BPE)
│   └── tokenizer_config.json            # Configuration
├── raw/
│   └── dataset_split.jsonl              # Raw JSONL data
├── dataset_text_document.bin            # Binary token data
├── dataset_text_document.idx            # Index file
└── dataset_deepseek_config.json         # Configuration metadata
```

## Using with Megatron Training

After preprocessing, use the data in your Megatron training script. The exact arguments depend on the detected tokenizer type:

### SentencePiece Tokenizer
```bash
python pretrain_gpt.py \
    --data-path /path/to/processed/data/dataset_text_document \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/processed/data/tokenizer/tokenizer.model \
    # ... other training arguments
```

### BPE Tokenizer
```bash
python pretrain_gpt.py \
    --data-path /path/to/processed/data/dataset_text_document \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /path/to/processed/data/tokenizer/vocab.json \
    --merge-file /path/to/processed/data/tokenizer/merges.txt \
    # ... other training arguments
```

### HuggingFace Tokenizer
```bash
python pretrain_gpt.py \
    --data-path /path/to/processed/data/dataset_text_document \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/processed/data/tokenizer \
    # ... other training arguments
```

## Performance Tips

1. **Model Selection**: Choose the appropriate DeepSeek model for your use case
   - V2-Lite: Smaller, faster preprocessing
   - V2: Balanced performance
   - V3: Latest model with best performance

2. **Workers**: Set `--workers` to match your CPU cores for faster processing

3. **Storage**: Use fast storage (NVMe SSD) for better I/O performance

4. **Memory**: Ensure sufficient RAM (32GB+ for large datasets)

5. **Network**: Stable connection for downloading models and datasets

## Troubleshooting

### Common Issues

1. **Tokenizer Download Error**:
   ```
   Error loading DeepSeek tokenizer
   ```
   - Check internet connection
   - Verify model name is correct
   - Try setting `--tokenizer-cache-dir` to a writable location

2. **Out of Memory**: Reduce `--workers` or use `--max-samples` for testing

3. **Dataset Configuration Error**:
   - For Wikipedia: Check available dates and languages
   - For custom datasets: Verify dataset name and text field

4. **Preprocessing Failure**: Check Megatron-LM installation and paths

### Validation

Check that preprocessing completed successfully:

```bash
# Verify files exist
ls -la /path/to/output/dataset_text_document.*

# Check tokenizer files
ls -la /path/to/output/tokenizer/

# Check configuration
cat /path/to/output/dataset_deepseek_config.json

# Sample the data
head -5 /path/to/output/raw/dataset_*.jsonl
```

## DeepSeek Model Information

### DeepSeek-V2-Lite
- **Size**: Smaller variant optimized for efficiency
- **Use Case**: Development, testing, smaller-scale training
- **Tokenizer**: Typically SentencePiece-based

### DeepSeek-V2
- **Size**: Full-scale model
- **Use Case**: Production training, research
- **Tokenizer**: Advanced tokenization optimized for code and text

### DeepSeek-V3
- **Size**: Latest and most advanced
- **Use Case**: State-of-the-art performance, latest features
- **Tokenizer**: Most recent tokenization improvements

## Advanced Usage

### Multiple Datasets

```bash
# Process multiple datasets with the same tokenizer
for dataset in slimpajama wikipedia; do
    python preload_deepseek_datasets.py \
        --dataset $dataset \
        --deepseek-model deepseek-ai/DeepSeek-V3 \
        --output-dir /data/${dataset}_deepseek \
        --workers 32 \
        --append-eod
done
```

### Custom Text Processing

```bash
# Custom dataset with specific text field and filtering
python preload_deepseek_datasets.py \
    --dataset custom \
    --custom-dataset-name your/code-dataset \
    --custom-text-field code \
    --deepseek-model deepseek-ai/DeepSeek-V3 \
    --output-dir /data/code_deepseek \
    --min-length 50 \
    --max-length 8192 \
    --workers 16
```

### Multi-language Wikipedia

```bash
# Process multiple Wikipedia languages
for lang in en es fr de zh; do
    python preload_deepseek_datasets.py \
        --dataset wikipedia \
        --language $lang \
        --deepseek-model deepseek-ai/DeepSeek-V3 \
        --output-dir /data/wikipedia_${lang}_deepseek \
        --workers 16 \
        --append-eod
done
```

## Integration with Benchmarking Scripts

To use DeepSeek-preprocessed data with the provided benchmarking scripts:

1. **Update `runtime_configs/benchmarking/common.conf`**:
   ```bash
   export DATA_PATH="/path/to/processed/data/dataset_text_document"
   export TOKENIZER_TYPE="GPTSentencePieceTokenizer"  # or detected type
   export TOKENIZER_MODEL="/path/to/processed/data/tokenizer/tokenizer.model"
   ```

2. **Run benchmarking**:
   ```bash
   ./local_benchmarking.sh
   ```

## Data Quality Considerations

**Advantages of DeepSeek Tokenizers:**
- Optimized for DeepSeek model architectures
- Better handling of code and technical text
- Consistent with DeepSeek training methodology
- Latest tokenization improvements

**Best Practices:**
- Use the same DeepSeek model tokenizer that you plan to train
- Consider data mixing ratios when combining datasets
- Validate tokenization quality on sample data
- Monitor preprocessing statistics for quality control

## Combining with Other Datasets

You can combine DeepSeek-tokenized datasets with other datasets using Megatron's data blending:

```bash
python pretrain_gpt.py \
    --data-path \
        0.7 /path/to/slimpajama_deepseek_text_document \
        0.3 /path/to/wikipedia_deepseek_text_document \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model /path/to/tokenizer/tokenizer.model \
    # ... other training arguments
```

This approach ensures consistent tokenization across all datasets while allowing flexible data mixing for optimal training results.
