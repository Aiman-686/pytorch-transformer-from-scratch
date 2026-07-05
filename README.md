# Transformer From Scratch (PyTorch)

An implementation of the Transformer architecture from **"Attention Is All You Need"** (Vaswani et al., 2017), built and trained for an English → Italian translation task.

This project was built to deeply understand the architecture behind modern LLMs — not just use a pre-built `nn.Transformer` module, but implement every core component by hand and trace tensor shapes through the full forward pass.

## What's implemented

- Multi-head self-attention (from scratch, no `nn.MultiheadAttention`)
- Positional encoding
- Encoder and decoder stacks
- Masked (causal) attention for autoregressive decoding
- Custom tokenizer training pipeline (WordLevel tokenizer via HuggingFace `tokenizers`)
- Full training loop with checkpointing, TensorBoard logging, and validation

## Dataset

Trained on the [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) English–Italian parallel corpus.

## Tech stack

- PyTorch
- HuggingFace `datasets` and `tokenizers`
- TensorBoard for training visualization
- TorchMetrics for evaluation (CER, WER, BLEU)

## Running it

```bash
pip install -r requirements.txt
python train.py
```

Adjust `config.py` to change hyperparameters (`d_model`, `batch_size`, `num_epochs`, `seq_len`, etc.)

## Acknowledgment

This implementation is based on the excellent walkthrough by [Umar Jamil (hkproj)](https://github.com/hkproj/pytorch-transformer) and the original paper:

Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

## Why I built this

Implementing a paper from scratch — reading the math and turning it into working, trainable code — is one of the most effective ways to actually understand an architecture instead of just using it as a black box. This project focuses on that understanding: tracing exactly how attention weights, positional encodings, and masking come together to make sequence-to-sequence translation work.
