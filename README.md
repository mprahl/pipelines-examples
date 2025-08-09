# Granite Build PoC - LLM Training and Evaluation Pipelines

This repository contains Kubeflow Pipelines for training and evaluating Large
Language Models (LLMs), specifically designed for the IBM Granite model family
but compatible with other transformer-based models.

## Overview

The project provides two main pipelines:

1. **Training Pipeline** (`train_pipeline.py`) - Fine-tune LLMs using
   distributed training with LoRA adapters
2. **Evaluation Pipeline** (`pipeline.py`) - Evaluate model performance on
   classification and summarization tasks
