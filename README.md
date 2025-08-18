# LLM Training and Evaluation Pipelines

This repository contains Kubeflow Pipelines components and pipelines for
training and evaluating Large Language Models (LLMs).

## Overview

The project provides two example pipelines:

1. **Training Pipeline** (`train_and_eval_pipeline.py`) - Complete end-to-end
   workflow for fine-tuning LLMs using distributed training with LoRA adapters.
   Features:

   - Multi-node distributed training with Kubeflow Trainer v2
   - Automatic dataset preparation (Yoda sentences dataset)
   - Model-specific LoRA target module optimization
   - Flash Attention 2 support for performance
   - Comprehensive metrics collection and artifact management
   - Integrated evaluation on classification and summarization tasks

2. **Evaluation Pipeline** (`eval_pipeline.py`) - Standalone model evaluation on
   multiple NLP tasks:
   - **Classification**: RTE (Recognizing Textual Entailment) and WNLI (Winograd
     NLI)
   - **Summarization**: XSum (Extreme Summarization) with formal templates
   - VLLM-powered inference for efficient evaluation
   - Configurable task selection and performance metrics
   - Optional custom translation dataset for an additional translation task

## Reusable Components

See [components/README.md](./components/README.md) for detailed component documentation of how to build your own pipelines.
