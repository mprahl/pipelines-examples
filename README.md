# Granite Build PoC - LLM Training and Evaluation Pipelines

This repository contains Kubeflow Pipelines for training and evaluating Large
Language Models (LLMs), specifically designed for the IBM Granite model family
but compatible with other transformer-based models.

## Overview

The project provides two main pipelines:

1. **Training Pipeline** (`train_and_eval_pipeline.py`) - Complete end-to-end workflow for fine-tuning LLMs using distributed training with LoRA adapters. Features:
   - Multi-node distributed training with Kubeflow Trainer
   - Automatic dataset preparation (Yoda sentences dataset)
   - Model-specific LoRA target module optimization
   - Flash Attention 2 support for performance
   - Comprehensive metrics collection and artifact management
   - Integrated evaluation on classification and summarization tasks

2. **Evaluation Pipeline** (`eval_pipeline.py`) - Standalone model evaluation on multiple NLP tasks:
   - **Classification**: RTE (Recognizing Textual Entailment) and WNLI (Winograd NLI)
   - **Summarization**: XSum (Extreme Summarization) with formal templates
   - VLLM-powered inference for efficient evaluation
   - Configurable task selection and performance metrics
