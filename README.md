# mT5-LatinSummarizer

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/AxelDlv00/LatinSummarizer)  [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue?logo=huggingface)](https://huggingface.co/LatinNLP/LatinSummarizerModel)  [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange?logo=huggingface)](https://huggingface.co/datasets/LatinNLP/LatinSummarizerDataset)

## Introduction

This repository contains **mT5-LatinSummarizer**, a specialized model fine-tuned from `mT5-small` for Latin summarization tasks. The model is first pre-trained to generate extractive summaries in Latin and traductions English-Latin, and then finetuned on High-Quality summaries to generate abstractive summaries of Latin texts, leveraging pre-existing datasets and techniques such as **LoRA (Low-Rank Adaptation)** for efficient fine-tuning.


## Repository Structure

```
├── Dataset_preparation.ipynb   # Notebook for dataset preprocessing
├── inference.ipynb             # Notebook for running inference on trained models
├── load_perseus.ipynb          # Notebook for loading and preprocessing the Perseus Latin dataset
├── train.ipynb                 # Notebook for training the model interactively
├── train_no_stanza.py          # Training script for Latin summarization without stanza-based tagging
├── train_with_stanza.py        # Training script using stanza-based linguistic features
├── README.md                   # Documentation of the project
├── requirements.txt            # Dependencies required for execution
├── LICENSE                     # License file
│
├── utils/                      # Utility functions for training, evaluation, and data processing
│   ├── bleu.py                 # Computes BLEU and CHRF scores for translation evaluation
│   ├── clean_dataframe.py      # Cleans and preprocesses dataset files
│   ├── encoding.py             # Handles tokenization and encoding of texts
│   ├── extractive_summary.py   # Implements extractive summarization techniques
│   ├── generate_translation.py # Functions for generating Latin translations
│   ├── grade_extractive_summary.py # Grades extractive summaries using heuristic rules
│   ├── loss_mT5.py             # Defines custom loss functions for mT5 fine-tuning
│   ├── mT5_train.py            # Main script for training mT5 with different configurations
│   ├── prompt_generator.py     # Generates structured prompts for training and inference
│   ├── rouge.py                # Computes ROUGE scores for evaluating summary quality
│   ├── split_chunks.py         # Splits large texts into smaller chunks for processing
│   ├── summary_mistral.py      # Uses the Mistral model to evaluate generated summaries
```

## Explanation of Key Files

### **Training & Evaluation**
- `train_no_stanza.py`: Trains mT5 for Latin summarization **without stanza-based annotations**.
- `train_with_stanza.py`: Trains mT5 with **linguistic features extracted using Stanza**.
- `train.ipynb`: Jupyter notebook for interactively training and debugging the model.
- `inference.ipynb`: Notebook to test model predictions after training.

### **Dataset Preparation**
- `Dataset_preparation.ipynb`: Prepares datasets for training, including formatting and tokenization.
- `load_perseus.ipynb`: Loads and processes the **Perseus Latin Corpus** for training data.

### **Utility Functions (`utils/` directory)**
- **Evaluation Scripts:**
  - `bleu.py`: Computes BLEU and CHRF scores for translation-based evaluation.
  - `rouge.py`: Calculates ROUGE scores for summarization quality assessment.
  - `grade_extractive_summary.py`: Implements heuristics to score extractive summaries.
- **Training Utilities:**
  - `mT5_train.py`: Core script for training mT5 models.
  - `loss_mT5.py`: Defines custom loss functions for training.
  - `prompt_generator.py`: Generates prompts for training and inference.
- **Data Processing:**
  - `clean_dataframe.py`: Cleans and processes dataset files.
  - `encoding.py`: Handles tokenization and encoding of text inputs.
  - `split_chunks.py`: Splits long texts into manageable chunks.
- **Summarization & Translation:**
  - `generate_translation.py`: Handles text generation for translation.
  - `extractive_summary.py`: Implements extractive summarization techniques.
  - `summary_mistral.py`: Uses the Mistral model for summary evaluation.

## Installation

To set up the project environment, install dependencies:
```sh
conda create --name LatinSummarizer python=3.11 -y
conda activate LatinSummarizer
conda install -y jupyter jupyterlab notebook nbconvert
pip install -r requirements.txt
```

## Citing
If you use this model, please cite:
```
@article{DelavalLubek2025,
  author    = {Axel Delaval, Elsa Lubek},
  title     = {Introduction of mT5-LatinSummarizer, a version of mT5-small fine-tuned on Latin summarization},
  journal   = {École Polytechnique},
  year      = {2025}
}
```