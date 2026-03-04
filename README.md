# Word2Vec  (NumPy only)

This repository contains a simple implementation of the Word2Vec algorithm, 
using both the Skip-gram and Continuous Bag-of-Words (both with Negative Sampling) architectures for training.

### It is used for two JetBrains internship tasks:
1. Learning to Reason with Small Models
    - I have implemented the SGNS method first as the assignment for this task, and it is available in the `sgns` branch.
2. Hallucination Detection
   - For the second task, I have utilized the code implemented before and enhanced it to support both Skip-gram and CBOW architectures, which is available in the `main` branch.

## Project Structure

```
word2vec/
├── README.md
├── requirements.txt
├── train.py                    # Entry point — CLI to kick off training
├── test.py                     # Evaluate embeddings on word similarity tasks
├── data/
│   ├── tokens/                 # Tokenized text corpus (e.g. text8)
│   ├── training_pairs/         # Preprocessed triplets
│   └── download_data.py        # Script to fetch the text corpus
├── src/
│   ├── __init__.py
│   ├── vocabulary.py           # Vocab building, subsampling
│   ├── dataset.py              # Corpus -> (center, context, negatives) triplets
│   ├── model.py                # Embedding matrices + forward pass
│   ├── loss.py                 # Negative-sampling loss
│   ├── gradients.py            # Manual backward pass / gradient derivations
│   ├── trainer.py              # Training loop, SGD update, logging
│   └── utils.py                # Cosine similarity, nearest-neighbours, helpers
└── outputs/                    # Saved embeddings (.npy), loss curves, plots
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py      # downloads text8 and create smaller training sets for quick experiments

# Train the model:
python train.py -h                # see all training options
python train.py --variant sgns    # train Skip-gram with Negative Sampling
python train.py --variant cbow    # train Continuous Bag-of-Words with Negative Sampling

# Test the trained embeddings on word similarity tasks:
python test.py -h                               # see all evaluation options
python test.py --task nearest anarchism         # get nearest neighbours for "anarchism"
python test.py --task analogy man woman king    # evaluate analogy "woman" - "man" + "king" = ?
```
