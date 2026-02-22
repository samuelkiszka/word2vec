# Word2Vec - skip-gram  (NumPy only)

Skip-gram with Negative Sampling implemented in pure NumPy.

## Project Structure

```
word2vec/
├── README.md
├── requirements.txt
├── train.py                    # Entry point — CLI to kick off training
├── test.py                     # Evaluate embeddings on word similarity
├── data/
│   ├── tokens/                 # Tokenized text corpus (e.g. text8)
│   ├── training_pairs/         # Preprocessed (center, context, negatives) triplets
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

# Train the skip-gram model:
python train.py -h                # see all training options
python train.py

# Test the trained embeddings on word similarity tasks:
python test.py -h                 # see all evaluation options
python test.py anarchism          # get nearest neighbours for "anarchism"
```
