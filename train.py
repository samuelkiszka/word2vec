"""
train.py — CLI entry point
Usage:
  python train.py --epochs 5 --emb-dim 100 --neg-samples 5 --window 5
"""

import argparse
import time
import matplotlib.pyplot as plt

from src.vocabulary import Vocabulary
from src.dataset import build_training_pairs
from src.model import Word2Vec
from src.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train Word2Vec (SGNS) in NumPy")
    p.add_argument("--corpus",      default="data/tokens/ten.txt")
    p.add_argument("--emb-dim",     type=int,   default=100)
    p.add_argument("--window",      type=int,   default=15)
    p.add_argument("--neg-samples", type=int,   default=5)
    p.add_argument("--min-count",   type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=0.025)
    p.add_argument("--subsample-t", type=float, default=1e-5)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output",      default="outputs/embeddings.npy")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Config: {vars(args)}")

    # --- Step 1: Load & tokenise corpus ---
    print(f"\nLoading corpus from {args.corpus}...")
    time1 = time.time()

    with open(args.corpus, "r") as f:
        text = f.read()
    tokens = text.split()  # very basic tokenisation; you can improve this!
    print(f"Loaded {len(tokens)} tokens in {time.time() - time1:.2f} seconds")

    # --- Step 2: Build vocabulary ---
    print("\nBuilding vocabulary...")
    time1 = time.time()
    vocab = Vocabulary(min_count=args.min_count, subsample_t=args.subsample_t)
    vocab.build(tokens)
    token_ids = [vocab.word2idx[t] for t in vocab.subsample(tokens) if t in vocab.word2idx]
    print(f"Vocab size: {len(vocab.idx2word)}")
    print(f"Built vocabulary and subsampled tokens in {time.time() - time1:.2f} seconds")

    # --- Step 3: Build training pairs ---
    print("\nBuilding training pairs...")
    time1 = time.time()
    pairs = build_training_pairs(token_ids, vocab, args.window, args.neg_samples)
    print(f"Built {len(pairs)} training pairs in {time.time() - time1:.2f} seconds")

    # --- Step 4: Initialise model ---
    model = Word2Vec(vocab_size=len(vocab.idx2word), emb_dim=args.emb_dim, seed=args.seed)

    # --- Step 5: Train - save the best embeddings in the proces ---
    print("\nTraining model...")
    time1 = time.time()
    input_file_name = args.corpus.split("/")[-1].split(".")[0]
    out_file = f"outputs/{input_file_name}_emb_{args.emb_dim}_win_{args.window}_neg_{args.neg_samples}"

    trainer = Trainer(model, lr_start=args.lr)
    losses = trainer.train(pairs, epochs=args.epochs, save_file=out_file + ".npy")
    print(f"Training completed in {time.time() - time1:.2f} seconds. Best embeddings saved to {out_file}.npy")

    # --- Step 6: Save performance plot ---
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss over Epochs")
    plt.savefig(out_file + "_loss.png")




if __name__ == "__main__":
    main()
