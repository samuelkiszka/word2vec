"""
test.py - Test trained models
Usage:
  python test.py <word>
"""

import argparse
import numpy as np
from src.vocabulary import Vocabulary
from src.utils import nearest_neighbours

def parse_args():
    p = argparse.ArgumentParser(description="Test Word2Vec embeddings")
    p.add_argument("--corpus",
                   help="Path to corpus file (for building vocabulary) - must be the same corpus used for training",
                   default="data/tokens/ten.txt")
    p.add_argument("--emb_file",
                   help="Path to .npy file containing word embeddings saved during training",
                   default="outputs/ten_emb_100_win_15_neg_5.npy")
    p.add_argument("word", help="Word to find nearest neighbours for")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Config: {vars(args)}")

    emb_file = args.emb_file
    embeddings = np.load(emb_file)

    corpus_file = args.corpus
    vocab = Vocabulary()
    with open(corpus_file, "r") as f:
        text = f.read()
    tokens = text.split()
    vocab.build(tokens)

    print(f"Vocabulary: {len(vocab.idx2word)} words")
    print(f"10 random words from vocabulary: {np.random.choice(vocab.idx2word, size=10, replace=False)}")

    word = args.word
    print(f"Nearest neighbours for '{word}':")
    try:
        neighbours = nearest_neighbours(word, vocab, embeddings, top_k=5)
        for neighbour, sim in neighbours:
            print(f"  {neighbour} (similarity: {sim:.4f})")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
