"""
test.py - Test trained models
Usage:
  python test.py <word>
"""

import argparse
import numpy as np
from src.vocabulary import Vocabulary
from src.utils import nearest_neighbours, analogy_test

def parse_args():
    p = argparse.ArgumentParser(description="Test Word2Vec embeddings")
    p.add_argument("--corpus",
                   help="Path to corpus file (for building vocabulary) - must be the same corpus used for training",
                   default="data/tokens/ten.txt")
    p.add_argument("--emb_file",
                   help="Path to .npy file containing word embeddings saved during training",
                   default="outputs/ten_emb_100_win_15_neg_5.npy")
    p.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    p.add_argument("--task", choices=["nearest", "analogy"], default="nearest", help="Evaluation task to perform\n"
                                                                                     "nearest: find nearest neighbours for a word\n"
                                                                                     "analogy: perform analogy test (A is to B as C is to ?)")
    p.add_argument("word_a", help="Word to find nearest neighbours for / word A in analogy test")
    p.add_argument("word_b", help="Word B in analogy test (for task analogy)", nargs="?", default=None)
    p.add_argument("word_c", help="Word C in analogy test (for task analogy)", nargs="?", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    # print(f"\nConfig: {vars(args)}")

    emb_file = args.emb_file
    embeddings = np.load(emb_file)

    corpus_file = args.corpus
    vocab = Vocabulary()
    with open(corpus_file, "r") as f:
        text = f.read()
    tokens = text.split()
    vocab.build(tokens)

    # print(f"\nVocabulary: {len(vocab.idx2word)} words")
    # print(f"\n10 random words from vocabulary: {np.random.choice(vocab.idx2word, size=10, replace=False)}")

    if args.task == "analogy":
        word_a, word_b, word_c = args.word_a, args.word_b, args.word_c
        print(f"Analogy test: '{word_a}' is to '{word_b}' as '{word_c}' is to ?")
        try:
            results = analogy_test(word_a, word_b, word_c, vocab, embeddings, top_k=args.k)
            print("Top candidates:")
            for candidate, sim in results:
                print(f"  {candidate} (similarity: {sim:.4f})")
        except ValueError as e:
            print(e)
    if args.task == "nearest":
        word = args.word_a
        print(f"Nearest neighbours for '{word}':")
        try:
            neighbours = nearest_neighbours(word, vocab, embeddings, top_k=args.k)
            for neighbour, sim in neighbours:
                print(f"  {neighbour} (similarity: {sim:.4f})")
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()
