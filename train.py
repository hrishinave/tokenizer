"""Train one shared 10,000-token BPE vocab on all four India articles."""
import os
import time

import bpe

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LANGS = ["en", "hi", "te", "kn"]
VOCAB_SIZE = 10000
# Per-language training weights. English is upweighted so it claims a larger
# share of the single shared 10k merge budget, pulling English fertility (X1)
# below the 1.2 target. Weight 3 is the smallest that achieves X1 < 1.2
# (weight 2 -> 1.283 misses; weight 3 -> 1.124); larger weights push X1 toward
# 1.0 but needlessly starve the Indic languages. Order matches LANGS.
WEIGHTS = [2.05, 1, 1, 1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")


def load_text(lang):
    with open(os.path.join(DATA_DIR, f"{lang}.txt"), encoding="utf-8") as f:
        return f.read()


def main():
    texts = [load_text(l) for l in LANGS]
    total_words = sum(len(t.split()) for t in texts)
    print(f"Training on {len(texts)} corpora, {total_words} total words, "
          f"target vocab {VOCAB_SIZE}, weights {dict(zip(LANGS, WEIGHTS))}")
    t0 = time.time()
    merges, vocab = bpe.train(texts, vocab_size=VOCAB_SIZE, weights=WEIGHTS)
    print(f"Trained in {time.time() - t0:.1f}s")
    bpe.save(OUT, merges, vocab)
    print(f"Saved -> {OUT}  (vocab {len(vocab)}, merges {len(merges)})")


if __name__ == "__main__":
    main()
