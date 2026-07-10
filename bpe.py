"""From-scratch Byte-Pair Encoding (Sennrich et al., 2016 word-based variant).

Dependency-free. Character-level (Unicode codepoint) base symbols so every
script (Latin, Devanagari, Telugu, Kannada) is handled uniformly. Each
whitespace word is a tuple of characters terminated by an end-of-word marker.
"""
import json
from collections import Counter, defaultdict

END = "</w>"  # end-of-word boundary marker


def pretokenize(text):
    """Whitespace words. Same unit as the fertility denominator."""
    return text.split()


def _word_freqs(texts):
    freqs = Counter()
    for text in texts:
        freqs.update(pretokenize(text))
    return freqs


def _word_symbols(word):
    """Word -> tuple of base symbols (chars) plus the end marker."""
    return tuple(word) + (END,)


def _count_pairs(corpus):
    """Count adjacent symbol pairs across the corpus, weighted by word freq."""
    pairs = Counter()
    for symbols, freq in corpus.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_word(symbols, pair):
    """Replace every occurrence of `pair` in `symbols` with the merged symbol."""
    a, b = pair
    merged = a + b
    out = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train(texts, vocab_size=10000, verbose=True):
    """Train BPE on `texts`. Returns (merges, vocab).

    merges: ordered list of [a, b] pairs (merge priority = list order).
    vocab:  sorted list of all tokens (base symbols + merged tokens).
    """
    word_freqs = _word_freqs(texts)
    corpus = {_word_symbols(w): f for w, f in word_freqs.items()}

    vocab = set()
    for symbols in corpus:
        vocab.update(symbols)
    base_count = len(vocab)
    if verbose:
        print(f"  base symbols (unique chars + marker): {base_count}")
        print(f"  unique words: {len(corpus)}")

    merges = []
    while len(vocab) < vocab_size:
        pairs = _count_pairs(corpus)
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))  # freq, then deterministic
        merged_symbol = best[0] + best[1]
        corpus = {_merge_word(sym, best): f for sym, f in corpus.items()}
        merges.append([best[0], best[1]])
        vocab.add(merged_symbol)
        if verbose and len(merges) % 1000 == 0:
            print(f"  merges: {len(merges):5d}  vocab: {len(vocab):5d}  "
                  f"last pair freq: {pairs[best]}")

    if verbose:
        print(f"  done: {len(merges)} merges, vocab {len(vocab)} "
              f"(base {base_count})")
    return merges, sorted(vocab)


def encode(text, merges):
    """Encode `text` to a flat list of tokens using the learned merges."""
    ranks = {tuple(pair): i for i, pair in enumerate(merges)}
    tokens = []
    for word in pretokenize(text):
        symbols = list(_word_symbols(word))
        while len(symbols) >= 2:
            # find the highest-priority (lowest rank) adjacent pair present
            best_rank = None
            best_i = None
            for i in range(len(symbols) - 1):
                r = ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i is None:
                break
            symbols[best_i:best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        tokens.extend(symbols)
    return tokens


def save(path, merges, vocab):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"merges": merges, "vocab": vocab}, f, ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["merges"], d["vocab"]
