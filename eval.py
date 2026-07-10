"""Evaluate per-language fertility X = total_tokens / total_words and sort."""
import os

import bpe

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")

# (code, human name, X-label as used in the task statement)
LANGS = [
    ("en", "English", "X1"),
    ("hi", "Hindi", "X2"),
    ("te", "Telugu", "X3"),
    ("kn", "Kannada", "X4"),
]


def load_text(lang):
    with open(os.path.join(DATA_DIR, f"{lang}.txt"), encoding="utf-8") as f:
        return f.read()


def strip_marker(token):
    return token.replace(bpe.END, "")


def roundtrip_ok(text, merges, n=200):
    """Check the first n words reconstruct exactly from their tokens."""
    for word in bpe.pretokenize(text)[:n]:
        toks = bpe.encode(word, merges)
        if "".join(strip_marker(t) for t in toks) != word:
            return False, word
    return True, None


def main():
    merges, vocab = bpe.load(TOK)
    print(f"Loaded tokenizer: vocab {len(vocab)}, merges {len(merges)}\n")

    results = []
    for code, name, label in LANGS:
        text = load_text(code)
        tokens = bpe.encode(text, merges)
        n_tokens = len(tokens)
        n_words = len(bpe.pretokenize(text))
        x = n_tokens / n_words
        ok, bad = roundtrip_ok(text, merges)
        results.append((label, code, name, n_words, n_tokens, x, ok, bad))

    # ---- table ----
    print(f"{'Label':5} {'Lang':8} {'Words':>7} {'Tokens':>8} "
          f"{'X=tok/word':>11} {'round-trip':>11}")
    print("-" * 55)
    for label, code, name, w, t, x, ok, bad in results:
        rt = "OK" if ok else f"FAIL:{bad!r}"
        print(f"{label:5} {name:8} {w:7d} {t:8d} {x:11.4f} {rt:>11}")

    # ---- sorted (largest -> smallest) ----
    ordered = sorted(results, key=lambda r: r[5], reverse=True)
    print("\nSorted by fertility (largest -> smallest):")
    chain = "  >  ".join(f"{r[0]} ({r[2]} {r[5]:.3f})" for r in ordered)
    print("  " + chain)

    # ---- sample word -> token splits ----
    print("\nSample word -> token splits:")
    for code, name, _ in LANGS:
        text = load_text(code)
        words = [w for w in bpe.pretokenize(text) if len(w) > 4][:3]
        print(f"  {name}:")
        for w in words:
            toks = bpe.encode(w, merges)
            pretty = " | ".join(strip_marker(t) for t in toks)
            print(f"    {w}  ->  {pretty}   ({len(toks)} tokens)")


if __name__ == "__main__":
    main()
