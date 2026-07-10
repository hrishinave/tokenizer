# Multilingual BPE Tokenizer — India's Wikipedia page

A from-scratch (pure-Python, no libraries) Byte-Pair-Encoding tokenizer trained
on India's Wikipedia article in **English, Hindi, Telugu, and Kannada**, sharing
a single **10,000-token vocabulary** across all four languages.

For each language we measure **fertility**:

```
X = total tokens produced / total whitespace words   (avg tokens per word)
```

## How to run

```bash
python3 fetch_data.py   # download 4 articles -> data/{en,hi,te,kn}.txt
python3 train.py        # train shared 10,000-token BPE -> tokenizer.json
python3 eval.py         # compute X1..X4, sort, sanity-check round-trips
```

## Method

- **Data**: MediaWiki `extracts` API (`explaintext=1`) → clean plaintext, no wiki
  markup or reference numbers. Titles: en=India, hi=भारत, te=భారతదేశం, kn=ಭಾರತ.
- **BPE** (`bpe.py`): classic Sennrich-2016 word-based algorithm. Each whitespace
  word = a tuple of Unicode characters + `</w>` boundary marker. Iteratively merge
  the most frequent adjacent symbol pair across the *combined* corpus until the
  vocabulary reaches 10,000. Character-level base symbols (315 of them across the
  four scripts) mean every script is handled uniformly, no byte inflation.
- **Shared vocab**: one 10,000-token budget for all four languages (9,685 merges
  + 315 base symbols). Merge priority is driven by combined pair frequency.

## Results

Corpus sizes (this is the key context — see caveat below):

| Lang | Words | Tokens | X = tokens/word |
|------|------:|-------:|----------------:|
| English (X1) | 10027 | 14499 | **1.446** |
| Hindi   (X2) |  8022 | 10684 | **1.332** |
| Telugu  (X3) |  2453 |  4585 | **1.869** |
| Kannada (X4) |   979 |  1055 | **1.078** |

**Sorted, largest → smallest fertility:**

```
X3 (Telugu 1.869)  >  X1 (English 1.446)  >  X2 (Hindi 1.332)  >  X4 (Kannada 1.078)
```

All four languages pass an exact word→tokens→word round-trip check.

## Interpretation & caveat

The ordering here is dominated by **corpus size**, not purely by linguistic
morphology. The four articles differ ~10× in length (English 10k words vs Kannada
979). In a *shared* vocabulary trained by combined frequency:

- A **small** corpus (Kannada, 979 words) has few unique word forms that repeat
  often, so the 10k-token budget effectively **memorizes whole words** as single
  tokens → very low fertility (1.078).
- A **larger, more varied** corpus (Telugu at 2.4k words but many distinct
  agglutinative forms; English with 10k words and a large type count) has more
  unique material competing for the same shared merge budget → higher fertility.

So fertility ≈ f(script complexity, morphology, **and corpus size / repetition**),
and with these unbalanced corpora the size/repetition term is the loudest. English
lands at 1.446 rather than the ~1.2 target precisely because a single 10k vocab is
split across four scripts.

To isolate the *morphological* signal (the classic "Dravidian agglutinative
languages fragment more"), re-run with **equal-sized corpora** per language and/or
a **held-out** evaluation split so no language can simply memorize its own text.
