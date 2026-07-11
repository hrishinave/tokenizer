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
- **Language weighting**: the goal is English fertility **X1 < 1.2**. With equal
  weighting a single 10k vocab split across four scripts gives English only
  ~2,400 effective merges and X1 stalls at **1.446**. So English is upweighted
  (`WEIGHTS = [3, 1, 1, 1]` in `train.py`) — a standard multilingual-tokenizer
  technique (mBERT/XLM-R weight languages rather than treating corpora equally).
  English then claims a larger share of the *same* 10k budget and its fertility
  drops to its near-monolingual floor. Weight 3 is the **smallest** weight that
  clears the target (weight 2 → 1.283 misses; weight 3 → 1.124; weight 8+
  saturates at 1.000 but needlessly starves the Indic languages).

## Results (English weight = 3, shared vocab = 10,000)

| Lang | Words | Tokens | X = tokens/word |
|------|------:|-------:|----------------:|
| English (X1) | 10027 | 11266 | **1.124** ✅ < 1.2 |
| Hindi   (X2) |  8022 | 12361 | **1.541** |
| Telugu  (X3) |  2453 |  6230 | **2.540** |
| Kannada (X4) |   979 |  2629 | **2.685** |

**Sorted, largest → smallest fertility:**

```
X4 (Kannada 2.685)  >  X3 (Telugu 2.540)  >  X2 (Hindi 1.541)  >  X1 (English 1.124)
```

English is the least-fertile language and sits below the 1.2 target. All four
languages pass an exact word→tokens→word round-trip check.

## Interpretation

- **English (1.124)**: upweighted, so nearly every English word — including
  punctuated forms like `India,` and longer words like `officially` — is a single
  token. This is its near-monolingual floor.
- **Hindi (1.541)**: moderate fertility; Devanagari with sizeable corpus (8k
  words) gets a fair share of the remaining budget.
- **Telugu & Kannada (2.54, 2.69)**: highest. Dravidian agglutination produces
  long, mostly-unique surface forms, and (especially Kannada at 979 words) small
  corpora give BPE little repetition to amortize merges — so words fragment into
  several subwords.

### Design notes / things tried

- **Fertility floor is the shared budget.** One 10k vocab across four scripts is
  the structural reason equal weighting can't reach X1 = 1.2; weighting reallocates
  that fixed budget rather than enlarging it.
- **Punctuation is a fixed tax.** Force-splitting punctuation into its own tokens
  was tested and *raised* fertility for 3 of 4 languages (a standalone mark is a
  guaranteed extra token), so punctuation is kept attached to its word — BPE then
  amortizes it into a subword merge. See the note in `bpe.pretokenize`.
- **Trade-off.** Upweighting English lowers X1 but raises X2–X4. To instead
  minimize *overall* fertility, drop the weighting and/or give each language its
  own budget (or a larger shared vocab) and use a held-out eval split.
