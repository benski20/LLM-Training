"""Builds dashboard/index.html (the Tiny LLM Explorer) from dashboard/explorer_src.html.

Bakes in the Shakespeare text for the in-browser counting model, the confusion score for each
memory length, and an example training_log.json for the Replay tab. Run:
    python tools/build_explorer.py [optional/path/for/body-only.html]
"""
import collections
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_CHARS = 200_000            # text the counting model learns from
HELD = (1_000_000, 1_004_000)     # later plays, used for the guessing game and confusion scores
MAX_MEMORY = 10


def memory_confusion(train, held, vocab):
    """Confusion (perplexity) of the counting model for each look-back length; same maths as nextProbs() in the page."""
    cache = {}
    unigram = collections.Counter(train)

    def counts(ctx):
        if ctx not in cache:
            c = collections.Counter()
            i = train.find(ctx)
            while i != -1 and i + len(ctx) < len(train):
                c[train[i + len(ctx)]] += 1
                i = train.find(ctx, i + 1)
            cache[ctx] = c
        return cache[ctx]

    def prob(ch, ctx, k):
        p = (unigram[ch] + 1) / (len(train) + len(vocab))
        for j in range(1, k + 1):
            c = counts(ctx[-j:])
            total = sum(c.values())
            if not total:
                break
            p = (c[ch] + len(c) * p) / (total + len(c))
        return p

    scores = []
    for k in range(MAX_MEMORY + 1):
        nll = [-math.log(prob(held[i], held[i - MAX_MEMORY:i], k)) for i in range(MAX_MEMORY, 2000)]
        scores.append(round(math.exp(sum(nll) / len(nll)), 2))
    return scores


def build():
    text = open(os.path.join(ROOT, "data", "shakespeare.txt"), encoding="utf-8").read()
    vocab = "".join(sorted(set(text)))
    corpus, held = text[:CORPUS_CHARS], text[HELD[0]:HELD[1]]
    log_path = os.path.join(ROOT, "dashboard", "example_training_log.json")
    example = json.load(open(log_path)) if os.path.exists(log_path) else None

    src = open(os.path.join(ROOT, "dashboard", "explorer_src.html"), encoding="utf-8").read()
    for key, value in {
        "CORPUS": corpus, "HELD": held, "VOCAB": vocab,
        "MEM_CONFUSION": memory_confusion(corpus, held, vocab), "EXAMPLE_LOG": example,
    }.items():
        placeholder = f"/*__{key}__*/" + ("null" if key == "EXAMPLE_LOG" else "[]" if key == "MEM_CONFUSION" else '""')
        assert placeholder in src, placeholder
        src = src.replace(placeholder, json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))

    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    title_end = src.index("</style>") + len("</style>")
    page = head + src[:title_end] + "\n</head>\n<body>\n" + src[title_end:] + "\n</body>\n</html>\n"
    if len(sys.argv) > 1:  # also write the page body alone (for hosts that add their own <head>)
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(src)
    out = os.path.join(ROOT, "dashboard", "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out} ({len(page) / 1e6:.2f} MB, example log: {'yes' if example else 'no'})")


if __name__ == "__main__":
    build()
