# Train Your Own Mini-LLM 🧠

A one-period (50-minute) high-school lesson where every student trains their own GPT-style language model on free Kaggle GPUs. They watch it go from random gibberish to Shakespeare in about 7 minutes, with a live dashboard that shows what's happening inside.

- **[TEACHER_GUIDE.md](TEACHER_GUIDE.md)**: setup checklist, minute-by-minute lesson plan, tuning, troubleshooting
- **[STUDENT_HANDOUT.md](STUDENT_HANDOUT.md)**: account setup homework and in-class steps
- **[notebooks/train_your_own_llm.ipynb](notebooks/train_your_own_llm.ipynb)**: the student notebook (Kaggle or Colab) with a built-in live training dashboard
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benski20/LLM-Training/blob/main/notebooks/train_your_own_llm.ipynb)
- **[dashboard/index.html](dashboard/index.html)**: *Tiny LLM Explorer*, the projector companion: tokens, a next-letter guessing game, memory, temperature, attention, replay of a student's training run, and model scale

## How it fits together

```
Kaggle notebook (each student)                       Tiny LLM Explorer (projector / browser)
┌───────────────────────────────────┐                ┌──────────────────────────────────────┐
│ 1. Load text → tokens             │                │ 1 Letters → numbers                  │
│ 2. Build a ~5M-param Transformer  │                │ 2 Guess the next letter (game)       │
│ 3. Train for 7 minutes            │  training_     │ 3 Memory: why look-back matters      │
│    └─ live dashboard: confusion,  │  log.json ───► │ 4 Rolling the dice: temperature      │
│       surprise curve, next-letter │                │ 5 Attention                          │
│       guesses, attention, samples │                │ 6 Replay a training run              │
│ 4. Play: prompts, temperature     │                │ 7 Tiny vs. giant                     │
└───────────────────────────────────┘                └──────────────────────────────────────┘
```

The model and training loop are adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), simplified and commented for beginners. Training is **time-limited** (`TRAIN_MINUTES`), so it always fits the class period whatever GPU a student gets.

## Rebuilding

The notebook and Explorer are generated from source, so edit those and rebuild:

```bash
python tools/build_notebook.py   # tools/build_notebook.py  → notebooks/train_your_own_llm.ipynb
python tools/build_explorer.py   # dashboard/explorer_src.html + data → dashboard/index.html
```

`dashboard/example_training_log.json` and `checkpoints/shakespeare_pretrained.pt` come from a full run of the notebook (Shakespeare, medium model).

## Data
- `shakespeare.txt`: Tiny Shakespeare (from Karpathy's char-rnn)
- `names.txt`: 32K common US baby names (from Karpathy's makemore)
- `sherlock.txt`: *The Adventures of Sherlock Holmes*, Arthur Conan Doyle (Project Gutenberg, public domain)
- `python_code.txt`: modules from the Python standard library (PSF license)
