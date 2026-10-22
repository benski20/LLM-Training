# Teacher Guide: Train Your Own Mini-LLM in One Class Period

Students train a small GPT-style language model on Shakespeare using free Kaggle GPUs. A live dashboard shows what's happening inside the model while it trains. The model code is adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). It is a character-level Transformer with about 5 million parameters.

**What's in this repo**

| File | What it is |
|---|---|
| `notebooks/train_your_own_llm.ipynb` | The student notebook. Works on Kaggle and Google Colab. Has the live dashboard built in. |
| `dashboard/index.html` | **Tiny LLM Explorer**: a web page for the projector with 7 hands-on concept demos and a "replay your training run" view. Runs fully offline in a browser. |
| `data/` | Datasets: `shakespeare.txt`, `sherlock.txt`, `names.txt`, `python_code.txt` |
| `checkpoints/shakespeare_pretrained.pt` | A model trained ahead of time, as a backup if training fails |
| `STUDENT_HANDOUT.md` | One page for students: account setup (homework) and steps for the day |
| `tools/` | Scripts that rebuild the notebook and the Explorer (you don't need these) |

---

## Before class (allow about an hour, a few days ahead)

### 1. Student Kaggle accounts: assign this as homework
This is the step most likely to break the lesson, so do it ahead of time.
- Every student (or one per group of 2–3) needs a Kaggle account that is **phone-verified**. Kaggle won't give an account a GPU until it's verified (*Settings → Phone verification*).
- Kaggle's terms require users to be at least 13. Check your school's policy on student accounts.
- GPU quota is **per account, about 30 hours per week**. One class run uses about 10–15 minutes, so quota isn't a problem. The point is that each student needs their own account; one teacher account can't serve the whole class.
- Fallback for students without a verified account: **Google Colab**. It has a free T4 GPU, needs no phone verification, and needs only a Google account. The same notebook works there.

### 2. Upload the data as a Kaggle Dataset
1. On Kaggle: **Create → New Dataset**. Upload the 4 `.txt` files from `data/` and `checkpoints/shakespeare_pretrained.pt`. Name it something like `llm-class-data`. Set it to **Public**, or share it with students.
2. This means students don't need the internet toggle turned on (it also requires phone verification) or a working download.

### 3. Set up the class notebook
1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and upload `notebooks/train_your_own_llm.ipynb`.
2. In the right panel: **+ Add Input** → choose your `llm-class-data` dataset.
3. **Session options → Accelerator → GPU T4 x2**. Avoid P100: newer PyTorch versions don't support it well. The notebook detects that and falls back to CPU, which is far too slow.
4. **Do a full dry run.** Click **Run All** and time it. Check that:
   - the setup cell prints `✅ GPU is on: Tesla T4`,
   - the dashboard updates during training,
   - after 7 minutes the "What it writes" samples look like Shakespeare (names in capitals, line breaks, real words).
   If it's too slow or too rough, adjust `TRAIN_MINUTES` / `MODEL_SIZE` in the Settings cell (see Tuning below).
5. **Save Version**, then **Share → Public** (or share the link with students). On the day, students open the link and click **Copy & Edit**. Their copy gets the dataset attached automatically.

### 4. Set up the Explorer
`dashboard/index.html` is a single file with no server and no internet needed (apart from fonts). You can:
- open it directly in a browser on the projector computer, or
- use the shared link if you got one from Claude, or
- host it with GitHub Pages (repo **Settings → Pages → Deploy from branch**). The page is then at `https://<user>.github.io/<repo>/dashboard/`.

Send the link to students too. Each tab has its own link, e.g. `…/dashboard/#guess`.

---

## The 50-minute lesson

| Time | What happens | Where |
|---|---|---|
| **0–5** | **Hook.** Show ChatGPT writing something. Then ask: "What is it actually doing?" Reveal: *it's only ever guessing the next piece of text.* Play the **Guess the next letter** game together. Students shout guesses; you type them. | Explorer → *2 Guess the next letter* |
| **5–10** | **Launch.** Everyone opens the notebook → **Copy & Edit** → turns on the GPU → **Run All**. Walk around and help. Most problems are a missing GPU or a missing dataset (see Troubleshooting). Training starts by about minute 10. | Kaggle |
| **10–17** | **Watch it learn (training is running).** Put one student's dashboard on the projector. Point out: the **confusion score** dropping from ~65 toward ~5, the **writing samples** going from gibberish to word-shapes to words to Shakespeare-ish, and the next-letter chart for "…yonder window brea_" (the real answer is "k", but "t" for "breath" is a perfectly sensible guess, and the trained model ends up torn between the two). Meanwhile, students scroll up to the **Letters → numbers** cells and try encoding their own name. | Kaggle dashboard |
| **17–27** | **Concepts while it finishes.** Use the Explorer tabs: *1 Letters → numbers* (tokens), *3 Memory* (why looking back matters, and why counting breaks down after ~4 letters, which is why we need Transformers), *5 Attention* (the "it = animal / street" example, then the real model's attention). | Explorer |
| **27–40** | **Play.** Students run the *Play with your model* cells: their own prompts, the **temperature experiment**, *Look inside the model's head*, *Writing in slow motion*. Show the *4 Rolling the dice* tab to explain temperature. Fast finishers start a **Challenge**: new dataset, bigger model, their own text. | Kaggle + Explorer |
| **40–45** | **Show and tell.** 2–3 students read out their funniest or best generated text. Optional: load a student's `training_log.json` into *6 Replay* and scrub through their run on the projector. | Explorer → *6 Replay* |
| **45–50** | **Debrief: tiny vs. giant.** Their model: ~5M parameters, ~1 MB of text, 7 minutes. Llama 3.1: 405B parameters, ~15 trillion tokens, ~3,500 GPU-years. Ask: *Why can't your model answer a question?* (It was only trained to continue text. Chatbots need instruction tuning and feedback training on top.) *What happens if the training text is biased?* | Explorer → *7 Tiny vs. giant* |

**Key ideas to land (no maths needed)**
1. An LLM is a **next-token guessing machine**.
2. **Tokens**: text becomes numbers.
3. **Training** = guess, measure the **surprise** (loss), nudge millions of numbers (parameters) to be less surprised, repeat.
4. **Attention** lets each position decide which earlier text matters.
5. Writing = guessing + **rolling dice**. **Temperature** controls how risky the rolls are.
6. ChatGPT is the same idea, **scaled up** enormously, plus extra training to be a helpful assistant.

### Ways to explain the dashboard
- **Confusion ("like picking from N characters")**: "At the start it's like closing your eyes and pointing at a keyboard with 65 keys. By the end it's narrowed most guesses down to about 4 or 5 likely keys." (This is perplexity, exp(loss). You don't need to say that.)
- **Surprise score (loss)**: "Every time the right answer is something it thought was unlikely, it's surprised. Training is just making it less surprised."
- **Purple line vs. orange dots**: practice text vs. secret test text. Both going down means it's learning patterns rather than memorising. If you train a long time on a small dataset, the dots stop improving while the line keeps going down. That's **overfitting**.
- **Attention highlights**: "To guess the next letter in 'windo_', the model mostly looks at the last few letters of the word. Sometimes it reaches further back, e.g. to the character name at the top."

---

## Tuning (do this during your dry run)

These are set in the notebook's **Settings** cell.

| Setting | Default | Notes |
|---|---|---|
| `TRAIN_MINUTES` | 7 | Training is time-limited, not step-limited, so it always finishes on schedule whatever GPU students get. 10 gives noticeably better text if your period allows. |
| `MODEL_SIZE` | `"medium"` (~5M params) | `"small"` (~0.8M) is fast but rough. `"large"` (~11M, nanoGPT's "baby Shakespeare" size) is better but needs more minutes to pay off. |
| `DATASET` | `"shakespeare"` | Also `"sherlock"`, `"names"`, `"python_code"`, or a path to any `.txt` file. |

Rough expectations: on a T4, the medium model should reach a surprise score around 1.5–1.7 on the test text (confusion about 5). That's recognisable Shakespeare-style text with mostly real words. These numbers weren't measured on Kaggle, so trust your dry run over this paragraph.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `⚠️ No working GPU found` | Accelerator is off, or set to P100. Choose **GPU T4 x2** and **Run All** again. If the GPU option is greyed out, the account isn't phone-verified: pair the student with a neighbour or use Colab. |
| `Couldn't find shakespeare.txt nearby, downloading…` then an error | The dataset isn't attached and the internet is off. **+ Add Input** → your class dataset. (Or turn on Internet in Session options, which needs a verified phone.) |
| Dashboard doesn't update / training seems frozen | Kaggle sometimes delays output. Wait 10–20 seconds. If nothing changes, **Run All** again. |
| Training ran out of time / student arrived late | Scroll to the bottom → **Backup plan** cell → remove the `#` in front of `load_pretrained()` → run it. All the "Play" cells now work with the pre-trained model. |
| `CUDA out of memory` | Only likely with `"large"` plus a bigger batch. Restart the session and use `"medium"`. |
| Session stopped | Kaggle shuts down idle sessions after a while. Click **Run All** again, or use the backup plan. |

## Extension ideas (follow-up lessons)
- **Bring your own data**: song lyrics, a favourite book from Project Gutenberg, a group chat export, recipes. About 200 KB+ of text works best. Talk about copyright and privacy when choosing.
- **Overfitting experiment**: train on `names` for 15 minutes and watch the test dots flatten out while the practice line keeps falling.
- **Read the code**: the model is about 80 lines. Walk through `SelfAttention` with stronger students.
- Watch Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY), which builds this exact kind of model.
- Karpathy's [nanochat](https://github.com/karpathy/nanochat) shows the full pipeline to a real chatbot (pre-training → instruction tuning → RL). It needs serious GPUs, but it's good to show where this leads.
