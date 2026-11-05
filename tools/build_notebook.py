"""Builds notebooks/train_your_own_llm.ipynb from the cell sources below.

Edit the cells here, then run:  python tools/build_notebook.py
(Editing the .ipynb by hand works too, but then this file goes stale.)
"""
import json
import os

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------
md(r'''
# 🧠 Train Your Own Mini-LLM

Today you will build a **language model**: the same kind of AI that ChatGPT, Claude and Gemini are built on, just much, much smaller.

**The big idea:** a language model is a machine that has read a lot of text and plays one game over and over:

> *"Given the text so far, what comes next?"*

Ours reads one **letter** at a time. It starts out knowing nothing, so it types random gibberish. In a few minutes of training it will learn to write text that looks like Shakespeare.

### ▶️ Do this right now
1. **Turn on the GPU.** Kaggle: right-hand panel → **Session options** → **Accelerator** → **GPU T4 x2**, and check **Internet** is on. Colab: **Runtime → Change runtime type → T4 GPU**.
2. In the **Settings** cell below, type your teacher's **class code** and a **nickname**, so your training shows up live on the class board.
3. Click **Run All**. On Kaggle it's at the top; on Colab it's **Runtime → Run all**.
4. Scroll down to **🚀 Training** and watch the live dashboard while your teacher explains what's happening.

Training takes 6 minutes. While you wait, read through the notebook from the top.
''')

md(r'''
## ⚙️ 1. Settings: the knobs you can turn

You don't need to change anything the first time. Later, come back here, change something, and click **Run All** again.
''')

code(r'''
# ⚙️ SETTINGS
DATASET = "shakespeare"   # what the model reads. Try: "shakespeare", "sherlock", "names", "python_code"
                          # ...or the path to your own .txt file, like "/kaggle/input/my-data/lyrics.txt"
TRAIN_MINUTES = 6         # how long to train. Longer = better writing (up to a point)
MODEL_SIZE = "medium"     # "small" (fast but dumb), "medium", or "large" (smarter, needs more time)
WATCH_PROMPT = None       # text the dashboard uses for "what letter comes next?" (None = pick one for you)

CLASS_CODE = ""           # from your teacher, like "room-4821". Leave "" to keep your training private
NICKNAME = "anonymous"    # how you appear on the class board (please don't use your full name)
''')

md(r'''
## 🔧 2. Setup

This cell loads the tools we need and checks that the GPU is on. A **GPU** is a chip that can do thousands of small calculations at once, which is exactly what training needs.
''')

code(r'''
import glob, json, math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import HTML, display

torch.manual_seed(1337)
GITHUB_RAW = "https://raw.githubusercontent.com/benski20/LLM-Training/main"

def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:  # some older GPUs (like Kaggle's P100) don't work with new PyTorch versions
        (torch.ones(2, device="cuda") * 2).sum().item()
        return "cuda"
    except Exception:
        return "cpu"

device = pick_device()
if device == "cuda":
    print(f"✅ GPU is on: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No working GPU found, so training will be VERY slow.\n"
          "   Kaggle: right panel → Session options → Accelerator → 'GPU T4 x2', then Run All again.\n"
          "   Colab: Runtime → Change runtime type → T4 GPU.")

# 📡 Class board: if you typed a class code, send a short progress update every minute or so.
# It goes through ntfy.sh, a free public message relay. Only your nickname, scores and writing samples are sent.
import re, threading, uuid
LIVE_RELAY = "https://ntfy.sh"
live_topic = "tinyllm-" + re.sub(r"[^A-Za-z0-9_-]", "", CLASS_CODE.strip().lower())[:40] if CLASS_CODE.strip() else None
run_id = uuid.uuid4().hex[:8]
_live_warned = False

def send_live(update):
    def post():
        global _live_warned
        body = json.dumps(dict(v=1, run=run_id, nick=NICKNAME[:24], **update)).encode()
        for wait in (0, 10, 25, 45):   # the relay limits how fast each address can post, so retry politely
            time.sleep(wait)
            try:
                urllib.request.urlopen(urllib.request.Request(f"{LIVE_RELAY}/{live_topic}", data=body, method="POST"), timeout=8)
                return
            except Exception as e:
                error = e
        if not _live_warned:
            _live_warned = True
            print(f"(Couldn't reach the class board: {error}. Training continues. Is Internet on?)")
    if live_topic:
        threading.Thread(target=post, daemon=True).start()

if live_topic:
    print(f"📡 Class board: sending live updates as '{NICKNAME}' to class code '{CLASS_CODE}'")
''')

md(r'''
## 📚 3. Load the text the model will learn from

This text is the model's **entire world**. It will never know anything that isn't in here.
''')

code(r'''
DATA_URLS = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "names": "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
}

def find_text_file(name):
    """Look for the dataset in all the usual places (Kaggle inputs, this repo, Colab uploads)."""
    if os.path.exists(name):
        return name
    filename = os.path.basename(name if name.endswith(".txt") else name + ".txt")
    patterns = [f"/kaggle/input/**/{filename}", f"data/{filename}", f"../data/{filename}", f"/content/**/{filename}"]
    for pattern in patterns:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    url = DATA_URLS.get(name, f"{GITHUB_RAW}/data/{filename}")
    print(f"Couldn't find {filename} nearby, downloading it from {url}")
    urllib.request.urlretrieve(url, filename)  # needs Internet turned on (Kaggle: Session options → Internet)
    return filename

data_path = find_text_file(DATASET)
with open(data_path, encoding="utf-8", errors="ignore") as f:
    text = f.read()

print(f"📖 Loaded {data_path}")
print(f"   {len(text):,} characters, about {len(text.split()):,} words "
      f"(roughly {len(text.split()) // 90_000 or 1} novel(s) worth of text)\n")
print("Here's how it starts:\n")
print(text[:500])
''')

md(r'''
## 🔢 4. Turn letters into numbers ("tokens")

Computers only understand numbers, so we give every character its own ID number. Each of these pieces is called a **token**.

Our model uses one token per character. Big models like ChatGPT use chunks of words instead (`"un" + "believ" + "able"`), but the idea is the same.
''')

code(r'''
chars = sorted(set(text))             # every unique character in the text
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # "string to integer":  'a' -> 39
itos = {i: ch for i, ch in enumerate(chars)}   # "integer to string":  39 -> 'a'

def encode(s):
    """text -> list of numbers (characters the model has never seen are skipped)"""
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    """list of numbers -> text"""
    return "".join(itos[i] for i in ids)

print(f"The model's whole 'alphabet' has {vocab_size} characters:")
print(repr("".join(chars)))
print()
print("encode('Hello there!') =", encode("Hello there!"))
print("decode back       =", repr(decode(encode("Hello there!"))))
''')

md(r'''
**✏️ Try it:** change the text below to your name and run the cell (Shift+Enter).
''')

code(r'''
my_text = "Your Name Here"
print(encode(my_text))
''')

md(r'''
## 🎯 5. The guessing game

We cut the text into short windows. For every position in a window, the model must guess the **next** character. One window gives us *lots* of practice questions:
''')

code(r'''
data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]   # 90% to learn from, 10% kept secret to test it

window = text[1000:1012]
print(f"Window of text: {window!r}\n")
for i in range(1, len(window)):
    print(f"  Given {window[:i]!r:>15}  →  the answer is {window[i]!r}")
print("\n(The model never gets to see the 10% 'test' text while learning, so we can check that it")
print(" actually learned to write and didn't just memorise.)")
''')

md(r'''
## 🏗️ 6. Build the model's "brain"

This is a **Transformer**, the design behind every modern LLM. This code is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. You don't need to understand every line. Here's what the parts do:

| Part | What it does, in plain English |
|---|---|
| **Embedding** | Turns each character's ID into a list of numbers the model can adjust: its "meaning" so far. |
| **Attention** | Every character looks back at the earlier characters and decides which ones matter for guessing what's next. |
| **Feed-forward** | Each character "thinks" about what it gathered from attention. |
| **Block** | Attention + thinking. We stack several blocks so the model can build up more complex ideas. |
| **Output head** | Turns the final result into a score for every possible next character. |

All the model's knowledge is stored in **parameters**: millions of numbers that start out random and get nudged a tiny bit after every guess.
''')

code(r'''
MODEL_SIZES = {   # n_layer = how many blocks, n_head = attention "spotlights" per block, n_embd = numbers per character
    "small":  dict(n_layer=4, n_head=4, n_embd=128),
    "medium": dict(n_layer=6, n_head=8, n_embd=256),
    "large":  dict(n_layer=6, n_head=6, n_embd=384),
}

class Config:
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size=128, dropout=0.1):
        self.vocab_size, self.n_layer, self.n_head, self.n_embd = vocab_size, n_layer, n_head, n_embd
        self.block_size = block_size   # how many characters the model can look back at
        self.dropout = dropout         # randomly ignore some info while training, so it doesn't just memorise

    def to_dict(self):
        return dict(self.__dict__)


class SelfAttention(nn.Module):
    """Each character looks back at the earlier characters and decides which ones matter."""
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.dropout = cfg.n_head, cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # each char makes a question, a label, and a message
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.peek = False            # when True, save the attention pattern so we can draw it
        self.saved_attention = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.peek:   # the slow, see-through version
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)  # no peeking at the future!
            self.saved_attention = weights.detach()
            y = weights @ v
        else:           # the fast version (same maths)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class Block(nn.Module):
    """One layer of the brain: look around (attention), then think (feed-forward)."""
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # what each character means
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # where each character sits
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # reuse the same "meanings" for reading and writing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))      # a score for every possible next character
        loss = None
        if targets is not None:               # how wrong were the guesses?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()


cfg = Config(vocab_size, **MODEL_SIZES[MODEL_SIZE])
model = GPT(cfg).to(device)
print(f"🧠 Your model has {model.num_params():,} parameters (adjustable numbers).")
print(f"   For comparison, GPT-3 has 175,000,000,000. Yours is about {175e9 / model.num_params():,.0f}x smaller.")
''')

md(r'''
## ✍️ 7. Writing, one letter at a time

This is how *every* LLM writes, including ChatGPT:

1. Look at the text so far.
2. Give every possible next character a probability ("`e`: 40%, `a`: 20%, …").
3. **Roll the dice** using those probabilities to pick one.
4. Add it to the text and go back to step 1.

**Temperature** controls the dice. Low temperature means it almost always picks the safest choice (boring, repetitive). High temperature means it takes more risks (creative, then chaotic).

Let's see what our model writes **before** any training. Its parameters are completely random:
''')

code(r'''
@torch.no_grad()
def next_char_probs(prompt):
    """Ask the model: given this text, how likely is each possible next character?"""
    model.eval()
    ids = torch.tensor([encode(prompt)[-cfg.block_size:] or [0]], device=device)
    logits, _ = model(ids)
    return logits[0, -1].float().softmax(dim=-1)

@torch.no_grad()
def generate_text(prompt="\n", length=300, temperature=0.8, show=True):
    """Write `length` new characters, one at a time, continuing from `prompt`."""
    model.eval()
    ids = encode(prompt) or [stoi.get("\n", 0)]
    idx = torch.tensor([ids], device=device)
    for _ in range(length):
        logits, _ = model(idx[:, -cfg.block_size:])
        probs = (logits[0, -1].float() / max(temperature, 1e-3)).softmax(dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)          # roll the dice!
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1)
    out = decode(idx[0].tolist())
    if show:
        print(out)
    return out

generate_text("\n", length=200);
''')

md(r'''
Pure gibberish. The model has no idea yet which letters are common, that words are separated by spaces, or anything else.

## 🔧 8. The live dashboard

The next cell holds the code that draws the dashboard. You can skip reading it. It's just drawing, no AI.
''')

code(r'''
# 🔧 Behind the scenes: drawing the dashboard (you can skip this cell)
import html as _html

def _show(c):
    return {" ": "␣", "\n": "↵", "\t": "⇥"}.get(c, c)

DASH_CSS = """<style>
.llmd{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#1f2430;background:#f4f6fb;border-radius:14px;padding:14px;max-width:1150px}
.llmd *{box-sizing:border-box}
.llmd h3{margin:0 0 6px;font-size:15px;color:#1f2430}
.llmd .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:12px;margin-top:12px}
.llmd .card{background:#fff;border:1px solid #e1e5ee;border-radius:12px;padding:12px 14px}
.llmd .hint{color:#5b6475;font-size:12.5px;margin-top:8px;line-height:1.45}
.llmd .big{font-size:30px;font-weight:700;line-height:1.1}
.llmd .bar{height:12px;background:#e6e9f2;border-radius:6px;overflow:hidden;margin-top:6px}
.llmd .bar>div{height:100%;background:linear-gradient(90deg,#6366f1,#22c55e)}
.llmd pre{white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;background:#111827;color:#e5e7eb;border-radius:8px;padding:8px 10px;margin:3px 0 10px;max-height:160px;overflow:auto}
.llmd .tag{display:inline-block;font-size:11.5px;color:#3f4757;background:#e9ecf5;border-radius:999px;padding:2px 9px}
.llmd .stage{font-size:17px;font-weight:650;margin-top:2px}
.llmd .tiles{display:flex;flex-wrap:wrap;gap:3px;margin-top:8px}
.llmd .tiles span{width:12px;height:12px;border-radius:3px;background:#e1e5ee}
.llmd .tiles span.on{background:#f59e0b}
.llmd .row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px}
.llmd .lbl{width:34px;text-align:center;font-family:ui-monospace,Menlo,monospace;font-weight:700;background:#e9ecf5;border-radius:4px;color:#1f2430}
.llmd .track{flex:1;background:#eef0f6;border-radius:4px;height:16px;overflow:hidden}
.llmd .fill{height:100%;border-radius:4px;background:#6366f1}
.llmd .ans .fill{background:#16a34a}
.llmd .ans .lbl{background:#bbf7d0}
.llmd .pct{width:48px;text-align:right;color:#3f4757;font-variant-numeric:tabular-nums}
.llmd .att{font-family:ui-monospace,Menlo,monospace;font-size:14px;line-height:2;word-break:break-all;color:#1f2430}
.llmd .att span{border-radius:3px}
.llmd .stats{display:flex;flex-wrap:wrap;gap:18px;margin-top:10px;font-size:13px;color:#3f4757}
.llmd .stats b{color:#1f2430;font-size:15px}
.llmd table{border-collapse:collapse;font-size:13px;width:100%}
.llmd td{border-top:1px solid #e1e5ee;padding:5px 6px;vertical-align:top}
.llmd .mono{font-family:ui-monospace,Menlo,monospace}
</style>"""

STAGES = [  # (confusion level, label, explanation)
    (25, "🍼 Babbling", "Random characters. It hasn't learned anything yet."),
    (13, "🔤 Learning the alphabet", "It's figured out which characters are common (spaces, e, t, a...)."),
    (7.5, "🧩 Making word-shaped chunks", "It knows which letters tend to go together, like 'th' and 'ing'."),
    (5, "📖 Spelling real words", "Real words are appearing, and the layout of the text is taking shape."),
    (0, "🎭 Writing with style", "Words, punctuation and structure that look like the original text."),
]

def stage_for(confusion):
    for threshold, label, explanation in STAGES:
        if confusion >= threshold:
            return label, explanation
    return STAGES[-1][1:]

def guess_bars_html(guesses, answer=None):
    rows = []
    for ch, p in guesses:
        cls = "row ans" if ch == answer else "row"
        rows.append(f'<div class="{cls}"><span class="lbl">{_html.escape(_show(ch))}</span>'
                    f'<div class="track"><div class="fill" style="width:{max(p * 100, 0.5):.1f}%"></div></div>'
                    f'<span class="pct">{p * 100:.1f}%</span></div>')
    return "".join(rows)

def attention_html(context, weights):
    top = max(weights) or 1.0
    spans = []
    for ch, w in zip(context, weights):
        a = (w / top) ** 0.7
        spans.append(f'<span style="background:rgba(245,158,11,{a:.2f})">{_html.escape(_show(ch))}</span>')
        if ch == "\n":
            spans.append("<br>")
    return f'<div class="att">{"".join(spans)}<span style="outline:2px dashed #6366f1;border-radius:3px">&nbsp;?&nbsp;</span></div>'

def loss_chart_svg(train_pts, val_pts, total_seconds, ymax, w=520, h=210):
    L, R, T, B = 42, 12, 10, 30
    X = lambda t: L + min(t / total_seconds, 1) * (w - L - R)
    Y = lambda v: T + (1 - min(v, ymax) / ymax) * (h - T - B)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;display:block">']
    for v in range(0, int(ymax) + 1):
        parts.append(f'<line x1="{L}" x2="{w - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e6e9f2"/>'
                     f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="#5b6475">{v}</text>')
    minutes = total_seconds / 60
    step = 1 if minutes <= 12 else 5
    for m in range(0, int(minutes) + 1, step):
        parts.append(f'<text x="{X(m * 60):.1f}" y="{h - 10}" font-size="11" text-anchor="middle" fill="#5b6475">{m}m</text>')
    if len(train_pts) > 1:
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in train_pts)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#6366f1" stroke-width="2.2" stroke-linejoin="round"/>')
    for t, v in val_pts:
        parts.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3.8" fill="#f59e0b" stroke="#fff" stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)

def render_dashboard(S):
    snaps = S["snapshots"]
    last = snaps[-1] if snaps else None
    progress = min(S["elapsed"] / S["total"], 1.0)
    left = max(S["total"] - S["elapsed"], 0)
    status = "✅ Training finished!" if S["done"] else f"⏳ Training… {int(left // 60)}:{int(left % 60):02d} left"
    confusion = math.exp(last["val"]) if last else S["vocab_size"]
    stage, stage_text = stage_for(confusion)
    n_on = max(1, min(S["vocab_size"], round(confusion)))
    tiles = "".join('<span class="on"></span>' if i < n_on else "<span></span>" for i in range(S["vocab_size"]))
    times_read = S["tokens_seen"] / S["data_len"]

    header = f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:baseline">
        <h3 style="font-size:18px">🧠 Live training dashboard</h3><span class="tag">{status}</span></div>
      <div class="bar"><div style="width:{progress * 100:.1f}%"></div></div>
      <div class="stats">
        <span>Practice rounds<br><b>{S['step']:,}</b></span>
        <span>Characters read<br><b>{S['tokens_seen']:,}</b></span>
        <span>Times through the whole text<br><b>{times_read:.1f}×</b></span>
        <span>Brain size<br><b>{S['n_params'] / 1e6:.1f} million</b> parameters</span>
      </div>
    </div>"""

    confusion_card = f"""
    <div class="card">
      <h3>🎲 How confused is the model?</h3>
      <div class="big">like picking from {confusion:.1f} characters</div>
      <div class="tiles">{tiles}</div>
      <div class="stage">{stage}</div>
      <div class="hint">{stage_text}<br>Each square is one character it could type next. At the start it's guessing blindly
      among all {S['vocab_size']}. As it learns, it narrows its guess down to just a few likely ones.</div>
    </div>"""

    ymax = max(4, math.ceil(math.log(S["vocab_size"]) + 0.3))
    chart_card = f"""
    <div class="card">
      <h3>📉 Surprise score (lower is better)</h3>
      {loss_chart_svg(S['train_pts'], [(s['t'], s['val']) for s in snaps], S['total'], ymax)}
      <div class="hint"><span style="color:#6366f1">━ purple line</span>: how surprised the model was by the
      correct answers in its practice text. <span style="color:#b45309">● orange dots</span>: surprise on the
      secret test text it has never trained on. If both go down, it's really learning, not just memorising.</div>
    </div>"""

    if last:
        shown = snaps[:1] + snaps[-3:] if len(snaps) > 4 else snaps
        samples = "".join(f'<span class="tag">after {s["t"]:.0f} seconds · round {s["step"]:,}</span>'
                          f'<pre>{_html.escape(s["sample"])}</pre>' for s in reversed(shown))
        ctx = S["watch_context"]
        guess_card = f"""
        <div class="card">
          <h3>🔮 What letter comes next?</h3>
          <div class="mono" style="font-size:13px;margin-bottom:6px">…{_html.escape(ctx[-60:]).replace(chr(10), '↵')}<b style="color:#6366f1">?</b></div>
          {guess_bars_html(last['guesses'], S['watch_answer'])}
          <div class="hint">The model's top guesses for the next character and how sure it is about each.
          {"The <b style='color:#16a34a'>green</b> one is the right answer." if S['watch_answer'] else ""}
          Watch the right answer climb as it learns.</div>
        </div>"""
        attn_card = f"""
        <div class="card">
          <h3>🔦 Where is it looking? (attention)</h3>
          {attention_html(ctx[-len(last['attn']):], last['attn'])}
          <div class="hint">To guess the <b>?</b>, the model looks back at earlier characters. The
          <span style="background:rgba(245,158,11,.8);border-radius:3px">&nbsp;brighter&nbsp;</span> a character,
          the more attention it paid to it.</div>
        </div>"""
        sample_card = f"""
        <div class="card" style="grid-column:1/-1">
          <h3>✍️ What it writes (newest first)</h3>{samples}
          <div class="hint">Every so often we pause training and ask the model to write something.
          Compare the newest writing with the oldest one at the bottom.</div>
        </div>"""
    else:
        guess_card = attn_card = sample_card = ""

    return (DASH_CSS + '<div class="llmd">' + header + '<div class="grid">' + confusion_card + chart_card
            + guess_card + attn_card + sample_card + "</div></div>")

@torch.no_grad()
def peek_attention(prompt):
    """Run the model once and return how much the last position paid attention to each earlier character."""
    model.eval()
    ids = encode(prompt)[-cfg.block_size:]
    for block in model.blocks:
        block.attn.peek = True
    try:
        model(torch.tensor([ids], device=device))
        maps = torch.stack([b.attn.saved_attention[0, :, -1, :] for b in model.blocks])  # layers × heads × positions
    finally:
        for block in model.blocks:
            block.attn.peek = False
    return maps.float().mean(dim=(0, 1)).tolist()

def top_guesses(prompt, k=8):
    probs = next_char_probs(prompt)
    p, i = probs.topk(min(k, len(probs)))
    return [(itos[int(j)], float(q)) for q, j in zip(p, i)]

def show_guesses(prompt, k=8):
    """Draw the model's top guesses for the next character after `prompt`."""
    display(HTML(DASH_CSS + f'<div class="llmd"><div class="card"><div class="mono">…{_html.escape(prompt[-60:])}'
                 f'<b style="color:#6366f1">?</b></div>{guess_bars_html(top_guesses(prompt, k))}</div></div>'))

def show_attention(prompt):
    """Highlight which earlier characters the model looks at to guess what comes after `prompt`."""
    weights = peek_attention(prompt)
    context = decode(encode(prompt)[-cfg.block_size:])
    display(HTML(DASH_CSS + f'<div class="llmd"><div class="card">{attention_html(context, weights)}</div></div>'))

print("Dashboard ready.")
''')

md(r'''
## 🚀 9. Training

This is where the learning happens. Each **practice round**:

1. Grab 64 random windows of text.
2. The model guesses the next character at every position.
3. We measure how **surprised** it was by the right answers (this is called the **loss**).
4. Nudge all the parameters a tiny bit in the direction that would have made it *less* surprised.

Repeat thousands of times. Nobody programs in grammar rules or spelling. The model works all of it out from the text.

**Watch the dashboard below.** It updates live:
- **🎲 Confusion:** how many characters it is effectively choosing between. It starts out guessing blindly.
- **📉 Surprise score:** should drop fast at first, then slowly.
- **🔮 Next letter:** watch the right answer climb to the top.
- **🔦 Attention:** which earlier characters it looks at when guessing.
- **✍️ Writing samples:** gibberish → word-ish → words → Shakespeare-ish.
''')

code(r'''
start_fresh = True        # set to False to keep training the model you already have
BATCH_SIZE = 64           # windows of text per practice round
LEARNING_RATE = 1e-3      # how big each nudge is

if start_fresh:
    torch.manual_seed(1337)
    model = GPT(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), weight_decay=0.1,
                              fused=device == "cuda")
use_amp = device == "cuda"   # use faster, lower-precision maths on the GPU
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
train_gpu, val_gpu = train_data.to(device), val_data.to(device)
offsets = torch.arange(cfg.block_size, device=device)

def get_batch(source):
    starts = torch.randint(len(source) - cfg.block_size - 1, (BATCH_SIZE,), device=device)
    x = source[starts[:, None] + offsets]
    y = source[starts[:, None] + offsets + 1]   # the answers: each character shifted by one
    return x, y

@torch.no_grad()
def test_loss(rounds=10):
    model.eval()
    total = 0.0
    for _ in range(rounds):
        x, y = get_batch(val_gpu)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / rounds

# pick the "what comes next?" example for the dashboard
if WATCH_PROMPT:
    watch_context, watch_answer = WATCH_PROMPT, None
elif DATASET == "shakespeare":
    watch_context, watch_answer = "ROMEO:\nBut soft, what light through yonder window brea", "k"
else:  # find a spot in the test text in the middle of a word
    val_text = text[split:]
    pos = next((i for i in range(2000, len(val_text) - 1) if val_text[i - 3:i + 1].isalpha()), len(val_text) - 1)
    watch_context, watch_answer = val_text[max(0, pos - 60):pos], val_text[pos]

total_seconds = TRAIN_MINUTES * 60
SNAPSHOT_ROUNDS = {0, 25, 50, 100, 200, 350, 500, 750, 1000}
log = {"dataset": DATASET, "model_size": MODEL_SIZE, "config": cfg.to_dict(), "n_params": model.num_params(),
       "vocab": chars, "data_len": len(train_data), "batch_size": BATCH_SIZE, "total_seconds": total_seconds,
       "watch_context": watch_context, "watch_answer": watch_answer, "train_pts": [], "snapshots": []}
state = dict(step=0, elapsed=0.0, total=total_seconds, done=False, vocab_size=vocab_size,
             n_params=model.num_params(), tokens_seen=0, data_len=len(train_data),
             train_pts=log["train_pts"], snapshots=log["snapshots"],
             watch_context=watch_context, watch_answer=watch_answer)

def take_snapshot(step, elapsed):
    snap = dict(step=step, t=round(elapsed, 1), val=test_loss(),
                sample=generate_text("\n", length=180, temperature=0.8, show=False).strip("\n"),
                guesses=top_guesses(watch_context), attn=[round(a, 4) for a in peek_attention(watch_context)])
    log["snapshots"].append(snap)
    model.train()

def send_progress(step, elapsed, done=False):
    snap = log["snapshots"][-1]
    send_live(dict(dataset=DATASET, size=MODEL_SIZE, step=step, t=round(elapsed), total=total_seconds,
                   val=round(snap["val"], 4), train=round(smooth_loss or snap["val"], 4), vocab=vocab_size,
                   sample=snap["sample"][:160], done=done))

dashboard = display(HTML(render_dashboard(state)), display_id=True)
model.train()
smooth_loss, step, last_draw, next_timed_snapshot, last_sent = None, 0, 0.0, 30.0, -1e9
start = time.time()
paused = 0.0   # time spent drawing the dashboard doesn't count against training time

while True:
    elapsed = time.time() - start - paused
    if elapsed >= total_seconds:
        break
    if step in SNAPSHOT_ROUNDS or elapsed >= next_timed_snapshot:
        t0 = time.time()
        take_snapshot(step, elapsed)
        if elapsed >= next_timed_snapshot:
            next_timed_snapshot += 30.0
        if elapsed - last_sent >= 70:   # a class-board update about every 75 seconds
            send_progress(step, elapsed)
            last_sent = elapsed
        paused += time.time() - t0

    # learning rate: warm up gently, then take smaller and smaller steps as training goes on
    lr = LEARNING_RATE * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * elapsed / total_seconds)))
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_gpu)                          # 1. grab practice text
    with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
        _, loss = model(x, y)                            # 2+3. guess, and measure surprise
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()                        # 4. work out which way to nudge every parameter...
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)                               #    ...and nudge them
    scaler.update()
    step += 1

    if step % 10 == 0:
        value = loss.item()
        smooth_loss = value if smooth_loss is None else 0.9 * smooth_loss + 0.1 * value
        log["train_pts"].append((round(elapsed, 1), round(smooth_loss, 4)))
    if time.time() - last_draw > 2.0:
        state.update(step=step, elapsed=elapsed, tokens_seen=step * BATCH_SIZE * cfg.block_size)
        dashboard.update(HTML(render_dashboard(state)))
        last_draw = time.time()

take_snapshot(step, total_seconds)
send_progress(step, total_seconds, done=True)
state.update(step=step, elapsed=total_seconds, done=True, tokens_seen=step * BATCH_SIZE * cfg.block_size)
dashboard.update(HTML(render_dashboard(state)))
log["final_step"] = step
print(f"Done! {step:,} practice rounds. Final surprise score on the test text: {log['snapshots'][-1]['val']:.2f}")
''')

md(r'''
## 🎭 10. Play with your model!

Your model is trained. Change the **prompt** (how the text starts), the **length**, and the **temperature**, then run the cell again. Every run is different because the model rolls dice.
''')

code(r'''
generate_text(prompt="JULIET:\n", length=500, temperature=0.8);
''')

md(r'''
### 🌡️ Temperature experiment

Same prompt, three temperatures. Which one sounds most like Shakespeare? Which is the most "creative"?
''')

code(r'''
for temp in [0.3, 0.8, 1.5]:
    print(f"━━━━━━━━━━ temperature {temp} ━━━━━━━━━━")
    generate_text(prompt="KING:\n", length=250, temperature=temp)
    print()
''')

md(r'''
### 🔬 Look inside the model's head

An LLM never "decides" on a whole sentence. At every step it only produces **probabilities for the next character**. Try different prompts:
- `"To be or not to b"`: does it know the famous line?
- `"The kin"`: what words could this be?
- Something the model has never seen, like your own name.
''')

code(r'''
show_guesses("To be or not to b")
show_attention("To be or not to b")
''')

md(r'''
### 🐢 Writing in slow motion

Here's the model writing 12 characters, showing its top 3 choices each time and the one the dice picked.
''')

code(r'''
def slow_motion(prompt, steps=12, temperature=0.8):
    text_so_far = prompt
    rows = []
    for _ in range(steps):
        probs = (next_char_probs(text_so_far).log() / temperature).softmax(dim=-1)
        picked = itos[int(torch.multinomial(probs, 1))]
        top = probs.topk(3)
        choices = " &nbsp; ".join(f"<b>{_html.escape(_show(itos[int(i)]))}</b> {float(p) * 100:.0f}%"
                                  for p, i in zip(top.values, top.indices))
        rows.append(f'<tr><td class="mono">…{_html.escape(text_so_far[-25:]).replace(chr(10), "↵")}</td>'
                    f'<td>{choices}</td><td class="mono" style="color:#16a34a"><b>{_html.escape(_show(picked))}</b></td></tr>')
        text_so_far += picked
    display(HTML(DASH_CSS + '<div class="llmd"><div class="card"><table><tr><td><b>Text so far</b></td>'
                 '<td><b>Top 3 guesses</b></td><td><b>Dice picked</b></td></tr>' + "".join(rows) + "</table></div></div>"))

slow_motion("ROMEO:\nI will")
''')

md(r'''
## 💾 11. Save your work

This saves your model and a **training log**. Download `training_log.json` from the **Output** panel (Kaggle) or the Files panel (Colab). You can replay your training run in the **LLM Concept Explorer** web page your teacher shares.
''')

code(r'''
out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
torch.save({"config": cfg.to_dict(), "chars": chars, "dataset": DATASET,
            "model": {k: v.half() for k, v in model.state_dict().items()}},
           os.path.join(out_dir, "my_model.pt"))
with open(os.path.join(out_dir, "training_log.json"), "w") as f:
    json.dump(log, f)
print(f"Saved my_model.pt and training_log.json to {os.path.abspath(out_dir)}")
''')

md(r'''
## 🏆 12. Challenges (pick one!)

1. **New author.** Set `DATASET = "sherlock"` in Settings and Run All. Can it write like a detective story?
2. **Baby-name inventor.** Set `DATASET = "names"`. Then run `generate_text("\n", 300)`. Are the names real, or did it invent new ones?
3. **Robot programmer.** Set `DATASET = "python_code"`. Does it learn indentation and `def`?
4. **Your own data.** Make a `.txt` file (lyrics, your favourite book, a chat log, recipes), at least ~200 KB for good results. On Kaggle: **+ Add Input → Upload → New Dataset**. Then set `DATASET = "/kaggle/input/<your-dataset>/<file>.txt"`.
5. **Brain surgery.** Try `MODEL_SIZE = "small"` vs `"large"`. Does a bigger brain write better in the same 6 minutes?
6. **Patience.** Set `TRAIN_MINUTES = 15`. How much better does it get? Does it ever stop improving?

### 🤔 Think about it
- Your model has about 5 million parameters and read about 1 MB of text. Modern LLMs have **hundreds of billions** of parameters and read **trillions** of words: basically a big chunk of the internet. What changes when you scale up that much?
- Your model can write Shakespeare-ish text, but it can't answer a question. Why not? (Hint: what was the *only* thing it was ever trained to do?)
- If the text it learns from has mistakes or unfair stereotypes in it, what will the model learn?
''')

md(r'''
---
## 🛟 Backup plan: load a model your teacher already trained

If training didn't work (no GPU, ran out of time…), run this cell. It loads a Shakespeare model trained ahead of time, and then all the "Play with your model" cells above work.
''')

code(r'''
def load_pretrained(path=None):
    global model, cfg, chars, stoi, itos, vocab_size
    if path is None:
        hits = (glob.glob("/kaggle/input/**/shakespeare_pretrained.pt", recursive=True)
                + glob.glob("checkpoints/shakespeare_pretrained.pt") + glob.glob("../checkpoints/shakespeare_pretrained.pt"))
        if hits:
            path = hits[0]
        else:
            path = "shakespeare_pretrained.pt"
            urllib.request.urlretrieve(f"{GITHUB_RAW}/checkpoints/shakespeare_pretrained.pt", path)
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["config"])
    chars = ckpt["chars"]
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = GPT(cfg).to(device)
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    model.eval()
    print(f"Loaded {path}: a {model.num_params() / 1e6:.1f}M parameter model trained on {ckpt.get('dataset', '?')}")

# load_pretrained()   # ← remove the # at the start of this line, then run the cell
''')


def build(path):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook", "dataSources": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(root, "notebooks", "train_your_own_llm.ipynb"))
