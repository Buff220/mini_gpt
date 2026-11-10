"""
TINY GPT  -  build an LLM from scratch AND watch it learn, as a slide presentation.

Run:        python tiny_gpt_presentation.py
Options:    --steps 3000     number of training steps (default 3000)
            --headless       no windows; only saves slides (PNG + PDF). Use this on Colab/servers
            --chat-only      skip training, load saved tiny_gpt.pt and just chat
            --no-chat        train + slides only

In the live version: CLICK the window or PRESS A KEY to go to the next slide.
Everything is also saved to ./presentation/  (slide_XX.png + tiny_gpt_presentation.pdf)

Install:    pip install torch matplotlib numpy
"""
import os, sys, math, time, argparse, textwrap, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--chat-only", action="store_true")
ap.add_argument("--no-chat", action="store_true")
args = ap.parse_args()

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
if args.headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Shakespeare text contains '$'; matplotlib would treat pairs of them as math formulas and crash.
try:
    plt.rcParams["text.parse_math"] = False
except KeyError:
    pass  # very old matplotlib: run  pip install -U matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

HEADLESS = args.headless or "agg" in matplotlib.get_backend().lower()

# ----------------------------------------------------------------------------
# 0. SETTINGS  (the "size" of our brain). Change them and see what happens!
# ----------------------------------------------------------------------------
BLOCK   = 64      # context window: how many previous characters the model can see
N_EMBD  = 128     # size of the vector that represents each character
N_HEAD  = 4       # attention heads (4 "ways of looking" at the context in parallel)
N_LAYER = 4       # how many transformer blocks are stacked
DROPOUT = 0.1
BATCH   = 64      # chunks processed at once
LR      = 1e-3    # learning rate = size of each nudge to the weights
STEPS   = args.steps

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")
torch.manual_seed(1337)

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    print("Downloading Tiny Shakespeare ...")
    urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt", encoding="utf-8").read()

chars = sorted(set(text))                 # the vocabulary: every distinct character
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]      # text  -> numbers
decode = lambda ids: "".join(itos[i] for i in ids)         # numbers -> text

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]      # 90% to learn from, 10% to test on


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])            # inputs
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])    # answers = same text shifted by 1 char
    return x.to(DEVICE), y.to(DEVICE)


# ----------------------------------------------------------------------------
# 2. THE MODEL  (a small GPT)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    """One transformer block = self-attention (characters talk to each other)
                             + MLP (each character thinks on its own)."""
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)      # makes Query, Key, Value for every character
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK, BLOCK)))   # can't look into the future
        self.keep, self.attn = False, None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // N_HEAD)   # "how relevant is each earlier char to me?"
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf")).softmax(-1)
        if self.keep:
            self.attn = att.detach().cpu()                         # saved only so we can draw it
        y = (self.drop(att) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))                            # residual connection
        x = x + self.drop(self.mlp(self.ln2(x)))                   # residual connection
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, N_EMBD)         # char id -> vector of 128 numbers
        self.pos_emb = nn.Embedding(BLOCK, N_EMBD)     # position -> vector of 128 numbers
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, V)               # vector -> one score per possible next char
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))               # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8):
        for _ in range(n):
            logits, _ = self(idx[:, -BLOCK:])
            probs = F.softmax(logits[:, -1, :] / max(temp, 1e-3), dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


model = GPT().to(DEVICE)
N_PARAMS = sum(p.numel() for p in model.parameters())


def sample(prompt, n=300, temp=0.8):
    was_training = model.training
    model.eval()
    ids = encode(prompt) or [stoi.get("\n", 0)]
    out = model.generate(torch.tensor([ids], device=DEVICE), n, temp)[0].tolist()
    model.train(was_training)
    return decode(out)


@torch.no_grad()
def estimate_loss(iters=10):
    model.eval()
    res = {}
    for split in ("train", "val"):
        res[split] = float(np.mean([model(*get_batch(split))[1].item() for _ in range(iters)]))
    model.train()
    return res


EX_CONTEXT = "First Citizen:\nBefore we proceed any f"     # the model must guess the next char: 'u'
EX_TRUTH = "u"


@torch.no_grad()
def peek(context=EX_CONTEXT):
    """Run the model on one example and capture probabilities + attention maps (for drawing)."""
    model.eval()
    for b in model.blocks:
        b.keep = True
    ids = torch.tensor([encode(context)], device=DEVICE)
    logits, _ = model(ids)
    probs = F.softmax(logits[0, -1], -1).cpu().numpy()
    atts = [b.attn[0].numpy() for b in model.blocks]       # each (heads, T, T)
    for b in model.blocks:
        b.keep = False
    model.train()
    return probs, atts


# ----------------------------------------------------------------------------
# 3. PRESENTATION HELPERS
# ----------------------------------------------------------------------------
BG, PANEL, FG = "#0f172a", "#1e293b", "#e2e8f0"
ACC, PINK, GOOD, WARN, DIM = "#38bdf8", "#f472b6", "#4ade80", "#fbbf24", "#94a3b8"
os.makedirs("presentation", exist_ok=True)
pdf = PdfPages("presentation/tiny_gpt_presentation.pdf")
slide_no = [0]
vis = lambda c: {"\n": "\u23ce", " ": "\u2423"}.get(c, c)       # make invisible chars visible


def new_fig(title, subtitle=""):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.04, 0.95, title, color=ACC, fontsize=26, weight="bold", va="top")
    if subtitle:
        fig.text(0.04, 0.885, textwrap.fill(subtitle, 120), color=FG, fontsize=13.5, va="top")
    return fig


def style(ax, title=None):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=DIM, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#334155")
    if title:
        ax.set_title(title, color=FG, fontsize=11, loc="left", pad=6)


def save_slide(fig):
    slide_no[0] += 1
    fig.savefig(f"presentation/slide_{slide_no[0]:02d}.png", dpi=100, facecolor=BG)
    pdf.savefig(fig, facecolor=BG)


def show(fig):
    """Save the slide, then (if not headless) show it and wait for a click / key."""
    save_slide(fig)
    if not HEADLESS:
        fig.text(0.97, 0.02, "click or press any key  \u25b6", color=WARN, fontsize=11, ha="right")
        plt.show(block=False)
        try:
            plt.waitforbuttonpress()
        except Exception:
            pass
    plt.close(fig)


def box(ax, x, y, w, h, txt, ec=ACC, fs=12, tc=FG, fc=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=tc, fontsize=fs)


def arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=FG, lw=2, connectionstyle=f"arc3,rad={rad}"))


# ----------------------------------------------------------------------------
# 4. SLIDES BEFORE TRAINING
# ----------------------------------------------------------------------------
def slide_title():
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.5, 0.62, "Building an LLM from scratch", color=ACC, fontsize=44, weight="bold", ha="center")
    fig.text(0.5, 0.50, "a tiny GPT that learns to write like Shakespeare, one character at a time",
             color=FG, fontsize=20, ha="center")
    fig.text(0.5, 0.36,
             f"{N_PARAMS:,} parameters  \u2022  {len(text):,} characters of training text  \u2022  device: {DEVICE}\n"
             f"(ChatGPT-class models have ~1,000,000x more parameters. Same idea, much bigger.)",
             color=DIM, fontsize=14, ha="center", linespacing=1.8)
    show(fig)


def slide_data():
    fig = new_fig("Step 1 \u00b7 The data",
                  "An LLM learns from raw text. We use Tiny Shakespeare: a plain .txt file with plays by Shakespeare.")
    fig.text(0.04, 0.80, "First lines of the file:", color=WARN, fontsize=13)
    sample_txt = "\n".join(text.split("\n")[:17])
    fig.text(0.04, 0.77, sample_txt, color=FG, fontsize=11, family="monospace", va="top", linespacing=1.5)
    fig.text(0.04, 0.17,
             f"{len(text):,} characters in total\n{V} different characters  \u2192  our \"vocabulary\"\n"
             f"90% = {n_train:,} chars to learn from\n10% = {len(data) - n_train:,} chars kept hidden to test it",
             color=GOOD, fontsize=14, va="top", linespacing=1.7)
    ax = fig.add_axes([0.58, 0.10, 0.38, 0.68])
    style(ax, "How often each character appears (top 25)")
    counts = np.bincount(data.numpy(), minlength=V)
    top = np.argsort(counts)[::-1][:25][::-1]
    ax.barh([vis(itos[i]) for i in top], counts[top], color=ACC)
    ax.tick_params(axis="y", labelsize=10, labelcolor=FG)
    ax.text(0.98, 0.03, "\u2423 = space   \u23ce = new line", transform=ax.transAxes, ha="right", color=WARN, fontsize=10)
    show(fig)


def slide_tokens():
    fig = new_fig("Step 2 \u00b7 Tokenization: text \u2192 numbers",
                  "Neural networks only understand numbers. Every character in our vocabulary gets an ID. "
                  "(ChatGPT uses chunks of words instead of single characters, but it's the same idea.)")
    s = "First Citizen:"
    ids = encode(s)
    ax = fig.add_axes([0.04, 0.42, 0.92, 0.36])
    ax.set_xlim(0, len(s)); ax.set_ylim(0, 3); ax.axis("off")
    for i, (c, t) in enumerate(zip(s, ids)):
        ax.add_patch(Rectangle((i + 0.06, 2.0), 0.88, 0.9, fc=PANEL, ec=ACC, lw=1.5))
        ax.text(i + 0.5, 2.45, vis(c), ha="center", va="center", color=FG, fontsize=20, family="monospace")
        arrow(ax, i + 0.5, 1.95, i + 0.5, 1.15)
        ax.add_patch(Rectangle((i + 0.06, 0.2), 0.88, 0.9, fc=PANEL, ec=PINK, lw=1.5))
        ax.text(i + 0.5, 0.65, str(t), ha="center", va="center", color=PINK, fontsize=20, family="monospace")
    fig.text(0.04, 0.37, "The complete vocabulary (id:character):", color=WARN, fontsize=13)
    items = [f"{i}:{vis(c)}" for i, c in enumerate(chars)]
    lines = [" ".join(items[j:j + 13]) for j in range(0, len(items), 13)]
    fig.text(0.04, 0.34, "\n".join(lines), color=FG, fontsize=12.5, family="monospace", va="top", linespacing=1.6)
    show(fig)


def slide_batches():
    fig = new_fig("Step 3 \u00b7 Making training examples",
                  "The task: given the characters so far, predict the NEXT one. The answer key is simply the same "
                  "text shifted by one character, so we never need to label anything by hand!")
    n = 16
    x_txt, y_txt = text[:n], text[1:n + 1]
    ax = fig.add_axes([0.04, 0.50, 0.92, 0.30])
    ax.set_xlim(-2.2, n); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.text(-2.1, 2.05, "input x", color=ACC, fontsize=14, va="center")
    ax.text(-2.1, 0.55, "target y", color=GOOD, fontsize=14, va="center")
    for i in range(n):
        ax.add_patch(Rectangle((i + .06, 1.6), .88, .9, fc=PANEL, ec=ACC, lw=1.5))
        ax.text(i + .5, 2.05, vis(x_txt[i]), ha="center", va="center", color=FG, fontsize=17, family="monospace")
        ax.add_patch(Rectangle((i + .06, .1), .88, .9, fc=PANEL, ec=GOOD, lw=1.5))
        ax.text(i + .5, .55, vis(y_txt[i]), ha="center", va="center", color=GOOD, fontsize=17, family="monospace")
        arrow(ax, i + .5, 1.55, i + .5, 1.05)
    lines = ["One chunk of 16 characters gives 16 separate lessons:\n"]
    for t in range(6):
        lines.append(f"   sees  {repr(x_txt[:t + 1]):<18}\u2192  must predict  {repr(y_txt[t])}")
    lines.append("   ...")
    fig.text(0.04, 0.44, "\n".join(lines), color=FG, fontsize=13.5, family="monospace", va="top", linespacing=1.6)
    fig.text(0.04, 0.12,
             f"In training we grab {BATCH} random chunks of {BLOCK} characters at once (a \"batch\")\n"
             f"= {BATCH} x {BLOCK} = {BATCH * BLOCK:,} predictions to check per training step.",
             color=GOOD, fontsize=14, va="top", linespacing=1.6)
    show(fig)


def slide_architecture():
    fig = new_fig("Step 4 \u00b7 The model: a Transformer (GPT)",
                  "Data flows bottom \u2192 top. B = batch size, T = number of characters, "
                  f"E = {N_EMBD} (vector size), V = {V} (vocabulary).")
    ax = fig.add_axes([0.04, 0.03, 0.52, 0.82]); ax.set_xlim(0, 10); ax.set_ylim(0, 12.2); ax.axis("off")
    box(ax, 1, 0.1, 8, 1.1, f"Input characters   shape (B, T)   e.g. \"To be or n\" \u2192 [32, 53, 1, 40, ...]", ec=DIM, fs=10.5)
    arrow(ax, 5, 1.25, 5, 1.85)
    box(ax, 1, 1.9, 8, 1.2, f"Token embedding  +  Position embedding\nid \u2192 {N_EMBD} numbers;  shape (B, T, E)", ec=PINK, fs=11)
    arrow(ax, 5, 3.15, 5, 3.75)
    ax.add_patch(FancyBboxPatch((0.6, 3.8), 8.8, 5.6, boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec=WARN, lw=1.5, ls="--"))
    ax.text(9.25, 9.1, f"\u00d7 {N_LAYER} blocks", color=WARN, fontsize=12, ha="right", weight="bold")
    box(ax, 1.2, 4.0, 7.6, 1.2, f"LayerNorm \u2192 Self-Attention ({N_HEAD} heads)\n\"each character looks back at earlier ones\"   + residual", fs=10.5)
    arrow(ax, 5, 5.25, 5, 5.9)
    box(ax, 1.2, 5.95, 7.6, 1.2, f"LayerNorm \u2192 MLP ({N_EMBD} \u2192 {4 * N_EMBD} \u2192 {N_EMBD})\n\"each character thinks on its own\"   + residual", fs=10.5)
    ax.text(5, 7.55, "(the two boxes above are repeated, each with its own weights)", color=DIM, fontsize=9, ha="center")
    arrow(ax, 5, 9.45, 5, 9.95)
    box(ax, 1, 10.0, 8, 1.0, f"Final LayerNorm + Linear   \u2192   logits (B, T, V={V})", ec=GOOD, fs=11)
    ax.text(5, 11.7, f"softmax \u2192 probability for each of the {V} possible next characters", color=GOOD, fontsize=11.5, ha="center")
    arrow(ax, 5, 11.05, 5, 11.4)

    tok, pos = model.tok_emb.weight.numel(), model.pos_emb.weight.numel()
    att = sum(p.numel() for p in list(model.blocks[0].qkv.parameters()) + list(model.blocks[0].proj.parameters()))
    mlp = sum(p.numel() for p in model.blocks[0].mlp.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    fig.text(0.60, 0.80, "Where the learnable numbers live", color=WARN, fontsize=16, weight="bold")
    fig.text(0.60, 0.75,
             f"Token embedding       {tok:>9,}   ({V} x {N_EMBD} lookup table)\n"
             f"Position embedding    {pos:>9,}   ({BLOCK} x {N_EMBD})\n"
             f"Attention  (per block){att:>9,}\n"
             f"MLP        (per block){mlp:>9,}\n"
             f"Output layer          {head:>9,}\n"
             f"{'-' * 52}\n"
             f"TOTAL                 {N_PARAMS:>9,}   <- these get nudged\n",
             color=FG, fontsize=11.5, family="monospace", va="top", linespacing=1.6)
    fig.text(0.60, 0.46,
             "These numbers are the model's \"memory\".\n\n"
             "Before training they are small random values,\n"
             "so the model knows nothing.\n\n"
             "Training = slowly adjusting all of them so that\n"
             "the correct next character gets a higher probability.",
             color=FG, fontsize=13.5, va="top", linespacing=1.4)
    show(fig)


def slide_untrained():
    probs, _ = peek()
    loss0 = estimate_loss(5)["val"]
    fig = new_fig("Before training: the model knows nothing",
                  f"Random weights \u2192 it assigns almost the same probability to all {V} characters.")
    fig.text(0.04, 0.78, "What it writes now (prompt: \"ROMEO:\"):", color=WARN, fontsize=13)
    gib = sample("ROMEO:\n", 330, 1.0)
    fig.text(0.04, 0.75, "\n".join(textwrap.fill(l, 95) for l in gib.split("\n")),
             color=FG, fontsize=11.5, family="monospace", va="top", linespacing=1.5)
    fig.text(0.04, 0.22,
             f"Loss measures \"how surprised is the model by the right answer\":  loss = -ln(P(correct char))\n"
             f"A pure random guess gives   -ln(1/{V}) = {math.log(V):.3f}\n"
             f"Our untrained model measures  {loss0:.3f}        \u2190 almost exactly the random-guess number\n"
             f"Goal of training: push the loss down.",
             color=GOOD, fontsize=14, va="top", linespacing=1.8)
    show(fig)


def slide_loop():
    fig = new_fig("Step 5 \u00b7 The training loop",
                  f"These five steps repeat {STEPS:,} times. Each repeat is called a \"step\". Next slide: watch it live.")
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.50]); ax.set_xlim(0, 16); ax.set_ylim(0, 6); ax.axis("off")
    labels = [("1  Grab a batch", f"{BATCH} random chunks\nx = inputs\ny = right answers", ACC),
              ("2  Forward pass", "model(x) gives a\nscore for every\npossible next char", ACC),
              ("3  Loss", "how far from y?\n-ln P(correct)\none number", PINK),
              ("4  Backward", "gradients: for each\nweight, which way\nreduces the loss?", WARN),
              ("5  Update", "weight \u2190 weight\n  \u2212 lr \u00d7 gradient\n(a tiny nudge)", GOOD)]
    for i, (t, d, col) in enumerate(labels):
        x = 0.2 + i * 3.1
        box(ax, x, 2.6, 2.7, 3.0, "", ec=col)
        ax.text(x + 1.35, 5.1, t, ha="center", color=col, fontsize=13, weight="bold")
        ax.text(x + 1.35, 3.7, d, ha="center", va="center", color=FG, fontsize=11.5, linespacing=1.5)
        if i < 4:
            arrow(ax, x + 2.72, 4.1, x + 3.08, 4.1)
    arrow(ax, 14.5, 2.5, 1.5, 2.5, rad=-0.35)
    ax.text(8, 0.4, f"repeat  {STEPS:,}  times", color=WARN, fontsize=14, ha="center")
    fig.text(0.04, 0.24,
             f"Optimizer: AdamW, learning rate {LR} (warm-up, then slowly decreasing).\n"
             "After many repeats: the right characters get higher probability \u2192 the loss falls \u2192 the text gets better.",
             color=FG, fontsize=13.5, va="top", linespacing=1.6)
    show(fig)


# ----------------------------------------------------------------------------
# 5. LIVE TRAINING DASHBOARD
# ----------------------------------------------------------------------------
class Dashboard:
    def __init__(self):
        if not HEADLESS:
            plt.ion()
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.text(0.04, 0.975, "Step 5 \u00b7 Training, live", color=ACC, fontsize=22, weight="bold", va="top")
        self.cap = self.fig.text(0.04, 0.915, "", color=FG, fontsize=13, va="top")
        gs = GridSpec(3, 3, figure=self.fig, left=0.05, right=0.97, top=0.86, bottom=0.04,
                      hspace=0.62, wspace=0.28, height_ratios=[1, 1, 0.62])
        self.a_loss = self.fig.add_subplot(gs[0, 0])
        self.a_pred = self.fig.add_subplot(gs[0, 1])
        self.a_att = self.fig.add_subplot(gs[0, 2])
        self.a_gen = self.fig.add_subplot(gs[1, 0:2])
        self.a_emb = self.fig.add_subplot(gs[1, 2])
        self.a_stat = self.fig.add_subplot(gs[2, :])
        self.steps, self.train, self.vsteps, self.val = [], [], [], []
        self.gen_txt, self.t0, self.ema = "", time.time(), None
        self.emb_im = None
        self.snap = {0, 200, 1000, STEPS}

    def caption(self, loss):
        if loss > 3.3:
            return "Still almost random guessing (random = 4.17). It starts by learning which characters are common."
        if loss > 2.6:
            return "Learning letter frequencies: lots of e, t, a, spaces and line breaks. Words aren't real yet."
        if loss > 2.1:
            return "Short words start to appear. It's discovering spelling, one pattern at a time."
        if loss > 1.8:
            return "It learns the script FORMAT: NAME in capitals, a colon, then a line of speech."
        return "Word-like text with grammar-ish rhythm. About the best a model this tiny can do."

    def update(self, step, loss, gnorm, lr, new_val=None, do_gen=False, final=False):
        self.steps.append(step)
        self.ema = loss if self.ema is None else 0.9 * self.ema + 0.1 * loss
        self.train.append(self.ema)
        if new_val is not None:
            self.vsteps.append(step); self.val.append(new_val)
        probs, atts = peek()
        self.cap.set_text(self.caption(self.ema))

        # (1) loss curve
        a = self.a_loss; a.clear(); style(a, "\u2460 LOSS = how wrong it is (lower is better)")
        a.plot(self.steps, self.train, color=ACC, lw=2, label="train")
        if self.val:
            a.plot(self.vsteps, self.val, color=PINK, lw=2, marker="o", ms=3, label="validation (unseen text)")
        a.axhline(math.log(V), color=DIM, ls="--", lw=1)
        a.text(STEPS * 0.98, math.log(V) + 0.05, "random guess", color=DIM, fontsize=8, ha="right")
        a.set_xlim(0, STEPS); a.set_ylim(1.0, 4.5)
        a.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=FG, fontsize=8, loc="upper right")
        a.set_xlabel("step", color=DIM, fontsize=8)

        # (2) next-char prediction
        a = self.a_pred; a.clear()
        style(a, f"\u2461 PREDICTION after: \"...any f\"   (truth: '{EX_TRUTH}')")
        top = np.argsort(probs)[::-1][:8][::-1]
        cols = [GOOD if itos[i] == EX_TRUTH else ACC for i in top]
        a.barh([vis(itos[i]) for i in top], probs[top], color=cols)
        a.set_xlim(0, 1)
        a.tick_params(axis="y", labelsize=11, labelcolor=FG)
        for yy, i in enumerate(top):
            a.text(min(probs[i] + 0.01, 0.8), yy, f"{probs[i] * 100:.1f}%", va="center", color=FG, fontsize=8)
        a.text(0.97, 0.05, f"P(correct) = {probs[stoi[EX_TRUTH]] * 100:.1f}%", transform=a.transAxes,
               ha="right", color=GOOD, fontsize=10, weight="bold")

        # (3) attention
        a = self.a_att; a.clear()
        style(a, "\u2462 ATTENTION (last layer, head 1): who looks at whom")
        T = len(EX_CONTEXT)
        a.imshow(atts[-1][0], cmap="magma", aspect="auto", vmin=0, vmax=0.6)
        a.set_xticks(range(T)); a.set_xticklabels([vis(c) for c in EX_CONTEXT], fontsize=5.5, color=DIM)
        a.set_yticks(range(T)); a.set_yticklabels([vis(c) for c in EX_CONTEXT], fontsize=5.5, color=DIM)
        a.set_xlabel("character being looked at (earlier)", color=DIM, fontsize=7)
        a.set_ylabel("character doing the looking", color=DIM, fontsize=7)

        # (4) generated text
        if do_gen or not self.gen_txt:
            g = sample("ROMEO:\n", 320, 0.8)
            self.gen_txt = "\n".join(textwrap.fill(l, 92) for l in g.split("\n"))
        a = self.a_gen; a.clear(); style(a, "\u2463 WHAT IT WRITES RIGHT NOW   (prompt: \"ROMEO:\")")
        a.set_xticks([]); a.set_yticks([])
        a.text(0.015, 0.95, self.gen_txt, transform=a.transAxes, va="top", color=GOOD,
               fontsize=9.5, family="monospace", linespacing=1.35)

        # (5) embedding table
        W = model.tok_emb.weight.detach().cpu().numpy()
        lim = max(0.05, float(np.percentile(np.abs(W), 99)))
        a = self.a_emb
        if self.emb_im is None:
            style(a, "\u2464 EMBEDDING TABLE (the numbers moving)")
            self.emb_im = a.imshow(W, cmap="coolwarm", aspect="auto", vmin=-lim, vmax=lim)
            a.set_xlabel(f"{N_EMBD} numbers per character", color=DIM, fontsize=8)
            a.set_ylabel(f"{V} characters", color=DIM, fontsize=8)
        else:
            self.emb_im.set_data(W); self.emb_im.set_clim(-lim, lim)
        a.set_title(f"\u2464 EMBEDDING TABLE   colour = value, scale \u00b1{lim:.2f}", color=FG, fontsize=11, loc="left", pad=6)

        # stats
        el = time.time() - self.t0
        eta = el / (step + 1) * (STEPS - step)
        a = self.a_stat; a.clear(); a.axis("off"); a.set_facecolor(BG)
        a.text(0.0, 1.0,
               f"step            {step:>6,} / {STEPS:,}\n"
               f"train loss      {self.ema:>9.3f}\n"
               f"val loss        {(self.val[-1] if self.val else float('nan')):>9.3f}\n"
               f"learning rate   {lr:>9.5f}\n"
               f"gradient size   {gnorm:>9.3f}\n"
               f"chars seen      {step * BATCH * BLOCK:>9,}\n"
               f"time / ETA      {el:5.0f}s / {eta:4.0f}s",
               transform=a.transAxes, va="top", color=WARN, fontsize=11, family="monospace", linespacing=1.45)
        a.text(0.40, 1.0,
               "How to read this screen\n"
               "\u2460 the line should go DOWN: the model is making fewer mistakes\n"
               "\u2461 bars = its belief about the next letter; green bar = the true answer\n"
               "\u2462 bright square = this character is paying attention to that earlier one\n"
               "\u2463 free-running text from the current weights\n"
               "\u2464 every cell is a learnable number; each step nudges them a tiny bit",
               transform=a.transAxes, va="top", color=FG, fontsize=11, linespacing=1.5)

        if step in self.snap or final:
            save_slide(self.fig)
        if not HEADLESS:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)


# ----------------------------------------------------------------------------
# 6. SLIDES AFTER TRAINING
# ----------------------------------------------------------------------------
def slide_after(dash):
    # (a) PCA of embeddings
    W = model.tok_emb.weight.detach().cpu().numpy()
    Wc = W - W.mean(0)
    U, S, _ = np.linalg.svd(Wc, full_matrices=False)
    pts = U[:, :2] * S[:2]
    fig = new_fig("What it learned: similar characters end up close together",
                  f"Each character is {N_EMBD} numbers. Squashed to 2D here. Nobody told the model what a vowel is; "
                  "it discovered the groups just by predicting the next letter.")
    ax = fig.add_axes([0.05, 0.05, 0.62, 0.78]); style(ax)
    for i, c in enumerate(chars):
        col = (GOOD if c in "aeiou" else ACC if c.islower() else PINK if c.isupper() else WARN)
        ax.text(pts[i, 0], pts[i, 1], vis(c), color=col, fontsize=16, ha="center", va="center", family="monospace")
    ax.set_xlim(pts[:, 0].min() * 1.15, pts[:, 0].max() * 1.15)
    ax.set_ylim(pts[:, 1].min() * 1.15, pts[:, 1].max() * 1.15)
    fig.text(0.72, 0.75,
             "green  = vowels\nblue   = other lowercase\npink   = CAPITALS\nyellow = punctuation, space, newline\n\n"
             "Look for: vowels clustering,\ncapitals grouped apart from\nlowercase, punctuation together.",
             color=FG, fontsize=14, family="monospace", va="top", linespacing=1.6)
    show(fig)

    # (b) attention heads
    _, atts = peek()
    fig = new_fig("What it learned: attention heads",
                  f"Each of the {N_HEAD} heads in a layer learns its own way of looking back. Rows = the character looking, "
                  "columns = where it looks. Bright = strong attention.")
    for row, (li, lname) in enumerate([(0, "first layer"), (N_LAYER - 1, "last layer")]):
        for h in range(N_HEAD):
            ax = fig.add_axes([0.04 + h * 0.24, 0.52 - row * 0.44, 0.21, 0.31]); style(ax, f"{lname}, head {h + 1}")
            T = len(EX_CONTEXT)
            ax.imshow(atts[li][h], cmap="magma", aspect="auto", vmin=0, vmax=0.6)
            ax.set_xticks(range(T)); ax.set_xticklabels([vis(c) for c in EX_CONTEXT], fontsize=5, color=DIM)
            ax.set_yticks(range(T)); ax.set_yticklabels([vis(c) for c in EX_CONTEXT], fontsize=5, color=DIM)
    show(fig)

    # (c) temperature
    fig = new_fig("Generation: the temperature knob",
                  "The model outputs probabilities. Temperature decides how adventurous the sampling is. "
                  "Low = safe and repetitive, high = creative but messy.")
    for i, tmp in enumerate([0.5, 0.8, 1.2]):
        g = sample("ROMEO:\n", 230, tmp)
        fig.text(0.04 + i * 0.32, 0.80, f"temperature = {tmp}", color=WARN, fontsize=15, weight="bold")
        fig.text(0.04 + i * 0.32, 0.76, "\n".join(textwrap.fill(l, 36) for l in g.split("\n")),
                 color=FG, fontsize=10, family="monospace", va="top", linespacing=1.4)
    show(fig)

    # (d) how to chat
    fig = new_fig("Now YOU talk to it", "")
    fig.text(0.04, 0.86,
             "IMPORTANT: this is NOT ChatGPT. It was never taught to answer questions, only to continue text.\n"
             "So you don't ask it things, you START a piece of text and it keeps writing.",
             color=FG, fontsize=15, va="top", linespacing=1.6)
    fig.text(0.04, 0.68,
             "Prompts to try (type \\n for a new line):\n\n"
             "  ROMEO:\\n                       \u2192 writes a speech in Romeo's name\n"
             "  KING HENRY:\\nMy lord,           \u2192 continues in royal style\n"
             "  First Citizen:\\nWe are          \u2192 street-crowd dialogue\n"
             "  To be, or not to be              \u2192 mangles a famous line\n"
             "  JULIET:\\nO Romeo, Romeo!        \u2192 a dramatic answer\n"
             "  hello how are you                \u2192 chaos: it never saw modern English",
             color=GOOD, fontsize=14, family="monospace", va="top", linespacing=1.4)
    fig.text(0.04, 0.27,
             "What to expect:\n"
             "  \u2022 Mostly real English words, a few invented ones (\"thou dost bewitch'd\")\n"
             "  \u2022 Correct play format: CAPITAL NAME + colon + lines of speech\n"
             "  \u2022 Sentences that sound Shakespearean but make no real sense\n"
             "  \u2022 Plenty of mistakes: it has only 1 MB of text and ~0.8M parameters\n\n"
             "Chat commands:   /temp 0.5   /len 500   /quit",
             color=FG, fontsize=14, va="top", linespacing=1.55)
    show(fig)


# ----------------------------------------------------------------------------
# 7. TRAIN
# ----------------------------------------------------------------------------
def train():
    slide_title(); slide_data(); slide_tokens(); slide_batches(); slide_architecture()
    slide_untrained(); slide_loop()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    dash = Dashboard()
    model.train()
    for step in range(STEPS + 1):
        warm = min(1.0, (step + 1) / 100)
        decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        lr = LR * warm * decay
        for g in opt.param_groups:
            g["lr"] = lr
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)                       # forward pass + loss
        opt.zero_grad(set_to_none=True)
        loss.backward()                               # backward pass: gradients
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()                                    # update: nudge every weight

        view = (step < 100 and step % 10 == 0) or step % 50 == 0 or step == STEPS
        if view:
            new_val = estimate_loss(5)["val"] if (step % 100 == 0 or step == STEPS) else None
            do_gen = step in (0, STEPS) or step % 250 == 0
            dash.update(step, loss.item(), gnorm, lr, new_val, do_gen, final=(step == STEPS))
            if step % 100 == 0:
                print(f"step {step:5d} | train {loss.item():.3f} | val {dash.val[-1] if dash.val else float('nan'):.3f}")
    if not HEADLESS:
        plt.ioff()
        dash.fig.text(0.97, 0.01, "training done: click or press a key \u25b6", color=WARN, fontsize=11, ha="right")
        dash.fig.canvas.draw()
        try:
            plt.waitforbuttonpress()
        except Exception:
            pass
    plt.close(dash.fig)
    torch.save({"model": model.state_dict(), "chars": chars}, "tiny_gpt.pt")
    print("Saved model to tiny_gpt.pt")
    slide_after(dash)
    pdf.close()
    print("Slides saved in ./presentation/ (PNG per slide + tiny_gpt_presentation.pdf)")


# ----------------------------------------------------------------------------
# 8. CHAT
# ----------------------------------------------------------------------------
def chat():
    print("\n" + "=" * 70)
    print(" CHAT WITH YOUR TINY GPT.  It continues text, it does not answer questions.")
    print(" Try:  ROMEO:\\n   |   KING HENRY:\\nMy lord,   |   First Citizen:\\nWe are")
    print(" Commands: /temp 0.5   /len 500   /quit      (type \\n for a new line)")
    print("=" * 70)
    temp, length = 0.8, 400
    while True:
        try:
            s = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            break
        s = s.strip()
        if s == "/quit":
            break
        if s.startswith("/temp"):
            try: temp = float(s.split()[1]); print(f"temperature = {temp}")
            except Exception: print("usage: /temp 0.8")
            continue
        if s.startswith("/len"):
            try: length = int(s.split()[1]); print(f"length = {length}")
            except Exception: print("usage: /len 400")
            continue
        s = s.replace("\\n", "\n")
        print("\n" + sample(s, length, temp))


if __name__ == "__main__":
    if args.chat_only:
        ck = torch.load("tiny_gpt.pt", map_location=DEVICE)
        model.load_state_dict(ck["model"])
        chat()
    else:
        print(f"device: {DEVICE} | parameters: {N_PARAMS:,} | steps: {STEPS}")
        train()
        if not args.no_chat and not args.headless:
            chat()