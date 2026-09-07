# Installation

Everything in this artifact runs in **one** Python interpreter. The lattice
attack calls SageMath and the reconstruction stage calls PyTorch inside the same
process, so they cannot live in separate environments.

Tested on macOS 14/15 (Apple silicon and Intel) and Ubuntu 22.04, with
SageMath 10.x, Python 3.11 and PyTorch 2.x. CPU only — no GPU is required for
any result in the paper.

---

## Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate hssp-dfl
python -m pip install -r requirements.txt
```

`environment.yml` pulls SageMath from conda-forge, which is by far the least
painful way to get a working LLL/BKZ implementation.

## Option B — an existing SageMath installation

If Sage is already installed, install the remaining packages **into Sage's own
Python**, not the system one:

```bash
sage -python -m pip install -r requirements.txt
```

Then run every command in this artifact with `sage -python` instead of `python`:

```bash
sage -python reproduce.py table8
```

## Option C — no SageMath

The topology experiments need nothing but NumPy, NetworkX, pandas and
Matplotlib, and run in a plain environment:

```bash
python -m pip install numpy scipy networkx pandas matplotlib
python reproduce.py figure2 figure6 figure7 figure8
```

Everything else — Tables 1, 3, 5, 6, 7–11 and Figures 3, 4, 5, 9, 10, 11, 12–17
— requires Sage.

---

## Verifying the environment

```bash
python -c "import sage.all; print('sage ok')"
python -c "import torch, torchvision; print('torch', torch.__version__)"
python -c "import networkx, scipy, pandas, matplotlib; print('scientific stack ok')"
python -c "import vec2text, rouge_score, bert_score; print('text stack ok')"
python -m hssp_dfl.paths          # shows which assets are present
```

Then run the cheapest end-to-end result and diff it against the paper:

```bash
python reproduce.py table8
python scripts/check_reference.py table8     # must print PASS
```

---

## Common problems

**`FeatureNotPresentError: singular is not available`**
Sage's helper binaries are not on `PATH`. Activate the environment rather than
calling the interpreter by absolute path:

```bash
conda activate hssp-dfl && python ...        # not  /path/to/envs/hssp-dfl/bin/python ...
# or, non-interactively:
conda run -n hssp-dfl python ...
```

**`ModuleNotFoundError: No module named 'torch'` inside a Sage script**
PyTorch went into a different interpreter. Reinstall with
`sage -python -m pip install torch torchvision`.

**`FileNotFoundError: Missing t0.5 checkpoint`**
The real-dataset experiments need the assets. Run
`bash scripts/setup_assets.sh --check` to see what is missing, then either link
them in with `--from`, or train them (README §5).

**Step 2 times out on the 20- or 40-node configurations**
BKZ occasionally takes much longer than the 60 s default. Raise it:
`--timeout 300`. This affects only Tables 1 and 9, whose recall percentages are
averages over 100 topologies.

**Matplotlib opens a window or fails on a headless machine**
All plotting scripts already force the `Agg` backend. If a stray script does
not, set `MPLBACKEND=Agg`.

---

## Optional: the Breaching dependency (Table 6 only)

Table 6 evaluates three published gradient-inversion attacks through the
[Breaching](https://github.com/JonasGeiping/breaching) framework, pinned to a
specific commit:

```bash
python -m pip install -r experiments/gia_transfer/requirements.txt
```

Nothing else in the artifact depends on it. See
[`experiments/gia_transfer/README.md`](experiments/gia_transfer/README.md) for
the threat model, the per-attack iteration budgets and the resumable stages.

---

## Optional: an embedding-API key (Figure 10 and Tables 12-14 only)

`experiments/vec2text_eval.py` inverts recovered ada-002 embeddings back to text
by re-embedding its own hypotheses through a hosted embedding API, so it reads
`OPENAI_API_KEY` from the environment. It is the only script in the artifact
that reaches outside the machine.

Everything upstream of it is fully offline, including the lattice attack itself
and the embedding MSE and cosine similarity it is scored on. Section 7 of the
README shows how to build Figure 10 and Tables 12-14 from the published metrics
if you do not want to re-run the inversion.
