#!/usr/bin/env python3
"""Identifiability diagnostic and estimator-scale control.

Direction fidelity and MSE are algebraically equivalent for unit-normalized
reconstructions. This script demonstrates why pooled partial associations do not
identify an independent fidelity effect, then reports a direct estimator-scale
control:

  (1) per-bit Spearman rho(g,R@10) and rho(MSE,R@10) with bootstrap CIs (resample codecs);
  (2) the collinearity rho(g,MSE);
  (3) partial Spearman rho(g,R | MSE,bits) vs rho(MSE,R | g,bits), pooled, with a
      within-cell permutation p-value;
  (4) an estimator-axis control across EDEN, RaBitQ, and BlockQuant: same
      reconstruction, different estimator scale.

numpy only. Spearman = Pearson on ranks; partial = rank-residualization. Deterministic (seed 42).
"""
import re
from pathlib import Path
import numpy as np

RNG = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "generated" / "mechanism-inputs"

DIAG_RE = re.compile(
    r"\|\s*([a-z0-9_]+)\s*\|\s*(\d)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|"
    r"\s*[\d.]+\s*\|\s*([\d.]+)\s*\|\s*([-+][\d.]+)\s*\|\s*([\d.]+)")
RECALL_RE = re.compile(
    r"\|\s*([a-z0-9_]+)\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)")

def parse_diag(path):
    out = {}
    for m in DIAG_RE.finditer(open(path).read()):
        name, bits, g, sig, bias, mse = m.groups()
        out[name] = dict(g=float(g), sig=float(sig), bias=float(bias), mse=float(mse))
    return out

def parse_recall(path):
    out = {}
    for m in RECALL_RE.finditer(open(path).read()):
        name, bits, r1, r10, r100 = m.groups()
        if float(r10) <= 1.0 and name not in out:   # skip delta-section rows
            out[name] = float(r10)
    return out

def ranks(x):
    x = np.asarray(x, float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); start = cum - cnt
    return ((start + cum - 1) / 2.0)[inv]

def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def spearman(a, b): return pearson(ranks(a), ranks(b))

def resid_ranks(y, ctrls):
    """Residual of rank(y) regressed on [1, rank(ctrls...)]."""
    n = len(y)
    X = np.column_stack([np.ones(n)] + [ranks(c) for c in ctrls])
    beta, *_ = np.linalg.lstsq(X, ranks(y), rcond=None)
    return ranks(y) - X @ beta

def partial(a, b, ctrls):
    return pearson(resid_ranks(a, ctrls), resid_ranks(b, ctrls))

def normrank(x):
    """within-cell rank scaled to [0,1] (comparable across cells of different size)."""
    n = len(x)
    return ranks(x) / (n - 1) if n > 1 else np.zeros(n)

def pooled_partial(cells):
    """Pooled within-cell partial rank correlations over an arbitrary set of cells.

    Returns (N, rho(g,R@10|MSE), rho(MSE,R@10|g)) using the same within-cell
    normalized-rank pooling as section (2), so a subset is directly comparable to
    the full-pool figure.
    """
    Gr, Mr, Rr = [], [], []
    for cell in cells.values():
        Gr += list(normrank([c["g"] for c in cell]))
        Mr += list(normrank([c["mse"] for c in cell]))
        Rr += list(normrank([c["r10"] for c in cell]))
    Gr, Mr, Rr = map(np.array, (Gr, Mr, Rr))
    n = len(Rr)

    def presid(y, X):
        A = np.column_stack([np.ones(n)] + list(X))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta

    return (n,
            pearson(presid(Gr, [Mr]), presid(Rr, [Mr])),
            pearson(presid(Mr, [Gr]), presid(Rr, [Gr])))

def main():
    CORPORA = ["siftsmall", "sift", "glove", "gist"]
    cells = {}          # (corpus,bits) -> list of per-codec dicts
    per_corpus_bit = {} # for the per-bit table (siftsmall, the reference corpus)
    for cname in CORPORA:
        for b in [1, 2, 3, 4]:
            df = INPUT / f"diag-{cname}-b{b}.txt"
            rf = INPUT / f"recall-{cname}-b{b}.txt"
            if not (df.is_file() and rf.is_file()):
                raise FileNotFoundError(f"missing mechanism input: {df} or {rf}")
            d = parse_diag(df); r = parse_recall(rf)
            cell = [dict(corpus=cname, bits=b, codec=k, g=d[k]["g"], mse=d[k]["mse"],
                         sig=d[k]["sig"], r10=r[k]) for k in d if k in r]
            if len(cell) >= 4:
                cells[(cname, b)] = cell
                g = [c["g"] for c in cell]; mse = [c["mse"] for c in cell]; rr = [c["r10"] for c in cell]
                per_corpus_bit[(cname, b)] = dict(n=len(cell), rho_g=spearman(g, rr),
                    rho_mse=spearman(mse, rr), rho_gmse=spearman(g, mse))

    loaded = sorted({cn for cn, _ in cells})
    print("# Direction fidelity, MSE, and Recall@10\n")
    print(f"Corpora: {', '.join(loaded)} (IP/cosine, 100k subset where applicable, seed 42, the "
          f"oblivious 7-codec field per cell). {len(cells)} (corpus×bit) cells.\n")

    print("## (1) Per-corpus×bit associations and collinearity\n")
    print("| corpus | bits | n | ρ(g,R@10) | ρ(MSE,R@10) | ρ(g,MSE) |")
    print("|---|---|---|---|---|---|")
    for cn in CORPORA:
        for b in [1, 2, 3, 4]:
            if (cn, b) in per_corpus_bit:
                p = per_corpus_bit[(cn, b)]
                print(f"| {cn} | {b} | {p['n']} | {p['rho_g']:+.3f} | {p['rho_mse']:+.3f} | {p['rho_gmse']:+.3f} |")

    # ---- pooled cross-corpus separability via WITHIN-CELL normalized ranks ----
    # each (corpus,bits) cell is self-contained: rank codecs within it, scale to [0,1], pool.
    # this controls for both corpus and bit-rate (every cell contributes the same rank span).
    Gr, Mr, Rr, cidx = [], [], [], []
    for i, (_, cell) in enumerate(cells.items()):
        Gr += list(normrank([c["g"] for c in cell]))
        Mr += list(normrank([c["mse"] for c in cell]))
        Rr += list(normrank([c["r10"] for c in cell]))
        cidx += [i] * len(cell)
    Gr, Mr, Rr, cidx = map(np.array, (Gr, Mr, Rr, cidx))
    N = len(Rr)

    def presid(y, X):  # residual of y on [1,X...]
        A = np.column_stack([np.ones(N)] + list(X)); beta, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ beta
    rg_b  = pearson(Gr, Rr)
    rm_b  = pearson(Mr, Rr)
    # These residual associations are not independent effects for the normalized
    # families; the identity audit below shows which control supplies separation.
    rg_bm = pearson(presid(Gr, [Mr]), presid(Rr, [Mr]))
    rm_bg = pearson(presid(Mr, [Gr]), presid(Rr, [Gr]))
    coll  = pearson(Gr, Mr)
    print(f"\n## (2) Cross-corpus separability — pooled within-cell normalized ranks (N={N})\n")
    print(f"- ρ(g,   R@10)            = {rg_b:+.3f}   (within-cell, all corpora)")
    print(f"- ρ(MSE, R@10)            = {rm_b:+.3f}")
    print(f"- ρ(g,   R@10 | MSE)      = {rg_bm:+.3f}   (non-identifying diagnostic)")
    print(f"- ρ(MSE, R@10 | g)        = {rm_bg:+.3f}   (non-identifying diagnostic)")
    print(f"- ρ(g,   MSE)             = {coll:+.3f}   (collinearity)")
    # permutation: shuffle recall-ranks WITHIN each cell (preserves the cell structure)
    obs = abs(rg_bm) - abs(rm_bg)
    def shuffle_within(Y):
        out = Y.copy()
        for i in np.unique(cidx):
            m = cidx == i; out[m] = RNG.permutation(out[m])
        return out
    def stat(Y):
        return abs(pearson(presid(Gr, [Mr]), presid(Y, [Mr]))) - abs(pearson(presid(Mr, [Gr]), presid(Y, [Gr])))
    null = np.array([stat(shuffle_within(Rr)) for _ in range(20000)])
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    print(
        f"\nObserved |ρ(g·)|−|ρ(MSE·)| = {obs:+.3f};  "
        f"within-cell permutation p(g strictly more) = {p:.5f} "
        "(20,000 permutations; +1 correction; supplementary diagnostic only)"
    )

    # Leave-one-corpus-out sensitivity: does the partial association depend on any
    # single corpus? SIFTsmall is the smallest and the only prefix view, so it is
    # the obvious candidate for driving the pooled result.
    print("\n### Leave-one-corpus-out sensitivity of the partial association\n")
    print("| excluded corpus | cells | N | ρ(g,R@10\\|MSE) | ρ(MSE,R@10\\|g) |")
    print("|---|---|---|---|---|")
    print(f"| none | {len(cells)} | {N} | {rg_bm:+.3f} | {rm_bg:+.3f} |")
    for cn in CORPORA:
        sub = {k: v for k, v in cells.items() if k[0] != cn}
        if len(sub) < 2:
            continue
        n_sub, rg_sub, rm_sub = pooled_partial(sub)
        print(f"| {cn} | {len(sub)} | {n_sub} | {rg_sub:+.3f} | {rm_sub:+.3f} |")

    # For a unit-normalized reconstruction, MSE == 2(1-g) identically, so within a
    # cell rank(MSE) is the exact reverse of rank(g) and "controlling for MSE" is
    # vacuous for those codecs. Report which codecs break the identity and what the
    # partial association looks like without them. This identifies the effective
    # source of information behind the pooled separation.
    print("\n### Effective information behind the partial association\n")
    off_by_codec = {}
    for cell in cells.values():
        for c in cell:
            off_by_codec.setdefault(c["codec"], []).append(abs(c["mse"] - 2 * (1 - c["g"])))
    breakers = sorted(k for k, v in off_by_codec.items() if max(v) > 1e-3)
    n_obs = sum(len(v) for v in off_by_codec.values())
    n_break = sum(1 for v in off_by_codec.values() for x in v if x > 1e-3)
    print(f"- observations: {n_obs}; breaking MSE=2(1-g) by >1e-3: {n_break}")
    print(f"- identity-breaking codecs: {', '.join(breakers) if breakers else 'none'}")
    for codec in breakers:
        sub = {k: [c for c in v if c["codec"] != codec] for k, v in cells.items()}
        sub = {k: v for k, v in sub.items() if len(v) >= 4}
        n_sub, rg_sub, rm_sub = pooled_partial(sub)
        gr, mr = [], []
        for cell in sub.values():
            gr += list(normrank([c["g"] for c in cell]))
            mr += list(normrank([c["mse"] for c in cell]))
        print(f"- excluding {codec}: N={n_sub}, ρ(g,MSE)={pearson(np.array(gr), np.array(mr)):+.4f}, "
              f"ρ(g,R@10|MSE)={rg_sub:+.3f}, ρ(MSE,R@10|g)={rm_sub:+.3f}")

    print("\n## (3) Estimator-axis control (same codebook ⇒ same g+MSE, change scale)\n")
    unbiased = parse_control(
        INPUT / "control_unbiased.txt", INPUT / "control_unbiased_recall.txt"
    )
    biased = parse_control(INPUT / "control_biased.txt", INPUT / "control_biased_recall.txt")
    for codec in ("eden", "rabitq", "blockquant"):
        ub, bi = unbiased[codec], biased[codec]
        if abs(bi["g"] - ub["g"]) > 5e-4 or abs(bi["mse"] - ub["mse"]) > 5e-4:
            raise ValueError(f"{codec}: reconstruction changed across estimator control")
        print(f"- {codec}: unbiased g={ub['g']:.4f} MSE={ub['mse']:.4f} "
              f"bias={ub['bias']:+.4f} σ={ub['sig']:.4f} R@10={ub['r10']:.3f}; "
              f"biased bias={bi['bias']:+.4f} σ={bi['sig']:.4f} R@10={bi['r10']:.3f}; "
              f"ΔR@10={bi['r10']-ub['r10']:+.3f}")


def parse_control(diag_file, recall_file):
    diag = parse_diag(diag_file)
    recall = parse_recall(recall_file)
    required = {"eden", "rabitq", "blockquant"}
    missing = sorted(required - (set(diag) & set(recall)))
    if missing:
        raise ValueError(f"control rows not found: {', '.join(missing)}")
    return {
        codec: dict(
            g=diag[codec]["g"], sig=diag[codec]["sig"], bias=diag[codec]["bias"],
            mse=diag[codec]["mse"], r10=recall[codec],
        )
        for codec in required
    }

if __name__ == "__main__":
    main()
