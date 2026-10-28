"""Figures for the latent-space analysis, from results/analysis/stats.json -> report/figures/*.pdf.

  python analysis/plots.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATS = json.loads((ROOT / "results" / "analysis" / "stats.json").read_text())
OUT = ROOT / "report" / "figures"

# Validated categorical slots 1-3 (all pairs pass CVD and normal-vision floors); fixed per model.
COLOR = {"Clinical UltraBERT": "#2a78d6", "UltraBERT": "#eb6834", "BioClinical ModernBERT": "#1baf7a"}
MODELS = list(COLOR)
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"

# The project's Google fonts (fonts/): Product Sans for all figure text.
for ttf in (ROOT / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

plt.rcParams.update({
    "font.family": ["Product Sans", "Google Sans Code"], "axes.titleweight": "normal", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def geometry():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    bins = np.linspace(-0.2, 1.0, 61)
    for m in MODELS:
        g = STATS[m]["geometry"]
        a.hist(g["random_cos_sample"], bins=bins, histtype="step", linewidth=1.5, color=COLOR[m],
               label=f"{m} (mean {g['random_cos_mean']:.2f})")
        cv = g["cum_var"]
        b.plot(np.arange(1, len(cv) + 1), cv, color=COLOR[m], linewidth=1.5,
               label=f"{m} (eff. rank {g['effective_rank']:.0f})")
    a.set(xlabel="Cosine similarity of random document pairs", ylabel="Count",
          title="(a) Anisotropy: random pairs, SciFact corpus")
    a.set_ylim(0, a.get_ylim()[1] * 1.55)  # headroom so the legend clears the tall spike
    a.legend(loc="upper left")
    b.set(xlabel="Principal components", ylabel="Cumulative variance explained",
          title="(b) Spectrum of the centered embeddings", ylim=(0, 1.02))
    b.legend(loc="lower right")
    save(fig, "fig_geometry")


def separation():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), sharey=True)
    bins = np.linspace(-0.2, 1.0, 49)
    for ax, m in zip(axes, MODELS):
        r = STATS[m]["retrieval"]["SciFact"]
        ax.hist(r["rand_cos"], bins=bins, density=True, color=NEUTRAL, alpha=0.55, label="random document")
        ax.hist(r["rel_cos"], bins=bins, density=True, histtype="step", linewidth=1.5, color=COLOR[m],
                label="relevant document")
        ax.set(title=f"{m}\nd' = {r['d_prime']:.2f}", xlabel="Query–document cosine")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper left")
    save(fig, "fig_separation")


def lexical():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    edges = [0, 0.25, 0.5, 0.75, 1.01]
    labels = ["0–25%", "25–50%", "50–75%", "75–100%"]
    for m in MODELS:
        # NFCorpus is left out: its 12K loosely graded pairs would swamp the other two tasks.
        pairs = [p for t in ("SciFact", "MedicalQARetrieval") for p in STATS[m]["retrieval"][t]["pairs"]]
        ov = np.array([p["overlap"] for p in pairs])
        hit = np.array([p["hit10"] for p in pairs])
        rates = [hit[(ov >= lo) & (ov < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(range(4), rates, color=COLOR[m], linewidth=1.5, marker="o", markersize=4, label=m)
    counts = np.histogram([p["overlap"] for p in pairs], bins=edges)[0]
    ax.set_xticks(range(4), [f"{l}\n(n={c})" for l, c in zip(labels, counts)])
    ax.set(xlabel="Query content words found in the relevant document", ylabel="Relevant doc. in top 10",
           ylim=(0, 1), title="Recall@10 by lexical overlap\n(SciFact and MedicalQA)")
    ax.legend(loc="upper left")
    save(fig, "fig_lexical")


def whitening():
    w = json.loads((ROOT / "results" / "analysis" / "whitening.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for ax, task, title in zip(axes, ("SciFact", "MedicalQARetrieval"), ("SciFact", "MedicalQA")):
        for i, m in enumerate(MODELS):
            raw, white = w[m][task]["raw"], w[m][task]["whitened"]
            ax.bar(i - 0.19, raw, width=0.36, color=COLOR[m], label="raw cosine" if i == 0 else None)
            ax.bar(i + 0.19, white, width=0.36, color="white", edgecolor=COLOR[m], linewidth=1.2, hatch="////",
                   label="whitened" if i == 0 else None)
            for x, v in ((i - 0.19, raw), (i + 0.19, white)):
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(range(3), ["Clinical\nUltraBERT", "UltraBERT", "BioClinical\nModernBERT"])
        ax.set(title=title, ylim=(0, 1))
    axes[0].set_ylabel("Recall@10")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="////")]
    axes[1].legend(handles, ["raw cosine", "PCA-whitened (256 dims)"], loc="upper right")
    save(fig, "fig_whitening")


def tsne():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), gridspec_kw={"wspace": 0.15})
    classes = ["neoplasms", "digestive system diseases", "cardiovascular diseases"]
    class_color = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3, here encoding class, not model
    for ax, m in zip(axes, MODELS):
        t = STATS[m]["topics"]["MedicalAbstracts"]
        xy, lab = np.array(t["tsne"]["xy"]), np.array(t["tsne"]["label"])
        for c, col in zip(classes, class_color):
            k = lab == c
            ax.scatter(xy[k, 0], xy[k, 1], s=5, color=col, linewidths=0.2, edgecolors="white", label=c)
        ax.set(title=f"{m}\nsil. {t['silhouette']:.3f}, kNN acc. {t['knn10_acc']:.2f}", xticks=[], yticks=[])
        ax.grid(False)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncol=3, markerscale=2.5, handletextpad=0.2)
    save(fig, "fig_tsne")


def role_topic():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    groups = ["Same abstract,\ndifferent role\n(topic)", "Different abstract,\nsame role\n(rhetorical role)"]
    keys = ["topic_effect_z", "role_effect_z"]
    w = 0.26
    for i, m in enumerate(MODELS):
        vals = [STATS[m]["role_topic"][k] for k in keys]
        bars = ax.bar(np.arange(2) + (i - 1) * w, vals, width=w - 0.03, color=COLOR[m], label=m)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(2), groups)
    ax.set(ylabel="Similarity above unrelated pairs (SD)", title="What drives sentence similarity?\n(PubMed RCT, 250 abstracts)")
    ax.legend(loc="upper right")
    save(fig, "fig_role_topic")




def _legend(fig, y=-0.1):
    handles = [plt.Line2D([], [], color=COLOR[m], linewidth=2) for m in MODELS]
    fig.legend(handles, MODELS, loc="upper center", bbox_to_anchor=(0.5, y), ncol=3)


def fig5():
    """Geometry: anisotropy of the frozen encoders and the whitening control."""
    w = json.loads((ROOT / "results" / "analysis" / "whitening.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"wspace": 0.3})

    bins = np.linspace(-0.2, 1.0, 61)
    for m in MODELS:
        g = STATS[m]["geometry"]
        a.hist(g["random_cos_sample"], bins=bins, histtype="step", linewidth=1.4, color=COLOR[m], density=True)
        peak = np.histogram(g["random_cos_sample"], bins=bins, density=True)[0].max()
        a.text(g["random_cos_mean"], peak * 1.04 + 0.4, f"mean {g['random_cos_mean']:.2f}", ha="center",
               fontsize=6.8, color=INK)
    a.set(xlabel="Cosine similarity of random SciFact abstract pairs", ylabel="Density",
          title="Anisotropy of the embedding space")
    a.set_ylim(0, a.get_ylim()[1] * 1.12)

    xs = {"SciFact": 0, "MedicalQARetrieval": 1}
    for i, m in enumerate(MODELS):
        for task, x0 in xs.items():
            x = x0 * 3.6 + i
            raw, white = w[m][task]["raw"], w[m][task]["whitened"]
            b.bar(x - 0.19, raw, width=0.36, color=COLOR[m])
            b.bar(x + 0.19, white, width=0.36, color="white", edgecolor=COLOR[m], linewidth=1.1, hatch="////")
            for xx, v in ((x - 0.19, raw), (x + 0.19, white)):
                b.text(xx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    b.set_xticks([1, 4.6], ["SciFact", "MedicalQA"])
    b.set(ylabel="Recall@10", ylim=(0, 1.08), title="Whitening control")
    b.grid(axis="x", visible=False)
    b.legend([plt.Rectangle((0, 0), 1, 1, color=MUTED),
              plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="////")],
             ["raw cosine", "PCA-whitened"], loc="upper center", bbox_to_anchor=(0.5, 1.0))
    for ax, letter in zip((a, b), "ab"):
        ax.text(-0.14, 1.12, letter, transform=ax.transAxes, fontsize=11, va="top")
    _legend(fig, -0.12)
    save(fig, "fig5_geometry")


def fig6():
    """What fine-tuning taught: retrieval beyond word overlap and medical synonymy."""
    fig, (c, d) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"wspace": 0.3})
    edges = [0, 0.25, 0.5, 0.75, 1.01]
    for m in MODELS:
        pairs = [p for t in ("SciFact", "MedicalQARetrieval") for p in STATS[m]["retrieval"][t]["pairs"]]
        ov, hit = np.array([p["overlap"] for p in pairs]), np.array([p["hit10"] for p in pairs])
        rates = [hit[(ov >= lo) & (ov < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
        c.plot(range(4), rates, color=COLOR[m], linewidth=1.5, marker="o", markersize=4)
    counts = np.histogram(ov, bins=edges)[0]
    c.set_xticks(range(4), [f"{l}\n(n = {n})" for l, n in zip(["0–25%", "25–50%", "50–75%", "75–100%"], counts)])
    c.set(xlabel="Query content words found in the relevant document", ylabel="Recall@10", ylim=(0, 1),
          title="Recall by lexical overlap")

    kinds = ["abbreviation", "lay term", "brand name"]
    for i, m in enumerate(MODELS):
        vals = [STATS[m]["synonyms"][k]["acc@1"] for k in kinds]
        bars = d.bar(np.arange(3) + (i - 1) * 0.27, vals, width=0.25, color=COLOR[m])
        for bar, v in zip(bars, vals):
            d.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                   fontsize=6.2, color=INK)
    d.set_xticks(range(3), ["Abbreviation\n(e.g. DKA)", "Lay term\n(e.g. nosebleed)",
                            "Brand name\n(e.g. Lasix)"])
    d.set(ylabel="Top-1 accuracy", ylim=(0, 1), title="Medical synonym matching")
    d.grid(axis="x", visible=False)
    for ax, letter in zip((c, d), "ab"):
        ax.text(-0.14, 1.12, letter, transform=ax.transAxes, fontsize=11, va="top")
    _legend(fig, -0.2)
    save(fig, "fig6_learned")


if __name__ == "__main__":
    for f in (geometry, separation, lexical, whitening, tsne, role_topic, fig5, fig6):
        f()
        print(f"wrote {f.__name__}")
