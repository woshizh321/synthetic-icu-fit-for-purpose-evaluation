from pathlib import Path
import argparse
import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap, BoundaryNorm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced_r2_figures"
FIG = OUT / "FIGURES"
QC = OUT / "QC"

E1 = ROOT / "r2_source_data"
V2SRC = ROOT / "r2_source_data"
E35 = ROOT / "r2_source_data"

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

BLUE = "#0072B2"
ORANGE = "#D55E00"
PINK = "#CC79A7"
BLACK = "#333333"
SKY = "#56B4E9"
GOLD = "#E69F00"
TEAL = "#009E73"
GREY = "#8A8A8A"
LIGHT = "#F2F4F7"
STATE_COLORS = {"Q": TEAL, "I": GOLD, "NQ": "#6A5ACD"}
STRAT_COLORS = {"M1": BLUE, "M2": PINK, "M3": ORANGE, "M4": BLACK}
STRAT_MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D"}
DGP_ORDER = ["G", "T", "S"]
DGP_LABEL = {"G": "Gaussian", "T": r"Student $t_5$", "S": "Skew-normal"}
STRAT_NAME = {
    "M1_FULL_COV_CROSSED": "M1",
    "M2_DIAGONAL_COV_CROSSED": "M2",
    "M3_REALIZATION_AGGREGATED": "M3",
    "M4_SINGLE_REALIZATION": "M4",
}


def panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=9,
            va="bottom", ha="left")


def finish(fig, number, *, left=0.08, right=0.98, top=0.94, bottom=0.08):
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    pdf = FIG / f"Figure_{number}.pdf"
    png = FIG / f"Figure_{number}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def rounded_box(ax, xy, w, h, title, body, edge=GREY, face="white", lw=1.1):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.012,rounding_size=0.015",
                           linewidth=lw, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontweight="bold", color=edge if edge != GREY else BLACK, fontsize=8)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=7)
    return patch


def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                linewidth=1.0, color=color))


def figure1():
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.suptitle("Future-use suitability is a decision about a new realization at a future hospital",
                 fontsize=12, y=0.99)

    ax.text(0.01, 0.92, "a", fontweight="bold", fontsize=10)
    ax.text(0.045, 0.92, "Future-use question", fontweight="bold", fontsize=9)
    rounded_box(ax, (0.04, 0.77), 0.18, 0.11, "Source clinical cohort", "Train one generator\nunder a fixed specification")
    rounded_box(ax, (0.29, 0.77), 0.18, 0.11, "Stochastic synthesis", "New utility-estimable\nrealization")
    rounded_box(ax, (0.54, 0.77), 0.18, 0.11, "Eligible hospitals", "Defined exchangeable\ntarget population")
    rounded_box(ax, (0.79, 0.77), 0.18, 0.11, "Future destination", "New eligible hospital")
    arrow(ax, (0.22, 0.825), (0.29, 0.825)); arrow(ax, (0.72, 0.825), (0.79, 0.825))
    rounded_box(ax, (0.34, 0.61), 0.32, 0.10, "Primary Target B",
                "Latent AUROC utility loss for a new realization\nat a future eligible hospital", edge=BLUE, face="#EAF4FA")
    arrow(ax, (0.38, 0.77), (0.44, 0.71), BLUE); arrow(ax, (0.88, 0.77), (0.56, 0.71), BLUE)

    ax.text(0.01, 0.55, "b", fontweight="bold", fontsize=10)
    ax.text(0.045, 0.55, "Three actions carry different consequences", fontweight="bold", fontsize=9)
    rounded_box(ax, (0.05, 0.41), 0.25, 0.10, "QUALIFIED", "False qualification can expose\na future use that exceeds tolerance", edge=TEAL, face="#E8F5EF")
    rounded_box(ax, (0.375, 0.41), 0.25, 0.10, "INDETERMINATE", "Abstention defers a decision\nand can delay a proposed use", edge=GOLD, face="#FFF4D6")
    rounded_box(ax, (0.70, 0.41), 0.25, 0.10, "NOT QUALIFIED", "False rejection can discard\na use that meets tolerance", edge="#6A5ACD", face="#F0EEFA")

    ax.text(0.01, 0.35, "c", fontweight="bold", fontsize=10)
    ax.text(0.045, 0.35, "Evaluation strategies represent uncertainty differently", fontweight="bold", fontsize=9)
    xs = [0.04, 0.285, 0.53, 0.775]
    titles = ["M4", "M3", "M2", "M1"]
    bodies = ["One realization\nacross hospitals", "Repeated realizations\naggregated first",
              "Crossed realization–hospital\nmodel; diagonal covariance", "Same crossed model;\nfull covariance"]
    edges = [BLACK, ORANGE, PINK, BLUE]
    for x, t, b, c in zip(xs, titles, bodies, edges):
        rounded_box(ax, (x, 0.21), 0.18, 0.09, t, b, edge=c)
    for i in range(3): arrow(ax, (xs[i]+0.18, 0.255), (xs[i+1], 0.255))
    ax.text(0.50, 0.185, "M1–M2 is the matched crossed-model covariance comparison; other links compare broader strategies.",
            ha="center", va="top", fontsize=6.8, color=BLACK)

    ax.text(0.01, 0.13, "d", fontweight="bold", fontsize=10)
    ax.text(0.045, 0.13, "Connected evidence sequence", fontweight="bold", fontsize=9)
    seq = ["Empirical decision\ninstability", "Known-truth\ntrade-offs", "Cost-sensitive\nconsequences", "Generator-mixture\nsensitivity"]
    sx = [0.05, 0.29, 0.53, 0.77]
    for x, t in zip(sx, seq): rounded_box(ax, (x, 0.025), 0.18, 0.065, t, "", edge=GREY, face=LIGHT)
    for i in range(3): arrow(ax, (sx[i]+0.18, 0.057), (sx[i+1], 0.057))
    return finish(fig, 1)


def setting_labels(df):
    gen_order = ["GaussianCopula", "CTGAN", "TabDDPM"]
    learner_order = ["LR_L2", "XGBoost"]
    delta_order = [0.025, 0.05, 0.1, 0.2]
    return [(g, l, d) for g in gen_order for l in learner_order for d in delta_order]


def short_setting(g, l, d):
    gn = {"GaussianCopula": "GaussianCopula", "CTGAN": "CTGAN", "TabDDPM": "TabDDPM"}[g]
    ln = "LR" if l == "LR_L2" else "XGB"
    return f"{gn} / {ln} / δ={d:g}"


def decision_matrix(df, settings, col, values):
    mapping = {v: i for i, v in enumerate(values)}
    arr = np.full((len(settings), len(col)), np.nan)
    for i, (g, l, d) in enumerate(settings):
        for j, c in enumerate(col):
            q = df[(df.generator == g) & (df.learner == l) & (np.isclose(df.delta, d))]
            key = "method" if "method" in df.columns and c in set(df.get("method", [])) else "hospital_population"
            q = q[q[key] == c]
            if len(q): arr[i, j] = mapping[q.iloc[0].decision]
    return arr


def figure2():
    au = pd.read_csv(E1 / "E1A_REAL_TRAINED_BASELINE_AUROC.csv")
    m4 = pd.read_csv(E1 / "E1D_M4_DECISION_COUNTS.csv")
    mm = pd.read_csv(E1 / "E1C_EMPIRICAL_M1_M2_M3_P95_DECISIONS.csv")
    hp = pd.read_csv(E1 / "E1E_H100_H200_H300_M1_TARGETB_DECISIONS.csv")
    settings = setting_labels(mm)

    fig = plt.figure(figsize=(7.2, 9.7))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.15, 1.35], hspace=0.53, wspace=0.48)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1:, 0])
    ax4 = fig.add_subplot(gs[1:, 1])
    fig.suptitle("Empirical utility context and suitability-decision variability", fontsize=12, y=0.99)

    pooled = au[au.analysis_level == "POOLED_OBSERVATION"]
    destinations = ["HELD_OUT_MIMIC_IV", "SICDB", "POOLED_EICU"]
    labels = ["Held-out\nMIMIC-IV", "SICdb", "Pooled\neICU"]
    x = np.arange(3)
    for learner, color, marker, off in [("LR_L2", BLUE, "o", -0.06), ("XGBoost", ORANGE, "s", 0.06)]:
        q = pooled[pooled.learner == learner].set_index("destination").loc[destinations]
        ax1.scatter(x + off, q.real_trained_AUROC, s=30, color=color, marker=marker,
                    label="Logistic regression" if learner == "LR_L2" else "XGBoost")
    h = au[au.destination == "EICU_H200"]
    for learner, color, off in [("LR_L2", BLUE, -0.06), ("XGBoost", ORANGE, 0.06)]:
        r = h[h.learner == learner].iloc[0]
        ax1.errorbar(3 + off, r.hospital_AUROC_median,
                     yerr=[[r.hospital_AUROC_median-r.hospital_AUROC_q25],
                           [r.hospital_AUROC_q75-r.hospital_AUROC_median]],
                     fmt="o" if learner == "LR_L2" else "s", color=color, capsize=2)
    ax1.set_xticks(np.arange(4), labels + ["H200 hospital\nmedian [IQR]"])
    ax1.set_ylabel("Real-trained AUROC")
    ax1.set_ylim(0.74, 0.89)
    ax1.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax1.legend(frameon=False, loc="lower left")
    panel_label(ax1, "a")

    y = np.arange(24)
    left = np.zeros(24)
    for state, col in [("NQ", STATE_COLORS["NQ"]), ("I", STATE_COLORS["I"]), ("Q", STATE_COLORS["Q"])]:
        vals = []
        for g, l, d in settings:
            q = m4[(m4.generator == g) & (m4.learner == l) & np.isclose(m4.delta, d)].iloc[0]
            vals.append(q[f"{state}_proportion"])
        ax2.barh(y, vals, left=left, color=col, height=0.72, label=state)
        left += np.array(vals)
    mixed = ((m4[["Q_n", "I_n", "NQ_n"]] > 0).sum(axis=1) > 1).sum()
    ax2.set_yticks([]); ax2.set_xlim(0, 1)
    ax2.set_xlabel("Proportion of observed realizations")
    ax2.set_title(f"M4: {mixed}/24 settings had >1 decision state", fontsize=8)
    ax2.legend(frameon=False, ncol=3, loc="upper right")
    ax2.invert_yaxis(); panel_label(ax2, "b")

    cmap = ListedColormap([STATE_COLORS["NQ"], STATE_COLORS["I"], STATE_COLORS["Q"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    arr = decision_matrix(mm, settings, ["M1", "M2", "M3"], ["NQ", "I", "Q"])
    ax3.imshow(arr, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
    ax3.set_xticks(range(3), ["M1", "M2", "M3"])
    ax3.set_yticks(range(24), [short_setting(*s) for s in settings])
    disagree = sum(len(set(row[~np.isnan(row)])) > 1 for row in arr)
    ax3.set_title(f"Strategy comparison: {disagree}/24 settings disagreed", fontsize=8)
    ax3.set_xlabel("Evaluation strategy")
    panel_label(ax3, "c")

    arr2 = decision_matrix(hp, settings, ["H100", "H200", "H300"], ["NQ", "I", "Q"])
    ax4.imshow(arr2, aspect="auto", cmap=cmap, norm=norm, interpolation="none")
    ax4.set_xticks(range(3), ["H100\n127 hospitals", "H200\n85 hospitals", "H300\n66 hospitals"])
    ax4.set_yticks(range(24), [])
    changed = [i for i, row in enumerate(arr2) if len(set(row[~np.isnan(row)])) > 1]
    for i in changed:
        ax4.add_patch(plt.Rectangle((-0.5, i-0.5), 3, 1, fill=False, edgecolor="black", lw=1.5))
    ax4.set_title(f"Hospital population: {len(changed)}/24 settings changed", fontsize=8)
    ax4.set_xlabel("Eligible-hospital population")
    panel_label(ax4, "d")
    handles = [mpl.lines.Line2D([0],[0], marker="s", color="none", markerfacecolor=STATE_COLORS[s],
                               markeredgecolor="none", markersize=7, label=s) for s in ["Q","I","NQ"]]
    ax4.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.text(0.5, 0.012, "Empirical comparisons have no known-truth oracle and show variability, not decision error.",
             ha="center", fontsize=7)
    return finish(fig, 2, left=0.25, right=0.98, top=0.94, bottom=0.065)


def forest(ax, df, metric, title, show_y=True):
    q = df[df.metric == metric].copy()
    q["order"] = q.dgp.map({"G":0,"T":1,"S":2}) * 4 + q.delta.map({.025:0,.05:1,.1:2,.2:3})
    q = q.sort_values("order")
    y = np.arange(len(q))
    for _, r in q.iterrows():
        yy = y[q.index.get_loc(r.name)]
        ax.errorbar(r.paired_difference, yy,
                    xerr=[[r.paired_difference-r.mc_lower],[r.mc_upper-r.paired_difference]],
                    fmt="o", color=BLUE, markersize=3, capsize=1.5, lw=0.9)
    ax.axvline(0, color=GREY, ls="--", lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.invert_yaxis(); ax.grid(axis="x", color="#E5E5E5", lw=0.5)
    if show_y:
        ax.set_yticks(y, [f"{DGP_LABEL[r.dgp]}  δ={r.delta:g}" for _, r in q.iterrows()])
    else:
        ax.set_yticks(y, [])


def figure3():
    prof = pd.read_csv(V2SRC / "V2_FIGURE2_SOURCE_DATA.csv")
    con = pd.read_csv(V2SRC / "V2_FIGURE3_SOURCE_DATA.csv")
    prof["short"] = prof.strategy_short
    fig = plt.figure(figsize=(7.2, 8.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1.4], hspace=0.48, wspace=0.38)
    fig.suptitle("Known-truth evaluation reveals error–decisiveness trade-offs", fontsize=12, y=0.99)

    for j, dgp in enumerate(DGP_ORDER):
        ax = fig.add_subplot(gs[0, j])
        q = prof[prof.dgp == dgp]
        for s in ["M1","M2","M3","M4"]:
            z = q[q.short == s]
            ax.plot(z.DY, z.DER, color=STRAT_COLORS[s], alpha=0.35, lw=0.7)
            for _, r in z.iterrows():
                ax.scatter(r.DY, r.DER, s=26 + 12*np.log10(r.delta/.025+1),
                           marker=STRAT_MARKERS[s], facecolor="white", edgecolor=STRAT_COLORS[s], lw=1.2)
        ax.set_title(DGP_LABEL[dgp], fontweight="bold")
        ax.set_xlim(0.25, 1.0); ax.set_ylim(-0.01, 0.31)
        ax.set_xlabel("Definitive yield")
        if j == 0: ax.set_ylabel("Decision-error rate")
        ax.grid(color="#E5E5E5", lw=0.5)
        panel_label(ax, chr(ord('a')+j))
    handles = [mpl.lines.Line2D([0],[0], marker=STRAT_MARKERS[s], markerfacecolor="white",
                               markeredgecolor=STRAT_COLORS[s], color="none", label=s) for s in STRAT_COLORS]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.925))

    q = con[(con.policy == "P95") & (con.contrast == "M1_vs_M2_full_covariance")]
    metrics = [("DER", "ΔDER"), ("DY", "ΔDY"), ("CDY", "ΔCDY")]
    for j, (m, title) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, j])
        forest(ax, q, m, title, show_y=(j==0))
        ax.set_xlabel("M1 − M2")
        panel_label(ax, chr(ord('d')+j))
        if m == "DER":
            ax.text(0.02, 0.01, "M1 lower in 9; higher in 3", transform=ax.transAxes, fontsize=6.5)
        else:
            ax.text(0.02, 0.01, "M1 higher in all 12", transform=ax.transAxes, fontsize=6.5)
    fig.text(0.5, 0.015,
             "Points and whiskers are paired differences and 95% Monte Carlo intervals. M1–M2 is evaluated under matched sampling covariance; positive values mean a higher named metric.",
             ha="center", fontsize=6.8)
    return finish(fig, 3, top=0.88)


def figure4():
    cov = pd.read_csv(V2SRC / "V2_FIGURE4_SOURCE_DATA.csv")
    cov["M"] = cov.strategy.map(STRAT_NAME)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.4), gridspec_kw={"width_ratios":[1.35,1.0,1.15]})
    fig.suptitle("Coverage depends on whether the strategy represents the future-use target", fontsize=12, y=0.99)
    rng = np.random.default_rng(20261030)
    for j, target in enumerate(["B", "A"]):
        ax = axes[j]
        for i, s in enumerate(["M1","M2","M3","M4"]):
            q = cov[(cov.target == target) & (cov.M == s)]
            jitter = rng.uniform(-0.18, 0.18, len(q))
            ax.scatter(np.full(len(q), i)+jitter, q.coverage, s=9, alpha=0.55,
                       color=STRAT_COLORS[s], edgecolor="none")
            med = q.coverage.median()
            ax.plot([i-0.22, i+0.22], [med, med], color="black", lw=1.2)
        for y, ls, col in [(0.90, ":", GREY), (0.95, "--", BLACK), (0.99, ":", GREY)]:
            ax.axhline(y, color=col, ls=ls, lw=0.8)
        ax.set_xticks(range(4), ["M1","M2","M3","M4"])
        ax.set_ylim(0.48, 1.01)
        ax.set_title("Primary Target B" if target == "B" else "Secondary Target A", fontweight="bold")
        ax.set_xlabel("Strategy")
        if j == 0: ax.set_ylabel("Nominal 0.95 interval coverage")
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        panel_label(ax, chr(ord('a')+j))

    ax = axes[2]
    rows=[]
    for target in ["B","A"]:
        for s in ["M1","M2","M3","M4"]:
            q=cov[(cov.target==target)&(cov.M==s)]
            rows.append((target,s,int((q.coverage<.90).sum()),int((q.coverage>.99).sum())))
    yy=np.arange(8)
    low=[r[2] for r in rows]; high=[r[3] for r in rows]
    ax.barh(yy-0.17, low, height=.32, color="#6A5ACD", label="Coverage <0.90")
    ax.barh(yy+0.17, high, height=.32, color=GOLD, label="Coverage >0.99")
    ax.set_yticks(yy,[f"Target {t}  {s}" for t,s,_,_ in rows], fontsize=6.4)
    ax.invert_yaxis(); ax.set_xlabel("Cells (of 72)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#EEEEEE", lw=.5)
    panel_label(ax, "c")
    fig.text(0.5, 0.02,
             "All 576 distribution × template × regime × strategy × target cells are shown. Thresholds at 0.90 and 0.99 are descriptive; the dashed line marks nominal 0.95 coverage.",
             ha="center", fontsize=6.8)
    return finish(fig, 4, left=0.08, right=0.98, top=0.84, bottom=0.17)


def figure5():
    p = pd.read_csv(E35 / "E3_PAIRED_STRATEGY_CONTRASTS.csv")
    r = pd.read_csv(E35 / "E3_CONDITIONAL_REGRET_RESULTS.csv")
    e5 = pd.read_csv(E35 / "E5_LEAVE_ONE_GENERATOR_OUT_RESULTS.csv")
    mask = ((p.dgp=="S") & (p.template_id=="TabDDPM__XGBoost") &
            (p.regime=="R1_SEED_DOMINANT") & np.isclose(p.delta,.2) &
            (p.contrast=="M2_vs_M3_crossed_structure"))
    q=p[mask].copy()
    maskr = ((r.dgp=="S") & (r.template_id=="TabDDPM__XGBoost") &
             (r.regime=="R1_SEED_DOMINANT") & np.isclose(r.delta,.2))
    rr=r[maskr].copy()

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.33)
    fig.suptitle("Decision costs and generator composition change strategy comparisons", fontsize=12, y=0.99)

    ax=fig.add_subplot(gs[0,0])
    for a, z in q.groupby("a"):
        z=z.sort_values("b")
        ax.plot(z.b, z.paired_difference, marker="o", ms=3.5, lw=1.1, label=f"a={a:g}")
    ax.axhline(0,color=BLACK,ls="--",lw=.8)
    ax.set_xlabel("Indeterminate-decision cost b")
    ax.set_ylabel("M2 − M3 mean regret")
    ax.set_title("One fixed truth scenario across the full cost grid", fontsize=8)
    ax.legend(frameon=False,ncol=2)
    ax.grid(color="#EEEEEE",lw=.5)
    ax.text(.02,.02,r"Skew-normal; TabDDPM–XGBoost; R1; δ=0.20; $p^*=0.277341$",
            transform=ax.transAxes,fontsize=6.4)
    panel_label(ax,"a")

    ax=fig.add_subplot(gs[0,1])
    settings=[(.25,.10),(.25,.20),(4,.40)]
    labels=["a=.25\nb=.10","a=.25\nb=.20","a=4\nb=.40"]
    offsets=np.linspace(-.24,.24,4)
    for k,s in enumerate(["M1_FULL_COV_CROSSED","M2_DIAGONAL_COV_CROSSED","M3_REALIZATION_AGGREGATED","M4_SINGLE_REALIZATION"]):
        zz=[]
        for a,b in settings:
            zz.append(rr[(rr.strategy==s)&np.isclose(rr.a,a)&np.isclose(rr.b,b)].iloc[0])
        x=np.arange(3)+offsets[k]
        y=np.array([z.mean_regret for z in zz])
        lo=np.array([z.mc_lower for z in zz]); hi=np.array([z.mc_upper for z in zz])
        short=STRAT_NAME[s]
        ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=STRAT_MARKERS[short],color=STRAT_COLORS[short],
                    ms=4,capsize=2,lw=.9,label=short)
    ax.set_xticks(range(3),labels)
    ax.set_ylabel("Mean regret")
    ax.set_title("Strategy regret at three illustrative costs",fontsize=8)
    ax.legend(frameon=False,ncol=4,loc="upper center")
    ax.grid(axis="y",color="#EEEEEE",lw=.5)
    ax.text(.02,.77,"At a=.25, b=.20, M2 and M3 have deterministic zero regret\non all 995 common-success fits.",
            transform=ax.transAxes,fontsize=6.2,
            bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.85, "pad":1.5})
    panel_label(ax,"b")

    def e5_panel(ax, contrast, title, label):
        z=e5[(e5.excluded_generator=="CTGAN") & np.isclose(e5.delta,.05) &
             (e5.metric=="DER") & (e5.contrast==contrast)].copy()
        z["ord"]=z.dgp.map({"G":0,"T":1,"S":2}); z=z.sort_values("ord")
        y=np.arange(3)
        for i,(_,row) in enumerate(z.iterrows()):
            ax.plot([row.original_paired_difference,row.excluded_paired_difference],[i,i],color="#BBBBBB",lw=1)
            ax.errorbar(row.original_paired_difference,i,xerr=[[row.original_paired_difference-row.original_mc_lower],[row.original_mc_upper-row.original_paired_difference]],fmt="o",color=BLACK,ms=4,capsize=2,label="All generators" if i==0 else None)
            ax.errorbar(row.excluded_paired_difference,i,xerr=[[row.excluded_paired_difference-row.excluded_mc_lower],[row.excluded_mc_upper-row.excluded_paired_difference]],fmt="s",color=ORANGE,ms=4,capsize=2,label="After CTGAN exclusion" if i==0 else None)
        ax.axvline(0,color=GREY,ls="--",lw=.8)
        ax.set_yticks(y,[DGP_LABEL[d] for d in z.dgp]); ax.invert_yaxis()
        ax.set_xlabel("Paired DER difference")
        ax.set_title(title,fontsize=8)
        ax.legend(frameon=False,loc="best")
        ax.grid(axis="x",color="#EEEEEE",lw=.5)
        panel_label(ax,label)

    e5_panel(fig.add_subplot(gs[1,0]),"M2_vs_M3_crossed_structure","M2 − M3 after CTGAN exclusion","c")
    e5_panel(fig.add_subplot(gs[1,1]),"M1_vs_M2_full_covariance","Exploratory M1 − M2 reversal pattern","d")
    fig.text(0.5,0.015,
             "E3 costs are methodological and scenario-conditional. E5 contrasts are related, unadjusted comparisons; exclusion changes the benchmark mixture and does not identify a causal CTGAN effect.",
             ha="center",fontsize=6.8)
    return finish(fig, 5, bottom=0.075)


def main():
    global OUT, FIG, QC
    parser = argparse.ArgumentParser(description="Reproduce the five R2 figures from released aggregate inputs.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    FIG = OUT / "FIGURES"
    QC = OUT / "QC"
    FIG.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    outputs=[]
    for fn in [figure1, figure2, figure3, figure4, figure5]:
        pdf,png=fn(); outputs.extend([str(pdf),str(png)])
    manifest=[]
    import hashlib
    for path in outputs:
        b=Path(path).read_bytes()
        manifest.append({"path":str(Path(path).relative_to(OUT)),"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)})
    (QC/"FIGURE_OUTPUT_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\n".join(outputs))


if __name__ == "__main__":
    main()
