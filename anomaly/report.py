"""Eksport wyników: tabele LaTeX (booktabs, przecinek dziesiętny) i wykresy (PDF + PNG)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

DISPLAY = {"logreg": "Regresja logistyczna", "random_forest": "Random Forest",
           "xgboost": "XGBoost", "isolation_forest": "Isolation Forest", "lof": "LOF", "ocsvm": "One-Class SVM",
           "autoencoder": "Autoenkoder"}
KIND_PL = {"supervised": "nadzorowany", "unsupervised": "nienadzorowany"}
STYLES = [("black", "-"), ("0.35", "--"), ("0.5", "-."), ("0.65", ":"), ("0.2", (0, (5, 1))),
          ("0.45", (0, (3, 1, 1, 1))), ("0.3", (0, (1, 1))), ("0.55", (0, (8, 2)))]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def dn(name):
    return DISPLAY.get(name, name.replace("_", " "))


def num(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace(".", "{,}")


def pm(mean, std, d=3):
    return f"{num(mean, d)} $\\pm$ {num(std, d)}"


def ci(lo, hi, d=3):
    return f"[{num(lo, d)}; {num(hi, d)}]"


def latex_table(header, rows, caption, label, align=None, note=None):
    align = align or "l" + "r" * (len(header) - 1)
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["\\par\\vspace{4pt}", f"{{\\footnotesize {note}}}"]
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- tabele
def tab_split(df):
    rows = [[r.split, f"{int(r.n):,}".replace(",", "\\,"), int(r.n_fraud), num(r.fraud_pct, 3) + "\\%"]
            for r in df.itertuples()]
    return latex_table(["Zbiór", "Liczba transakcji", "Oszustwa", "Udział oszustw"], rows,
                       "Podział danych w czasie i liczba oszustw w poszczególnych zbiorach.", "tab:podzial")


def tab_main(summary):
    rows = [[dn(r.model), KIND_PL[r.kind]] + [pm(getattr(r, f"{k}_mean"), getattr(r, f"{k}_std"))
            for k in ("pr_auc", "roc_auc", "f1", "precision", "recall")] for r in summary.itertuples()]
    return latex_table(["Model", "Rodzaj", "PR-AUC", "ROC AUC", "F1", "Precision", "Recall"], rows,
                       "Wyniki modeli na zbiorze testowym (podział czasowy), średnia $\\pm$ odchylenie standardowe z powtórzeń.",
                       "tab:wyniki-glowne", align="llccccc",
                       note="Próg decyzyjny wybrano na zbiorze walidacyjnym (maksymalizacja $F_1$). Modele deterministyczne mają zerowe odchylenie.")


def tab_ci(summary):
    rows = [[dn(r.model)] + [f"{num(getattr(r, k + '_seed0'))} {ci(getattr(r, k + '_lo'), getattr(r, k + '_hi'))}"
            for k in ("pr_auc", "roc_auc", "f1", "recall")] for r in summary.itertuples()]
    return latex_table(["Model", "PR-AUC", "ROC AUC", "F1", "Recall"], rows,
                       "Wartości na zbiorze testowym z 95\\% przedziałami ufności (bootstrap).",
                       "tab:przedzialy", align="lcccc",
                       note="Przedziały wyznaczono metodą percentylową z próbkowania ze zwracaniem zbioru testowego; "
                            "przy małej liczbie oszustw są szerokie.")


def tab_niu(df):
    rows = [[dn(r.model), KIND_PL[r.kind], pm(r.auroc_mean, r.auroc_std), pm(r.pr_auc_mean, r.pr_auc_std),
             num(r.auroc_niu) if r.auroc_niu == r.auroc_niu else "--"] for r in df.itertuples()]
    return latex_table(["Model", "Rodzaj", "AUROC (własne)", "PR-AUC (własne)", "AUROC wg \\cite{niu2019}"], rows,
                       "Ustawienie zbliżone do pracy \\cite{niu2019}: losowa walidacja krzyżowa, niedopróbkowanie klasy większościowej dla modeli nadzorowanych.",
                       "tab:niu", align="llccc",
                       note="Liczby z \\cite{niu2019} dotyczą ich ustawienia eksperymentu i są podane orientacyjnie.")


def tab_pairs(df):
    rows = [[f"{dn(r.model_a)} vs {dn(r.model_b)}", num(r.delta), ci(r.lo, r.hi), (f"$<${num(2 / r.n_boot, 3)}" if r.p_boot == 0 else num(r.p_boot, 3)),
             "tak" if r.significant else "nie"] for r in df.itertuples()]
    return latex_table(["Para modeli (A vs B)", "$\\Delta$ PR-AUC", "95\\% przedział", "$p$ (bootstrap)", "Istotna"], rows,
                       "Różnice PR-AUC między parami modeli na zbiorze testowym (bootstrap parami).",
                       "tab:pary", align="lcccc",
                       note="$\\Delta$ = PR-AUC(A) $-$ PR-AUC(B). W każdej próbie bootstrapowej losowano te same transakcje dla obu modeli. "
                            "Wartość $p$ jest orientacyjna (dwustronna, z rozkładu bootstrapowego); „istotna'' oznacza, że przedział nie zawiera zera. "
                            "Uwzględniono tylko zmienność zbioru testowego, nie zmienność wynikającą z uczenia.")


def tab_sweep(df):
    rows = []
    for name, g in df.groupby("model", sort=False):
        for r in g.itertuples():
            if r.domyslna or r.wybrana:
                lab = " i ".join(x for x, f in (("domyślna", r.domyslna), ("wybrana na walidacji", r.wybrana)) if f)
                rows.append([dn(name), lab, r.config.replace("_", "\\_").replace("{", "").replace("}", "").replace('"', ""),
                             num(r.val_pr_auc), num(r.test_pr_auc), num(r.test_roc_auc)])
    return latex_table(["Model", "Konfiguracja", "Parametry", "PR-AUC (walid.)", "PR-AUC (test)", "ROC AUC (test)"], rows,
                       "Strojenie modeli nienadzorowanych: konfiguracja domyślna i wybrana na zbiorze walidacyjnym.",
                       "tab:strojenie", align="lllccc",
                       note="Wybór wg PR-AUC na walidacji (56 oszustw, więc wybór jest niepewny); zbiór testowy nie brał udziału w wyborze. "
                            "\\texttt{log\\_amount}: $\\log(1+\\text{Amount})$ i standaryzacja zamiast \\texttt{RobustScaler}. Jeden seed.")


def tab_imb(df):
    rows = [[dn(r.model), r.variant] + [pm(getattr(r, f"{k}_mean"), getattr(r, f"{k}_std"))
            for k in ("pr_auc", "f1", "precision", "recall")] for r in df.itertuples()]
    return latex_table(["Model", "Sposób", "PR-AUC", "F1", "Precision", "Recall"], rows,
                       "Wpływ sposobu radzenia sobie z niezbalansowaniem klas (zbiór testowy, podział czasowy).",
                       "tab:niezbalansowanie", align="llcccc")


def tab_windows(df, counts):
    rows = [[dn(r.model), pm(r.pr_auc_mean, r.pr_auc_std), pm(r.roc_auc_mean, r.roc_auc_std),
             pm(r.f1_mean, r.f1_std)] for r in df.itertuples()]
    cnt = ", ".join(str(int(c)) for c in counts)
    return latex_table(["Model", "PR-AUC", "ROC AUC", "F1"], rows,
                       "Wyniki w kolejnych oknach czasowych (ocena z przesuwanym początkiem), średnia $\\pm$ odchylenie.",
                       "tab:okna", align="lccc", note=f"Liczba oszustw w kolejnych oknach testowych: {cnt}.")


# ---------------------------------------------------------------- wykresy
def save_fig(fig, base):
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=200)
    plt.close(fig)


def plot_curves(scores, kind, base):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for i, (name, (y, s)) in enumerate(scores.items()):
        c, ls = STYLES[i % len(STYLES)]
        if kind == "pr":
            p, r, _ = precision_recall_curve(y, s)
            ax.plot(r, p, color=c, ls=ls, label=dn(name))
        else:
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=c, ls=ls, label=dn(name))
    if kind == "pr":
        y0 = next(iter(scores.values()))[0]
        ax.axhline(float(np.mean(y0)), color="0.6", lw=0.8, ls=":", label="poziom losowy")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    else:
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls=":", label="poziom losowy")
        ax.set(xlabel="Odsetek fałszywych alarmów (FPR)", ylabel="Czułość (TPR)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left" if kind == "pr" else "lower right", framealpha=0.9)
    save_fig(fig, base)


def plot_ci(summary, base):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    names = [dn(m) for m in summary.model]
    y = summary.pr_auc_seed0.values
    err = np.vstack([y - summary.pr_auc_lo.values, summary.pr_auc_hi.values - y])
    ax.errorbar(range(len(y)), y, yerr=np.clip(err, 0, None), fmt="o", color="black", capsize=4)
    ax.set_xticks(range(len(y)), names, rotation=20, ha="right")
    ax.set(ylabel="PR-AUC (95% przedział ufności)", ylim=(0, 1.02))
    save_fig(fig, base)


def plot_confusion(cms, base):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(np.log1p(m), cmap="Greys", vmin=0)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:,}".replace(",", " "), ha="center", va="center",
                    color="white" if np.log1p(v) > np.log1p(m).max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["norma", "oszustwo"], fontsize=8)
        ax.set_yticks([0, 1], ["norma", "oszustwo"], fontsize=8, rotation=90, va="center")
        ax.set_title(dn(name), fontsize=9)
        ax.set_xlabel("przewidywana", fontsize=8)
        ax.grid(False)
    axes[0][0].set_ylabel("rzeczywista", fontsize=8)
    save_fig(fig, base)


def plot_imbalance(df, base):
    models = list(dict.fromkeys(df.model))
    variants = list(dict.fromkeys(df.variant))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    w = 0.8 / len(variants)
    hatches = ["", "//", "xx", ".."]
    for j, v in enumerate(variants):
        d = df[df.variant == v].set_index("model").reindex(models)
        ax.bar(np.arange(len(models)) + j * w, d.pr_auc_mean, w, yerr=d.pr_auc_std, capsize=2,
               color=str(0.25 + 0.2 * j), edgecolor="black", hatch=hatches[j % 4], label=v)
    ax.set_xticks(np.arange(len(models)) + 0.4 - w / 2, [dn(m) for m in models])
    ax.set(ylabel="PR-AUC", ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    save_fig(fig, base)


def plot_niu(df, base):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(df))
    ax.bar(x, df.auroc_mean, 0.55, yerr=df.auroc_std, capsize=3, color="0.6", edgecolor="black", label="AUROC (własne, CV)")
    ref = df.auroc_niu.values
    m = ~np.isnan(ref)
    ax.scatter(x[m], ref[m], marker="D", color="black", zorder=3, label="AUROC wg [2]")
    ax.set_xticks(x, [dn(n) for n in df.model], rotation=20, ha="right")
    ax.set(ylabel="AUROC", ylim=(0.5, 1.01))
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, base)


def plot_windows(long, base):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, name in enumerate(dict.fromkeys(long.model)):
        d = long[long.model == name]
        c, ls = STYLES[i % len(STYLES)]
        ax.plot(d.window, d.pr_auc, color=c, ls=ls, marker="o", ms=4, label=dn(name))
    ax.set_xticks(sorted(long.window.unique()))
    ax.set(xlabel="Okno testowe (kolejne w czasie)", ylabel="PR-AUC", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    save_fig(fig, base)


def plot_pairs(df, base):
    fig, ax = plt.subplots(figsize=(5.8, max(2.5, 0.28 * len(df) + 1.0)))
    y = np.arange(len(df))[::-1]
    lab = [f"{dn(a)} vs {dn(b)}" for a, b in zip(df.model_a, df.model_b)]
    for yi, r in zip(y, df.itertuples()):
        ax.plot([r.lo, r.hi], [yi, yi], color="black", lw=1.2 if r.significant else 0.8, alpha=1 if r.significant else 0.5)
        ax.plot(r.delta, yi, "o" if r.significant else "s", color="black" if r.significant else "white",
                markeredgecolor="black", ms=4)
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(y, lab, fontsize=7)
    ax.set(xlabel="Różnica PR-AUC (A $-$ B), 95% przedział")
    ax.grid(axis="y", alpha=0.0)
    save_fig(fig, base)
