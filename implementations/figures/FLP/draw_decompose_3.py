import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_results(respath):
    data = pd.read_csv(f"../../data/{respath}.csv")
    penalty = data[data['method'] == 'penalty']
    cyclic = data[data['method'] == 'cyclic']
    commute = data[data['method'] == 'commute']
    HEA = data[data['method'] == 'HEA']
    # Convert columns with 'memory_error' to NaN
    penalty = penalty.apply(pd.to_numeric, errors='coerce').fillna(0)
    cyclic = cyclic.apply(pd.to_numeric, errors='coerce').fillna(0)
    commute = commute.apply(pd.to_numeric, errors='coerce').fillna(0)
    HEA = HEA.apply(pd.to_numeric, errors='coerce').fillna(0)

    scale = 0.9
    fig, axes = plt.subplots(figsize=(25*scale, 9*scale))

    mpl.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'Times New Roman',
        'font.size': 56,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'axes.linewidth': 5,
        'xtick.major.size': 20,
        'xtick.major.width': 10,
    })
    bar_width =  0.2
    bar_key = "best_solution_probs"
    axes = plt.axes([0, 0, 1, 1])
    colors = ["#7CBEEC", "#7EB57E", "#FF6F45", "#3274A1"]
    # print(penalty)
    mean_penalty = (
        penalty.groupby("variables")
        .mean()
        .reset_index()
        .sort_values("variables")
    )
    mean_cyclic = (
        cyclic.groupby("variables")
        .mean()
        .reset_index()
        .sort_values("variables")
    )
    mean_commute = (
        commute.groupby("variables")
        .mean()
        .reset_index()
        .sort_values("variables")
    )
    mean_HEA = (
        HEA.groupby("variables")
        .mean()
        .reset_index()
        .sort_values("variables")
    )

    # print(mean_HEA, mean_commute, mean_cyclic)
    axes.bar(
        mean_penalty.index - bar_width * 3/ 2,
        mean_penalty[bar_key],
        color=colors[0],
        label="penalty",
        linewidth=4,
        width=bar_width,
        edgecolor='black',
    )
    axes.bar(
        mean_cyclic.index - bar_width / 2,
        mean_cyclic[bar_key],
        color=colors[1],
        label="cyclic",
        linewidth=4,
        width=bar_width,
        edgecolor='black',
    )
    axes.bar(
        mean_commute.index + bar_width / 2,
        mean_commute[bar_key],
        color=colors[2],
        label="commute",
        linewidth=4,
        width=bar_width,
        edgecolor='black',
    )
    axes.bar(
        mean_HEA.index + bar_width * 3 / 2,
        mean_HEA[bar_key],
        color=colors[3],
        label="HEA",
        linewidth=4,
        width=bar_width,
        edgecolor='black',
    )

    # axes.set_yscale("log")
    # axes.set_yticks([1, 100, 10000, 1000000])
    # prob_lab = [prob + str(idx) for prob in ["F", "G", "K"] for idx in range(1, 6)]
    axes.tick_params(axis="x", which="major", width=5, length=20)
    axes.set_xticks(mean_commute.index)
    # axes.set_xticklabels([prob_lab[idx] for idx in mean_commute.variables])
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=1)
    axes.set_xlabel("# problem")
    axes.set_ylabel(bar_key)
    axes.legend(frameon=False, bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=4)
    script_path = os.path.abspath(__file__)
    new_path = script_path[:-3]
    fig.savefig(f"{new_path}.svg", dpi=600, format="svg", bbox_inches="tight")

plot_results('FLP/FLP_evalue_qiskit')
