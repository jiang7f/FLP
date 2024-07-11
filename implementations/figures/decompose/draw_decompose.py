def plot_results(respath):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    data = pd.read_csv(f"../../data/{respath}.csv")
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
    
    axes = plt.axes([0, 0, 1, 1])
    colors = ["#7CBEEC", "#7EB57E", "#FF6F45", "#3274A1"]
    names = [
        "culled_depth", "depth"
    ]
    datamean = (
        data[data.pid.isin(range(13))]
        .groupby(["pid"])
        .mean()
        .reset_index()
        .sort_values(["pid"])
    )
    for i, col in enumerate(names):
        axes.bar(
            datamean.index + 0.2 * i,
            datamean[col],
            color = colors[i],
            label = names[i],
            linewidth = 4,
            width = 0.2,
            edgecolor='black',
        )
    prob_lab = [prob + str(idx) for idx in range(1, 6) for prob in ["FLP", "GCP", "KPP"]]
    # axes.set_yscale("log")
    # axes.set_yticks([1, 100, 10000, 1000000])
    axes.tick_params(axis="x", which="major", width=5, length=20)
    axes.set_xticks(datamean.index + 0.1)
    
    axes.set_xticklabels([prob_lab[idx] for idx in datamean.pid])
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=1)
    axes.set_xlabel("# pid", fontsize=56)
    axes.set_ylabel("depth", fontsize=56)

    axes.legend(frameon=False, bbox_to_anchor=(0.5, 1.3), ncol=2)
    fig.savefig(f"../{respath}.svg", dpi=600, format="svg", bbox_inches="tight")

plot_results('decompose/process_test')