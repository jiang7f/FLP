def plot_results(respath):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    data = pd.read_csv(f"../../data/{respath}.csv")
    scale = 0.9
    fig, axes = plt.subplots(figsize=(15*scale, 9*scale))
    mpl.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'Times New Roman',
        'font.size': 64,
        'axes.unicode_minus': False,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'axes.linewidth': 5,
        'xtick.major.size': 20,
        'xtick.major.width': 5,
    })
    
    axes = plt.axes([0, 0, 1, 1])
    colors = ["#7CBEEC", "#7EB57E", "#FF6F45", "#3274A1"]
    names = [
        "culled_depth",
    ]
    datamean = (
        # data[data.num_shared_variables.isin([6, 8, 10])]
        data
        .groupby(["num_shared_variables"])
        .mean()
        .reset_index()
        .sort_values(["num_shared_variables"])
    )
    for i, col in enumerate(names):
        axes.bar(
            datamean.index,
            datamean[col],
            color = colors[i],
            label = names[i],
            linewidth = 1,
            width = 0.5,
        )
    # axes.set_yscale("log")
    # axes.set_yticks([1, 100, 10000, 1000000])
    axes.tick_params(axis="x", which="major", width=5, length=20)
    axes.set_xticks(datamean.index)
    axes.set_xticklabels(datamean.num_shared_variables)
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=5)
    axes.set_xlabel("# num_shared_variables")
    axes.set_ylabel("depth")

    axes.legend(frameon=False, bbox_to_anchor=(0.5, 1.3), ncol=2)
    fig.savefig(f"../{respath}.svg", dpi=600, format="svg", bbox_inches="tight")

plot_results('SV/shared_linear_depth')