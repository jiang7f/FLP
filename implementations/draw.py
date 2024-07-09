def plot_results(respath):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    data = pd.read_csv(f"{respath}")
    # datamean = data.groupby(["num_shared_variables"]).mean().reset_index().sort_values(["num_shared_variables"])
    # datamean.reset_index(drop=True, inplace=True)
    scale = 0.009
    fig = plt.figure(figsize=(1500 * scale, 900 * scale))
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = 62
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = "Times New Roman"
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    # set axes linewidth
    mpl.rcParams["axes.linewidth"] = 5
    ## set ticks linewidth
    mpl.rcParams["xtick.major.size"] = 20
    mpl.rcParams["xtick.major.width"] = 5
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
            color=colors[i],
            label=names[i],
            linewidth=1,
            width=0.5,
        )
    # axes.set_yscale("log")
    axes.tick_params(axis="x", which="major", width=5, length=20)
    # axes.set_yticks([1, 100, 10000, 1000000])
    axes.set_xlabel("# num_shared_variables")
    axes.set_xticks(datamean.index[::2])
    axes.set_xticklabels(datamean.num_shared_variables[::2])
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=5)
    axes.set_ylabel("depth")

    axes.legend(frameon=False, bbox_to_anchor=(0.5, 1.3), ncol=2)
    fig.savefig(f"{respath}.svg", dpi=600, format="svg", bbox_inches="tight")

plot_results('data/shared_constant_depth.csv')