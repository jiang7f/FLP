import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_results(respath):
    data = pd.read_csv(f"../../data/{respath}.csv")
    
    # Convert columns with 'memory_error' to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    
    data_false = data[data['use_decompose'] == False]
    data_true = data[data['use_decompose'] == True]

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
    bar_width =  0.27
    bar_key = "transpile_time"
    axes = plt.axes([0, 0, 1, 1])
    colors = ["#7CBEEC", "#7EB57E", "#FF6F45", "#3274A1"]
    
    datamean_false = (
        data_false.groupby("pid")
        .mean()
        .reset_index()
        .sort_values("pid")
    )
    datamean_true = (
        data_true.groupby("pid")
        .mean()
        .reset_index()
        .sort_values("pid")
    )
    
    axes.bar(
        datamean_false.index - bar_width / 2,
        datamean_false[bar_key],
        color=colors[0],
        label="use_decompose=False",
        linewidth=4,
        width=bar_width,
        edgecolor='black',
    )
    axes.bar(
        datamean_true.index + bar_width / 2,
        datamean_true[bar_key],
        color=colors[1],
        label="use_decompose=True",
        linewidth=4,
        width=bar_width,
        edgecolor='black',
    )
    axes.set_yscale("log")
    # axes.set_yticks([1, 100, 10000, 1000000])
    prob_lab = [prob + str(idx) for prob in ["F", "G", "K"] for idx in range(1, 6)]
    axes.tick_params(axis="x", which="major", width=5, length=20)
    axes.set_xticks(datamean_false.index)
    axes.set_xticklabels([prob_lab[idx] for idx in datamean_false.pid])
    axes.grid(axis="y", color="gray", linestyle="--", linewidth=1)
    axes.set_xlabel("# problem")
    axes.set_ylabel(bar_key)
    
    axes.legend(frameon=False, bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=2)
    fig.savefig(f"../{respath}.svg", dpi=600, format="svg", bbox_inches="tight")

plot_results('decompose/mp_evaluate')
