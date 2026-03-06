# We'll create two grouped column charts (bar charts), one for 30k and one for 7k.
# Each chart has 7 scene groups with 3 bars per group (N, avg, max). We only use matplotlib and avoid any custom colors.

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# Input data from the provided table
scenes = ["Bicycle", "Bonsai", "Counter", "Garden", "Kitchen", "Room", "Stump"]

data_30k = {
    "scene": scenes,
    "Gaussian #":     [5835584, 1211190, 1152507, 5006467, 1736173, 1516054, 4433445],
    "Avg Intersect #":   [8816976, 4313038, 7002919, 6971412, 8344250, 6472639, 4510527],
    "Max Intersect #":   [11173474, 5327574, 8699429, 8162134, 10382160, 8682429, 6847800],
}

data_7k = {
    "scene": scenes,
    "Gaussian #":     [3515515, 1141386, 978464, 3991956, 1591981, 1075775, 3512186],
    "Avg Intersect #":   [5724447, 3393182, 6030820, 5895578, 7693952, 4633338, 3581100],
    "Max Intersect #":   [7394387, 4126817, 7517942, 6846258, 9558733, 6423845, 5198985],
}

df_30k = pd.DataFrame(data_30k)
df_7k = pd.DataFrame(data_7k)

def plot_grouped(ax, df, title, ylabel=True):
    metrics = ["Gaussian #", "Avg Intersect #", "Max Intersect #"]
    x = np.arange(len(df["scene"]))  
    width = 0.22  
    colors = ["#fe300b", "#acb2c6", "#63668f"]
    
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[m].values, width, label=m if title=="7k" else "", color=colors[i])
    
    ax.set_title(title)
    ax.set_xlabel("Scene")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scene"], rotation=0)
    if ylabel:
        ax.set_ylabel("Value")
    else:
        ax.set_ylabel("")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

# Create side-by-side plots (30k left, 7k right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

plot_grouped(ax1, df_7k, "7k", ylabel=True)
plot_grouped(ax2, df_30k, "30k", ylabel=False)

# Single legend at the top center
fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))

fig.tight_layout(rect=[0,0,1,0.95])  # leave space for legend
fig.savefig("/mnt/ccnas2/bdp/lg524/column_chart_combined.png", dpi=200, bbox_inches="tight")
plt.show()