import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import StrMethodFormatter

mpl.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": "Times New Roman",
    }
)
plt.rcParams["text.usetex"] = True

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"]
# })
names = ["RPM", "VTC", "FS",
         "FS (W+O)", "VTC", "FS (W+O+I)", "FS (W+L)", "VTC"]
delays = [100, 9.67, 0.93]
hatch_pattern = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
colors = [
    "#90ee90",
    "#add8e6",
    "#ffb6c1",
    "#e6e6fa",
    "#d3d3d3",
    "#f08080",
    "#ffffe0",
    "#87cefa",
    "#d3d3d3",
]

fig = plt.figure(figsize=(6, 5))

plt.bar(
    names[0],
    delays[0],
    capsize=10,
    label=names[0],
    hatch=hatch_pattern[0],
    color=colors[0],
    alpha=0.99,
)
plt.bar(
    names[1],
    delays[1],
    capsize=10,
    label=names[1],
    hatch=hatch_pattern[1],
    color=colors[1],
    alpha=0.99,
)
plt.bar(
    names[2],
    delays[2],
    capsize=10,
    label=names[2],
    hatch=hatch_pattern[2],
    color=colors[2],
    alpha=0.99,
)

plt.ylabel("Delayed users (%)", fontsize=20)
# plt.xlabel("Scheduler", fontsize=20)
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))  # Scientific notation on y-axis
plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
# plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
fig.tight_layout()

# plt.savefig("saved_token_rpmvsfs.pgf", bbox_inches="tight")
plt.savefig("delayed_users.pdf", bbox_inches="tight", dpi=300)
# plt.show()
plt.close(fig)
