#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def single_table(labels, data_list, rect_cnt, rect_label, width, ylabel, title, scale = 1.0, numbering = False):
    plt.rcParams.update({'font.size': int(14 * scale), 'text.latex.preamble': [r'\usepackage{amsmath}'], 'text.usetex': True})
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    hatches = ['', '//', '\\\\', 'xx', '...', '//', '*', '\\\\', '---', '\\\\', '//', '', '...', '////', '\\\\\\', 'xxxx', '....', '//', '*', '\\\\', '---', '\\\\', '//', '', '...']
    x = np.arange(len(labels))
    #plt.xlabel("Coreset Size", fontsize=10)
    fig, ax = plt.subplots()
    plt.xlabel("Coreset Size", fontsize=16)
    fig.canvas.set_window_title(title)
    rects = [ ax.bar(x = x - width / 2 + width / rect_cnt / 2 + i * width / rect_cnt,
    height=data_list[i], edgecolor='black',
    width=width / rect_cnt, label = rect_label[i], hatch = hatches[i]) for i in range(rect_cnt) ]
    #if data_list_upper != None:
    #    rects.append([ ax.bar(x = x - width / 2 + width / rect_cnt / 2 + i * width / rect_cnt,
    #    height=data_list_upper[i], edgecolor='black',
    #    width=width / rect_cnt, label = rect_label_upper[i], hatch = hatches[rect_cnt + i],
    #    bottom=data_list[i]) for i in range(rect_cnt)])


    ax.set_ylabel(ylabel, fontsize=16)
    #ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()

    if numbering:
        for rect in rects:
            autolabel(ax, rect)

    fig.tight_layout()

def parse_runtime(path):
    coreset_sizes = []
    dijkstra_times = []
    vxs_times = []
    vxs_costs = []
    sxs_costs = []
    sxs_times = []
    for line in filter(lambda x: not x.startswith("#"), open(path)):
        toks = line.strip().split()
        coreset_sizes.append(int(toks[0]))
        dijkstra_times.append(int(toks[2]))
        vxs_times.append(int(toks[3]))
        vxs_costs.append(float(toks[4]))
        sxs_times.append(int(toks[5]))
        sxs_costs.append(float(toks[6]))
    return list(map(np.array, (coreset_sizes, dijkstra_times, vxs_times, vxs_costs, sxs_times, sxs_costs)))



BENCHDATA_PATH="brooklyn.coreset.25.runtime"


full_opt_cost = 177247664
thorup_opt_cost = 179024784

thorup_sampling_time = 17083
distmatgen = 48983 + 2977
lsearch_thorup = 443716
cost_time = 201
sampler_gen = 60
total_cs_construction_time = sum((thorup_sampling_time, distmatgen, lsearch_thorup, cost_time, sampler_gen))
full_distmat_time = 551612
full_lsearch = 41321203
total_full_time = full_lsearch + full_distmat_time

coreset_sizes, dijkstra_times, vxs_times, vxs_costs, sxs_times, sxs_costs = parse_runtime(BENCHDATA_PATH)
#vxs_times = np.log2(vxs_times)
#sxs_times = np.log2(sxs_times)

plt.rcParams["mathtext.fontset"] = "cm"
plt.style.use('tableau-colorblind10')


rect_label = [r'$T_{X \times V}$', r'$T_{cs}$', r'$T_{D \times V}$', r'$T_{D\times D}$']
labels = np.array(list(map(str, coreset_sizes)))
construction_times = np.array([total_cs_construction_time] * len(labels))
full_times = np.array([total_full_time] * len(labels))
Y = 'Runtime (ms)'
selection = np.array([3,5,7,9,11], dtype=np.uint32).astype(int)
if __name__ == "__main__":
    slab = labels[selection]
    single_table(labels[selection],
                 [full_times[selection], construction_times[selection], vxs_times[selection], sxs_times[selection]], 4, rect_label, 0.6, Y, "runtime")
    plt.savefig("fig4.png", dpi=600)
    plt.savefig("fig4.svg")
    plt.clf()
Y = 'Base-2 Logarithmic Runtime (ms)'
full_times = np.log2(full_times)
construction_times = np.log2(construction_times)
vxs_times = np.log2(vxs_times)
sxs_times = np.log2(sxs_times)
if __name__ == "__main__":
    single_table(labels[selection],
                 [full_times[selection], construction_times[selection], vxs_times[selection], sxs_times[selection]], 4, rect_label, 0.6, Y, "runtime")
    #single_table(labels, [full_times, construction_times, vxs_times, sxs_times], 4, rect_label, 0.6, Y, "runtime")
    plt.savefig("fig4.log2.png", dpi=600)
    plt.savefig("fig4.log2.svg")
    plt.clf()

# newlabels = ["Full", "Thorup"] + labels
Y = r"Objective Value ($10^8$ m)"

if __name__ == "__main__":
    sxs_costs *= 1e-8
    vxs_costs *= 1e-8
    #thorup_opt_cost *= 1e-8
    full_opt_cost *= 1e-8
    plt.plot(labels, sxs_costs.T, linewidth=1.5, linestyle='--')
    plt.plot(labels, vxs_costs.T, linewidth=1.5, linestyle='-.')
    plt.axhline(y=full_opt_cost, color='r',
                #linestyle='',
                linewidth=2)
    #plt.axhline(y=thorup_opt_cost, color='k', linestyle='--')
    
    #plt.legend((r"$D\times D$", r"$D\times V$", r"$V\times V$", r"$T_{\text{cs}}$")) 
    plt.legend((r"$D\times D$", r"$D\times V$", r"$X\times V$")) 
    plt.xlabel("Coreset Size", fontsize=18)
    plt.xticks(fontsize=10)
    plt.ylabel(Y, fontsize=18)
    plt.savefig("fig4b.png", dpi=600)
    plt.savefig("fig4b.svg")
    plt.tight_layout()
