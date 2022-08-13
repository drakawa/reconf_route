# %%
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:06:52 2020

@author: ryuut
"""


import csv
import matplotlib.pyplot as plt
import numpy as np
import random
from sys import exit
import csv

apps = ["BT", "CG", "EP", "FT", "IS", "LU", "MG", "SP"]
labels = apps
# topologies = list()
# xmls = list()
# for line in open("simgrid_edges.txt", "r"):
#     topology, xml = line.strip().split("\t")
#     topologies.append(topology)
#     xmls.append(xml)
# print(topologies, xmls)
# legends = topologies

script_name = "mysim_rroute_trace.py"
script_dir = "../src_nnc_calc1"
tp_dir = "rtgen_rr_edgefiles"
txt_dir = "rtgen_rr_edgefiles"
tr_dir = "trfiles"
log_dir = "log_rroute"
net_name = "reconfroute"

num_splits = [2**i for i in range(14)]
legends = num_splits

tr_names = [
    "crossbar_64_bt.W.64_trace_1.00e09_4096_620085_62590300.tr",
    "crossbar_64_cg.W.64_trace_1.00e09_4096_267004_41961500.tr",
    "crossbar_64_ep.W.64_trace_1.00e09_4096_2655_9327740.tr",
    "crossbar_64_ft.W.64_trace_1.00e09_4096_18223_2379540.tr",
    "crossbar_64_is.W.64_trace_1.00e09_4096_53946_4909650.tr",
    "crossbar_64_lu.W.64_trace_1.00e09_4096_2163744_243721000.tr",
    "crossbar_64_mg.W.64_trace_1.00e09_4096_72726_5670700.tr",
    "crossbar_64_sp.W.64_trace_1.00e09_4096_1234359_134421000.tr"
]

num_node = 64
degree_seeds = [(4,1),(8,4)]

# trans_margins = [1,5,10]
trans_margin = 10

import itertools as it
import os

for (degree, num_seed) in degree_seeds:
    heights = [[0.0 for _ in labels] for _ in legends]

    for num_split, tr_name in it.product(num_splits, tr_names):
        tp_name = "{}_{}_{}_hops_ud.tp".format(num_node, degree, num_seed)
        txt_name = "{}_{}_{}_{}_{}.txt".format(tr_name, num_split, degree, num_seed, trans_margin)
        # python mysim_rroute_trace.py ../src_nnc_calc1 rtgen_rr_edgefiles 64_8_4_hops_ud.tp rtgen_rr_edgefiles crossbar_64_is.W.64_trace_1.00e09_4096_53946_4909650.tr_8192_8_4_10.txt trfiles crossbar_64_is.W.64_trace_1.00e09_4096_53946_4909650.tr log_rroute reconfroute
        cmd_str = "python {} {} {} {} {} {} {} {} {} {}".format(script_name, script_dir, tp_dir, tp_name, txt_dir, txt_name, tr_dir, tr_name, log_dir, net_name)
        # print(cmd_str)
        # print("python %s %s")
        # print(trfile, num_split, (degree, num_seed), trans_margin)
        rlog_name = "%s_%s_%s.rlog" % (tp_name, txt_name, tr_name)
        if not os.path.exists(os.path.join(log_dir, rlog_name)):
            print("no such file:", rlog_name)
            continue

        with open(os.path.join(log_dir, rlog_name)) as f:
            reader = csv.reader(f, delimiter=' ')
            l = [row for row in reader]
            print(l)
            l_data = [float(i) for i in l[0]]
            print(l_data)
            heights[num_splits.index(num_split)][tr_names.index(tr_name)] = l_data[1]

    print(heights)

    height_base = list(heights[0])
    for height in heights:
        for i in range(len(height)):
            height[i] = height[i] / height_base[i]
            # exit(1)
        # print(edge_basename)
    print(heights)

    num_legends = len(legends)
    num_labels = len(labels)
    width= 0.8 / num_legends

    lefts = [[i + width * l for i in range(num_labels)] for l in range(num_legends)]
    print(lefts)

    # plt.cla()

    plt.rcParams['font.family'] ='Times New Roman'
    plt.rcParams['font.size'] =24

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.rcParams['figure.subplot.left'] = 0.1
    plt.rcParams['figure.subplot.top'] = 0.95
    plt.rcParams['figure.subplot.right'] = 0.75
    plt.rcParams['figure.subplot.bottom'] = 0.1

    plt.rcParams["xtick.bottom"] = False

    plt.rcParams["figure.figsize"] = [12.0, 5.0]

    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.framealpha"] = 0.5
    plt.rcParams["legend.labelspacing"] = 0.05
    plt.rcParams["legend.fontsize"] = 16

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)

    if False:
        print(plt.rcParams.keys())

    fig, ax = plt.subplots()

    # ax2 = ax.twinx()

    ps = list()
    for left, height, legend in zip(lefts, heights, legends):
        print(left, height, legend)
        p = ax.bar(left, height, width, label=legend)
        ps.append(p)
    # p1 = ax.bar(x,y1,width, color="royalblue", label=r"$T_\mathrm{step}=1$")
    ax.set_ylim(0.95, 1.075)
    # ax.set_yscale("log")
    ax.set_ylabel("Average Packet Latency (normalized)", fontsize=20)

    # p2 = ax.bar(sftx,y2,width,color="crimson",label=r"$T_\mathrm{step}=50$")
    # p2 = ax2.bar(sftx,y2,width,color="red",label="Tstep=50")
    # ax2.set_ylim(0,20000)
    ax.set_xticks(lefts[len(lefts) // 2 + 1])
    # ax.set_xticklabels(l,rotation = 30)
    ax.set_xticklabels(labels, fontdict={"horizontalalignment": "right"})
    # p = [p1,p2]

    ax.grid(which = "major", axis = "y", color = "gray", linestyle = "--", linewidth = 1)
    ax.legend(ps, [i.get_label() for i in ps], bbox_to_anchor=(1.02, 1), loc='upper left', title="# of Slots")
    # ax.get_legend().get_title().set_color("red")

    # plt.legend()
    plt.savefig("eval_added_time_%d_%d.eps" % (degree, num_seed))
    plt.savefig("eval_added_time_%d_%d.png" % (degree, num_seed))
    plt.savefig("eval_added_time_%d_%d.svg" % (degree, num_seed))

exit(1)
