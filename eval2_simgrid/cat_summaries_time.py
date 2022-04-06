import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import sys

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a): 
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()

mscale.register_scale(SquareRootScale)



def list_elem_strtoint(lname):
    result = []
    for e in lname:
        result.append(int(e))
    return result

def mycsvread(arg, delim):
    f = open(arg, 'rt')
    try:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            yield(list_elem_strtoint(row))
    finally:
        f.close()
        
#sf_sizes = dict()
symsa_sizes = dict()
#for n, d, g in mycsvread("sf_sizes.txt", "\t"):
#    sf_sizes[n] = {"d": d, "g": g}
for n, d, g in mycsvread("symsa_sizes.txt", "\t"):
    symsa_sizes[n] = {"d": d, "g": g}

#print(sf_sizes, symsa_sizes)
#num_nodes = {256: list(range(3, 13))}
num_nodes = {256: list(range(5, 13))}

#num_nodes = dict()
#init_node = 64
##max_node = 4096
##max_node = 1024
#max_node = 512
#init_deg = 8
#max_port = 64
#
#tmp_node = init_node
#while tmp_node <= max_node:
#    num_nodes[tmp_node] = list()
#    max_deg = tmp_node / 4
#    tmp_deg = init_deg
#    while tmp_deg <= max_deg and tmp_deg <= max_port:
#        num_nodes[tmp_node].append(tmp_deg)
#        tmp_deg *= 2
#    tmp_node *= 2

#print(num_nodes)
num_seeds = 10


plt.rcParams["font.size"] = 36
#plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = [12.0, 9.0]
plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["lines.markeredgewidth"] = 1.0
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.framealpha"] = 0.5
plt.rcParams["legend.labelspacing"] = 0.5

plt.rcParams["font.family"] = "Times New Roman"
for tick in ["xtick", "ytick"]:
    plt.rcParams[tick + ".direction"] = "in"
    plt.rcParams[tick + ".major.width"] = 1.0
    plt.rcParams[tick + ".major.size"] = 5.0
#plt.rcParams["xtick.direction"] = "in"
#plt.rcParams["ytick.direction"] = "in"
#plt.rcParams["xtick.major.width"] = 1.0
#plt.rcParams["ytick.major.width"] = 1.0
#plt.rcParams["xtick.major.size"] = 1.0
#plt.rcParams["ytick.major.size"] = 1.0
plt.rcParams["savefig.bbox"] = "tight"

if True:
    print(plt.rcParams.keys())

#fmts_orig = ["->", "-^", "-v", ":x", "--o"]
fmts_orig = ["->", "-^", "-v", ":x"]
font_d = "Times New Roman"

vals = dict()

p_ls = [0.0, 0.2500, 0.3300, 0.5000]
for p_l in p_ls:
    tmp_vals = dict()
    for n in sorted(symsa_sizes.keys()):
        tmp_vals[n] = dict()
    if p_l != 0:
        vals["SymSA_Local(%0.2f)" % p_l] = tmp_vals
    else:
        vals["SymSA"] = tmp_vals


for f in open("time_symsa.log"):
    types = [int, int, int, int, float, float, float]
    n, d, g, Ng, ratio, ave, std = [t(e) for t, e in zip(types, f.split())]
    if n in symsa_sizes:
        if symsa_sizes[n]["d"] == d and symsa_sizes[n]["g"] == g:
            if ratio != 0.0:
                vals_key = "SymSA_Local(%0.2f)" % ratio
            else:
                vals_key = "SymSA"
            tmp_vals = vals[vals_key]
            tmp_vals[n]["ave"] = ave
            tmp_vals[n]["std"] = std
print(vals.keys())

plt.cla()
fmts = list(fmts_orig)
plots = dict()
for v_key in sorted(vals):
    plots[v_key] = dict()
    plots[v_key]["x"] = list()
    plots[v_key]["y"] = list()
    plots[v_key]["yerr"] = list()
    for n in sorted(vals[v_key]):
        plots[v_key]["x"].append(n)
        plots[v_key]["y"].append(vals[v_key][n]["ave"])
        plots[v_key]["yerr"].append(vals[v_key][n]["std"])

for x_idx, x in enumerate(plots["SymSA"]["x"]):
    print(x, plots["SymSA"]["y"][x_idx], plots["SymSA_Local(0.25)"]["y"][x_idx], 
          plots["SymSA_Local(0.33)"]["y"][x_idx], 
          plots["SymSA_Local(0.50)"]["y"][x_idx])
print(min(plots["SymSA"]["y"]))
#print(plots)

cmap = plt.get_cmap("tab10")
for p_idx, p in enumerate(sorted(plots)):
    values = plots[p]
#        plt.errorbar(values["x"], values["y"], yerr = values["yerr"], capsize = 15, fmt=fmts.pop(), label=p, markersize=15)
    plt.errorbar(values["x"], values["y"], yerr = values["yerr"], color=cmap(p_idx + 1), capsize=10, fmt=fmts.pop(), label=p, markersize=10)
    plt.plot()
    plt.legend()

plt.xscale("squareroot")
plt.xticks(np.arange(0,81,10)**2)

plt.xlabel("# of switches")
plt.ylabel("Elapsed time [s]")
plt.ylim([0,42])
#    if m in ylogscales:
#        plt.yscale("log", basey=10)
    
#    plt.xticks(np.arange(0,81,2)**2, minor=True)
plt.savefig("time.eps", transparent=True)
plt.savefig("time.png", transparent=True)
            
#    diam, aspl, inl, outl, llen, lcost = mycsvread(fname, " ")[0]
#    #sf_vals[n] = {""}
#    print(diam, aspl, inl, outl, llen, lcost)
#for n in num_nodes:
#    plt.cla()
#    fmts = list(fmts_orig)
#    plots = {"acro": dict(), "dfdn": dict()}
#    for p in plots:
#        plots[p]["x"] = list()
#        plots[p]["y"] = list()
#        plots[p]["yerr"] = list()
#    for d in num_nodes[n]:
#        tmp_values = {"acro": list(), "dfdn": list()}
#        for s in range(num_seeds):
#            # acro_top.py_128_random_16_6.out
#            for tv in tmp_values.keys():
#                for num_vl, num_sl in mycsvread("%s_top.py_%d_random_%d_%d.out" % (tv, n, d, s)):
#                    print(num_vl, num_sl)
#                    if sys.argv[-1] == "VL":
#                        tmp_values[tv].append(num_vl)
#                    elif sys.argv[-1] == "SL":
#                        tmp_values[tv].append(num_sl)
#            #print("/usr/bin/python3.6 acro_top.py %d random %d %d" % (n, d, s))
#        print(tmp_values)
#        for tv in tmp_values:
#            plots[tv]["x"].append(d)
#            plots[tv]["y"].append(np.average(tmp_values[tv]))
#            plots[tv]["yerr"].append(np.std(tmp_values[tv]))
#    for p in plots:
#        values = plots[p]
##        plt.errorbar(values["x"], values["y"], yerr = values["yerr"], capsize = 15, fmt=fmts.pop(), label=p, markersize=15)
#        plt.errorbar(values["x"], values["y"], yerr = values["yerr"], capsize = 15, fmt=fmts.pop(), label=p.upper().replace("DFDN", "DF-DN").replace("ACRO", "ACRO+GC"), markersize=15)
#        plt.plot()
#        plt.legend()
#
#    print("plots:", plots)
#    print("num_nodes:", num_nodes[n])
#    print([a / d for a, d in zip(plots["acro"]["y"], plots["dfdn"]["y"])])
#
##    plt.xlabel("x", fontfamily=font_d)
##    plt.ylabel("y", fontfamily=font_d)
#    plt.xlabel("# of ports")
#    plt.ylabel("# of layers")
#    plt.ylim([1.6, 13.6])
#    #plt.xscale("log", basex=2)
#    plt.xticks(num_nodes[n], [str(d) for d in num_nodes[n]])
#    plt.yticks(np.arange(2, 14, 2))
#    plt.savefig("%d_%s.eps" % (n, sys.argv[-1]))
##    plt.savefig("%d_%s.png" % (n, sys.argv[-1]))
#
#exit(1)
#
