import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import sys
from sys import exit

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        # axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))
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
plt.rcParams["legend.labelspacing"] = 0.05

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
plt.rcParams["legend.loc"] = "best"
plt.rcParams["legend.fontsize"] = 24
plt.rcParams["axes.formatter.use_locale"] = True

if True:
    print(plt.rcParams.keys())

fmts_orig = ["->", "-^", "-v", ":x", "--o"]
font_d = "Times New Roman"

conv_topos = ["torusrows", "hcube", "random"]
topo_name = ["3D-Torus", "Hypercube", "Random"]

ssa_params = list()
for i in mycsvread("edge_params.txt", " "):
    ### n, d, g, Ng ###
    ssa_params.append([int(e) for e in i])

print(ssa_params)

for tn, ct in zip(topo_name, conv_topos):
    sf_sizes = dict()
    symsa_sizes = dict()
    
    
    for l_idx, line in enumerate(open("topo_gen.txt", "r")):
        n_nodes_str, edges = line.strip().split()
        n_nodes = int(n_nodes_str)
        topo = edges.split("_")[0]
        if ct == topo:
            sf_sizes[n_nodes] = edges
            symsa_sizes[n_nodes] = {"d": ssa_params[l_idx][1], 
                                    "g": ssa_params[l_idx][2]}
    print(sf_sizes)
    print(symsa_sizes)

    vals = dict()
    sf_vals = dict()
    ### SF
    for n in sorted(sf_sizes.keys()):
        # fname = "mms_%dd%dg%d.edges.out" % (n, sf_sizes[n]["d"], sf_sizes[n]["g"])
        fname = "{0}.out".format(sf_sizes[n])
        for f in open(fname):
            types = [str, int, float, int, int, float, float]
            diam, aspl, inl, outl, llen, lcost = [t(e) for t, e in zip(types, f.split())][1:]
        sf_vals[n] = dict()
        sf_vals[n]["diam"] = diam
        sf_vals[n]["aspl"] = aspl
        sf_vals[n]["inl"] = inl
        sf_vals[n]["outl"] = outl
        sf_vals[n]["rat_inl"] = inl / (inl + outl)
        sf_vals[n]["llen"] = llen / (10 ** 3)
        sf_vals[n]["lcost"] = lcost / (10 ** 6)
    print(sf_vals)
    # exit(1)
    
    vals[tn] = sf_vals
    p_ls = [0, 0.2500, 0.3300, 0.5000]
    for p_l in p_ls:
        p_l_str = ""
        if p_l != 0:
            p_l_str = "l%1.4f" % p_l
            
        tmp_vals = dict()
        for n in sorted(symsa_sizes.keys()):
            fname = "symsa_%dd%dg%d%s.edges.out" % (n, symsa_sizes[n]["d"], symsa_sizes[n]["g"], p_l_str)
            #print(fname)
            for f in open(fname):
                types = [str, int, float, int, int, float, float]
                diam, aspl, inl, outl, llen, lcost = [t(e) for t, e in zip(types, f.split())][1:]
                #print(f)
            #print(diam, aspl, inl, outl, llen, lcost)
            tmp_vals[n] = dict()
            tmp_vals[n]["diam"] = diam
            tmp_vals[n]["aspl"] = aspl
            tmp_vals[n]["inl"] = inl
            tmp_vals[n]["outl"] = outl
            tmp_vals[n]["rat_inl"] = inl / (inl + outl)
            tmp_vals[n]["llen"] = llen / (10 ** 3)
            tmp_vals[n]["lcost"] = lcost / (10 ** 6)
    
        if p_l != 0:
            vals["SymSA_Local(%0.2f)" % p_l] = tmp_vals
        else:
            vals["SymSA"] = tmp_vals
    
    print(vals.keys())
    # exit(1)
    
    metrices = ["diam", "aspl", "rat_inl", "llen", "lcost"]
    
    xlabels = ["# of switches"] * 5
    ylabels = ["# of hops", "# of hops", "Ratio of intra-rack links", "Total link length [km]", "Total link cost [M$]"]
    ylims = [[1.9, 3.1], [1.58, 2.39], [0.0, 0.99], [0, 1590], [0, 68]]
               
    for m, xlabel, ylabel, ylim in zip(metrices, xlabels, ylabels, ylims):
        plt.cla()
        fmts = list(fmts_orig)
        plots = dict()
        for v_key in sorted(vals):
            plots[v_key] = dict()
            plots[v_key]["x"] = list()
            plots[v_key]["y"] = list()
            for n in sorted(vals[v_key]):
                plots[v_key]["x"].append(n)
                plots[v_key]["y"].append(vals[v_key][n][m])
    
    
        for p in sorted(plots):
            values = plots[p]
            plt.errorbar(values["x"], values["y"], capsize=10, fmt=fmts.pop(), label=p, markersize=10)
            plt.plot()
            plt.legend()
    
        plt.xscale("squareroot")
        # plt.xticks(np.arange(0,41,10)**2)
        plt.xticks(np.arange(5,36,5)**2)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    
        plt.savefig("%s_%s.eps" % (tn,m), transparent=True)
        plt.savefig("%s_%s.png" % (tn,m), transparent=True)
        # exit(1)
                
