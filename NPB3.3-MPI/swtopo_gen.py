import csv
import sys
import networkx as nx
import numpy as np
import itertools as it
import collections
from csvwrite_h import *

class GenDFly:
    def __init__(self, nsw_ing):
        self.nsw_ing = nsw_ing

    def gen_dfly(self):
        nsw_ing = self.nsw_ing
        n_ol = nsw_ing // 2
        n_group = nsw_ing * n_ol + 1

        nodes = [list(range(nsw_ing * i, nsw_ing * (i + 1))) for i in range(n_group)]

        G = nx.Graph()
        G.add_nodes_from(range(nsw_ing * n_group))
        print("num_nodes:", len(G))

        # full connection within group
        for nodes_ing in nodes:
            G.add_edges_from(it.combinations(nodes_ing, 2))

        # quasi overfull
        pools = [it.cycle(n) for n in nodes]

        for p1, p2 in it.combinations(pools, 2):
            G.add_edge(next(p1), next(p2))

        return G

class GenKFTree:
    def __init__(self, k):
        self.k = k

    def gen_kftree(self):
        k = self.k
        print(k)
        nodes_list = list()
        nodes = dict()
        for pod in range(k):
            for e in range(k//2):
                nodes_list.append((pod, e, "e"))
        for pod in range(k):
            for a in range(k//2):
                nodes_list.append((pod, a, "a"))
        for c in range((k//2) ** 2):
            nodes_list.append((0, c, "c"))

        G = nx.Graph()
        for i, e in enumerate(nodes_list):
            nodes[e] = i
            G.add_node(i)

        for pod in range(k):
            for e in range(k//2):
                for a in range(k//2):
                    G.add_edge(nodes[(pod, e, "e")], nodes[(pod, a, "a")])

        for pod in range(k):
            for a in range(k//2):
                for c in range(a*(k//2), (a+1)*(k//2)):
                    G.add_edge(nodes[(pod, a, "a")], nodes[(0, c, "c")])

        print(G.degree)
        return G, k * k // 2

class GenTorus:
    def __init__(self, row, dim):
        self.row = row
        self.dim = dim

    def gen_torus(self):
        row, dim = self.row, self.dim
        nodes = dict()
        G_sw = nx.Graph()
        for i, p in enumerate(it.product(range(row), repeat=dim)):
            nodes[p] = i
            G_sw.add_node(i)

        for p in nodes:
            for d in range(dim):
                p_nbr = list(p)
                p_nbr[d] = (p_nbr[d] + 1) % row
                p_nbr = tuple(p_nbr)
                G_sw.add_edge(nodes[p], nodes[p_nbr])
        return G_sw

class GenRRG:
    def __init__(self, num_sw, degree, seed):
        self.num_sw = num_sw
        self.degree = degree
        self.seed = seed

    def gen_rrg(self):
        num_sw, degree, seed = self.num_sw, self.degree, self.seed
        G_sw = nx.random_regular_graph(degree, num_sw, seed)
        return G_sw

#        G = nx.Graph()
#        G.add_nodes_from(range(num_sw + num_host))
#        host_arr = np.split(np.arange(num_host), num_sw)
#        #print(host_arr)
#        for i, hosts in enumerate(host_arr):
#            for h in hosts:
#                G.add_edge(h, i + num_host)
#        for i, j in G_sw.edges():
#            G.add_edge(i + num_host, j + num_host)
#        #print(list(G.edges()))
#        return G

if __name__ == "__main__":
    topo = sys.argv[1]
    G = None

    if topo == "random":
        types = [int, int, int]
        if len(sys.argv[2:]) < len(types):
            print("python THISFILE TOPOLOGY #ofSW DEGREE SEED")
            exit(1)
        num_sw, degree, seed = [t(elem) for t, elem in zip(types, sys.argv[2:])]
        G = GenRRG(num_sw, degree, seed).gen_rrg()
    elif topo == "torus":
        types = [int, int]
        if len(sys.argv[2:]) < len(types):
            print("python THISFILE TOPOLOGY ROW DIM")
            exit(1)
        row, dim = [t(elem) for t, elem in zip(types, sys.argv[2:])]
        G = GenTorus(row, dim).gen_torus()

    elif topo == "kftree":
        types = [int]
        if len(sys.argv[2:]) < len(types):
            print("python THISFILE TOPOLOGY K")
            exit(1)
        #k = [t(elem) for t, elem in zip(types, sys.argv[2:])]
        k = int(sys.argv[2])
        G, num_esws = GenKFTree(k).gen_kftree()
        
    elif topo == "dfly":
        types = [int]
        if len(sys.argv[2:]) < len(types):
            print("python THISFILE TOPOLOGY NSW_ING")
            exit(1)
        #k = [t(elem) for t, elem in zip(types, sys.argv[2:])]
        nsw_ing = int(sys.argv[2])
        G = GenDFly(nsw_ing).gen_dfly()
        
            
    else:
        print("Topology %s is not defined" % topo)
        exit(1)


    outf_name = "_".join(sys.argv[1:]) + ".edges"
    if topo == "kftree":
        outf_name = "_".join(sys.argv[1:] + [str(num_esws)]) + ".edges"
        
    print(outf_name)

    rows = list()
#    rows.append((num_host, num_sw))
    for i, j in G.edges():
        rows.append((i, j))

    CsvWrite(rows, outf_name).csv_write()
