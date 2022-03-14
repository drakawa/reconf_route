import csv
import sys
import networkx as nx
import numpy as np
import scipy
# from scipy.sparse.csgraph import shortest_path as scsp
from scipy.sparse.csgraph import shortest_path
import itertools as it
import collections
import random
from csvwrite_h import *

SRC = -2
DST = -1

class GenRandom:
    def __init__(self, d, n, s):
        self.d = d
        self.n = n
        self.s = s
    def gen_random(self):
        d, n, s = self.d, self.n, self.s
        return nx.random_regular_graph(d, n, s).to_directed()

class GenChannelGraph:
    def __init__(self, d, n, s):
        self.d = d
        self.n = n
        self.s = s
    def gen_channel_graph(self):
        d, n, s = self.d, self.n, self.s
        G = GenRandom(d, n, s).gen_random()
        H = nx.DiGraph()
        H.add_nodes_from(G.edges())
        H.add_nodes_from(map(lambda x: (x, SRC), G.nodes()))
        H.add_nodes_from(map(lambda x: (x, DST), G.nodes()))
        for node in G.nodes():
            for pred in G.predecessors(node):
                H.add_edge((pred, node), (node, DST))
            for succ in G.successors(node):
                H.add_edge((node, SRC), (node, succ))
            for pred, succ in it.product(G.predecessors(node), G.successors(node)):
                H.add_edge((pred, node), (node, succ))
        return H
        
class GenURTraffic:
    def __init__(self, n, s):
        self.n = n
        self.s = s
    def gen_ur_traffic(self):
        n, s = self.n, self.s
        srcs = np.array(range(n))
        dsts = np.array(range(n))
        random.seed(s)

        while np.any(srcs == dsts):
            random.shuffle(dsts)
        return dict(zip(srcs, dsts))
        
class GenSpanningCDG:
    def __init__(self, G, src) -> None:
        self.G = G
        self.src = src
    def gen_spanning_cdg(self):
        G = self.G
        src = self.src
        bfs_nodes = [src] + [w for v,w in nx.bfs_edges(G, source=src)]
        # print(bfs_nodes)
        bfs_idx = {n: bfs_nodes.index(n) for n in sorted(G.nodes())}
        # print(bfs_idx)

        H = nx.DiGraph()
        H.add_nodes_from(G.edges())
        H.add_nodes_from(map(lambda x: (x, SRC), G.nodes()))
        H.add_nodes_from(map(lambda x: (x, DST), G.nodes()))

        for node in G.nodes():
            H.add_edge((node,SRC), (node,DST))
            for pred in G.predecessors(node):
                H.add_edge((pred, node), (node, DST))
            for succ in G.successors(node):
                H.add_edge((node, SRC), (node, succ))
            for pred, succ in it.product(G.predecessors(node), G.successors(node)):
                if not (bfs_idx[pred] < bfs_idx[node] and bfs_idx[node] > bfs_idx[succ]):
                    H.add_edge((pred, node), (node, succ))

        # print(H.edges())

        if not nx.is_directed_acyclic_graph(H):
            print("cyclic")
            exit(1)
        return H


if __name__ == "__main__":
    # ns = [256]
    # ds = list(range(3,17))
    # ss = list(range(10))
    ns = [64]
    ds = list(range(3,10))
    # ss = list(range(10))
    # ns = [256]
    # ds = [3]
    ss = [1]

    num_pools = 10

    from collections import defaultdict
    nested_dict = lambda: defaultdict(nested_dict)
    eval_result = nested_dict()

    for n, d, s in it.product(ns, ds, ss):
        print(n, d, s)

        # d, n, s = 16, 64, 1
        G = GenRandom(d, n, s).gen_random()
        # print(G.edges())

        ur_seed = 2
        ur_traf = GenURTraffic(n, ur_seed).gen_ur_traffic()
        # print(ur_traf)

        bfs_edges = nx.bfs_edges(G, source=0)
        # print(list(bfs_edges))

        Hs = [GenSpanningCDG(G,i).gen_spanning_cdg() for i in sorted(G.nodes)]

        H_sorted_nodes = sorted(Hs[0].nodes, key=lambda x: (x[1],x[0]))
        Hs = [nx.to_scipy_sparse_matrix(H, nodelist=H_sorted_nodes) for H in Hs]
        
        H1, H2 = Hs[0], Hs[3]
    
        coH = nx.Graph()
        coH.add_nodes_from(sorted(G.nodes))

        def calc_coH_edge(inputs):
            i,j = inputs
            H4 = Hs[i].multiply(Hs[j])
            H4_scsp = shortest_path(H4)
            H4_scsp_sd = H4_scsp[0:n,n:2*n]
            num_posinf = sum(np.isposinf(H4_scsp_sd))
            return sum(num_posinf) == 0

        from multiprocessing import Pool
        p = Pool(num_pools)
        result = p.map(calc_coH_edge, it.combinations(sorted(G.nodes), 2))
        coH_edges_p = [(i,j) for idx, (i, j) in enumerate(it.combinations(sorted(G.nodes), 2)) if result[idx]]
        # print(coH_edges_p)

        coH.add_edges_from(coH_edges_p)

        largest_cc = max(nx.connected_components(coH), key=len)
        coH_cc = coH.subgraph(largest_cc).copy()
        # print(len(coH_cc.edges), len(coH_cc))
        # print(coH_cc.edges, coH_cc)
        eval_result[n][d][s]["cc_node"] = len(coH_cc)
        eval_result[n][d][s]["cc_edge"] = len(coH_cc.edges)
        # print(ur_traf)
        ur_traf_sds = [(H_sorted_nodes.index((i,SRC)), H_sorted_nodes.index((j,DST))) for i,j in ur_traf.items()]
        # print(ur_traf_sds)

        # exit(1)

        def calc_coH(inputs):
            i, ur_traf_sds = inputs
            H_i_scsp = shortest_path(Hs[i])
            sd_offset = 1
            splens = [H_i_scsp[src,dst] - sd_offset for src,dst in ur_traf_sds]
            return np.average(splens)
            # i,j = inputs
            # H4 = Hs[i].multiply(Hs[j])
            # H4_scsp = shortest_path(H4)
            # H4_scsp_sd = H4_scsp[0:n,n:2*n]
            # num_posinf = sum(np.isposinf(H4_scsp_sd))
            # return sum(num_posinf) == 0
        # print(calc_coH((0, ur_traf_sds)))
        # print(calc_coH((1, ur_traf_sds)))
        # print(calc_coH((2, ur_traf_sds)))
        # print(calc_coH((3, ur_traf_sds)))
        # print(calc_coH((4, ur_traf_sds)))

        p = Pool(num_pools)
        coH_result = p.map(calc_coH, it.product(sorted(G.nodes), [ur_traf_sds]))
        # print(coH_result)

        coH_result_in_cc = [coH_result[i] for i in sorted(coH_cc.nodes)]
        # print(coH_result_in_cc)
        eval_result[n][d][s]["ur_max"] = np.max(coH_result_in_cc)
        eval_result[n][d][s]["ur_ave"] = np.average(coH_result_in_cc)
        eval_result[n][d][s]["ur_min"] = np.min(coH_result_in_cc)

    # print(eval_result)

    print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % ("n", "d", "average(cc_nodes)", "average(cc_edges)", "average(ur_maxs)", "average(ur_aves)", "average(ur_mins)", "average(cc_red_ave)", "average(cc_red_max)"))
    for n, d in it.product(ns, ds):
        cc_nodes = [eval_result[n][d][s]["cc_node"] for s in ss]
        cc_edges = [eval_result[n][d][s]["cc_edge"] for s in ss]
        ur_maxs = [eval_result[n][d][s]["ur_max"] for s in ss]
        ur_aves = [eval_result[n][d][s]["ur_ave"] for s in ss]
        ur_mins = [eval_result[n][d][s]["ur_min"] for s in ss]
        cc_red_ave = [eval_result[n][d][s]["ur_min"] / eval_result[n][d][s]["ur_ave"] for s in ss]
        cc_red_max = [eval_result[n][d][s]["ur_min"] / eval_result[n][d][s]["ur_max"] for s in ss]
        print("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t" % (n, d, np.average(cc_nodes), np.average(cc_edges), np.average(ur_maxs), np.average(ur_aves), np.average(ur_mins), np.average(cc_red_ave), np.average(cc_red_max)))
        pass
    
