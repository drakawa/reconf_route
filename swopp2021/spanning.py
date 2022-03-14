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
    d, n, s = 16, 64, 1
    G = GenRandom(d, n, s).gen_random()
    # print(G.edges())

    ur_seed = 2
    ur_traf = GenURTraffic(n, ur_seed).gen_ur_traffic()
    # print(ur_traf)

    bfs_edges = nx.bfs_edges(G, source=0)
    # print(list(bfs_edges))

    # H1 = GenSpanningCDG(G,0).gen_spanning_cdg()
    # H2 = GenSpanningCDG(G,3).gen_spanning_cdg()
    Hs = [GenSpanningCDG(G,i).gen_spanning_cdg() for i in sorted(G.nodes)]

    H_sorted_nodes = sorted(Hs[0].nodes, key=lambda x: (x[1],x[0]))
    # Hs = [nx.to_scipy_sparse_matrix(H, nodelist=H_sorted_nodes) for H in Hs]
    Hs = [nx.to_scipy_sparse_matrix(H, nodelist=H_sorted_nodes) for H in Hs]
    
    H1, H2 = Hs[0], Hs[3]
    # print(H1.edges)
    # print(nx.to_scipy_sparse_matrix(H1).toarray())
    # exit(1)
    coH = nx.Graph()
    coH.add_nodes_from(sorted(G.nodes))

    def calc_coH_edge(inputs):
        i,j = inputs
        H4 = Hs[i].multiply(Hs[j])
        H4_scsp = shortest_path(H4)
        H4_scsp_sd = H4_scsp[0:n,n:2*n]
        num_posinf = sum(np.isposinf(H4_scsp_sd))
        return sum(num_posinf) == 0

    import time
    time1 = time.time()

    # for i, j in it.combinations(sorted(G.nodes), 2):
    #     # print(i,j)
    #     # H3 = nx.intersection(Hs[i],Hs[j])
    #     # H3_sorted_nodes = sorted(H3.nodes, key=lambda x: (x[1],x[0]))
    #     # H4 = nx.to_scipy_sparse_matrix(H3,nodelist=H3_sorted_nodes)
    #     H4 = Hs[i].multiply(Hs[j])
    #     # print(H3.nodes)
    #     # print(H4, sum(H4))
    #     H4_scsp = shortest_path(H4)
    #     # print(H4_scsp)
    #     H4_scsp_sd = H4_scsp[0:n,n:2*n]
    #     # print(H4_scsp_sd)
    #     num_posinf = sum(np.isposinf(H4_scsp_sd))
    #     # print(num_posinf)
    #     if sum(num_posinf) == 0:
    #         coH.add_edge(i, j)
    #     # print(sum(num_posinf))
    #     # print(coH.edges)
    #     # exit(1)
    # print(coH.edges)

    time2 = time.time()
    from multiprocessing import Pool
    p = Pool(20)
    result = p.map(calc_coH_edge, it.combinations(sorted(G.nodes), 2))
    coH_edges_p = [(i,j) for idx, (i, j) in enumerate(it.combinations(sorted(G.nodes), 2)) if result[idx]]
    print(coH_edges_p)
    # print(list(coH.edges) == list(coH_edges_p))
    time3 = time.time()

    print(time2 - time1)
    print(time3 - time2)
    exit(1)


    H3 = nx.intersection(H1, H2)
    # print(len(H1.edges), len(H2.edges), len(H3.edges))
    # print(H3.edges())

    # print(H3.nodes)
    H3_sorted_nodes = sorted(H3.nodes, key=lambda x: (x[1],x[0]))
    print(H3_sorted_nodes)
    # exit(1)
    H4 = nx.to_scipy_sparse_matrix(H3,nodelist=H3_sorted_nodes)
    print(H3.nodes)
    print(H4, sum(H4))
    H4_scsp = shortest_path(H4)
    print(H4_scsp)
    H4_scsp_sd = H4_scsp[0:n,n:2*n]
    print(H4_scsp_sd)
    num_posinf = sum(np.isposinf(H4_scsp_sd))
    print(num_posinf)
    print(sum(num_posinf))
    # exit(1)
    print(G[0])

    H1_sp = nx.shortest_path(H1)
    # print(sorted([y for y in H1_sp[(0,SRC)].keys() if y[1] == DST]))
    H3_sp = nx.shortest_path(H3)
    H3_sp_len = [len([y for y in H3_sp[(x,SRC)].keys() if y[1] == DST]) for x in sorted(G.nodes)]
    print(H3_sp_len)
    print(set(H3_sp_len))
    # print(sorted([y for y in H3_sp[(0,SRC)].keys() if y[1] == DST]))
    # print(H3_sp[(0,SRC)].keys())


    # H = GenChannelGraph(d, n, s).gen_channel_graph()
    # print(H.nodes())
    # print(len(H.edges()))

    # # bfs_edges = nx.bfs_edges(H, source=(170, 6))
    # bfs_edges = nx.bfs_edges(H, source=(12, 3))
    # print(list(bfs_edges))
