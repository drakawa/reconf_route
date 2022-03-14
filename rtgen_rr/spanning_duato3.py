# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 12:36:40 2021

@author: ryuut
"""

import csv
import sys
import networkx as nx
from networkx.algorithms.dag import ancestors
import numpy as np
import scipy
# from scipy.sparse.csgraph import shortest_path as scsp
from scipy.sparse.csgraph import shortest_path
import itertools as it
import collections
from collections import deque
import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from functools import total_ordering
from collections import defaultdict

from sys import exit

from steiner2 import *
SRC = -2
DST = -1

@total_ordering
class QElem(tuple):
    def __eq__(self, other):
        return self[:-1] == other[:-1]
    def __lt__(self, other):
        return self[:-1] < other[:-1]

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
            H.add_edge((node,SRC), (node,DST))
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

class GenCDG(object):
    def __init__(self, G, root) -> None:
        self.G = G
        self.root = root
        
    def gen_cdg(self):
        pass
            
class GenUDCDG(GenCDG):
    def __init__(self, G, root) -> None:
        super().__init__(G, root)
    def gen_cdg(self):
        G = self.G
        root = self.root
        bfs_nodes = [root] + [w for v,w in nx.bfs_edges(G, source=root)]
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

    def gen_table(self):
        G = self.G
        H = self.gen_cdg()

        result_table = list()
        for dst in G.nodes():
            dst_channel = (dst, DST)
            # print("dst_channel: ", dst_channel)
            # ances = nx.ancestors(H, dst_channel)
            # print(len(ances), sorted(list(ances)))
            # nxt_channels = defaultdict(set)
            # for ance in ances:
            #     nxt_channels[ance[0]].add(ance[1])
            # print(nxt_channels)
            H_spl_tgt = nx.shortest_path_length(H, target=dst_channel)
            # print(len(H_spl_tgt), H_spl_tgt)
            # print(H[(0,SRC)])
            # print(H[(1,0)])
            # print(H[(7,0)])
            # print(H[(0,7)])
            for in_edge_H in H.nodes():
                for out_edge_H in H[in_edge_H]:
                    if out_edge_H in H_spl_tgt:
                        if out_edge_H[1] == DST:
                            continue
                        # elif in_edge_H[1] == SRC:
                        #     result_table.append((out_edge_H[0], out_edge_H[0], dst, out_edge_H[1], H_spl_tgt))
                        else:
                            result_table.append((in_edge_H[0], out_edge_H[0], dst, out_edge_H[1], H_spl_tgt[out_edge_H]))

                        # print(in_edge_H, out_edge_H, H_spl_tgt[out_edge_H])
                    pass
        # print(len(result_table), result_table[:10])
        return result_table
        # exit(1)

class GenNxSpanCDG(GenCDG):
    def __init__(self, G, alg="kruskal") -> None:
        self.G = G
        self.alg = alg

    def gen_cdg(self):
        G = self.G
        alg = self.alg

        G_undir = G.to_undirected()
        ST_undir = nx.minimum_spanning_tree(G_undir, algorithm=alg)
        ST = ST_undir.to_directed()
        print(ST.edges, len(ST.edges))

        H = nx.DiGraph()
        H.add_nodes_from(map(lambda x: (x, SRC), G.nodes()))
        H.add_nodes_from(map(lambda x: (x, DST), G.nodes()))
        H.add_nodes_from(ST.edges)
        
        for node in G.nodes():
            H.add_edge((node,SRC), (node,DST))
            for pred in ST.predecessors(node):
                H.add_edge((pred, node), (node, DST))
            for succ in ST.successors(node):
                H.add_edge((node, SRC), (node, succ))
            for pred, succ in it.product(ST.predecessors(node), ST.successors(node)):
                if pred != succ:
                    H.add_edge((pred, node), (node, succ))

        if not nx.is_directed_acyclic_graph(H):
            print("cyclic")
            exit(1)
        print(H.edges)

        # topo_sort = nx.topological_sort(H) 
        topo_sort = list(nx.lexicographical_topological_sort(H))
        self.bfs_edge_order = {k:v for v,k in enumerate(topo_sort)}
        print(topo_sort)
        print(self.bfs_edge_order)

        return H

    def get_bfs_edge_order(self):
        return self.bfs_edge_order
    
    def get_G(self):
        return self.G

class GenNxMinSpanCDG(GenNxSpanCDG):
    def __init__(self, G, alg="kruskal") -> None:
        super().__init__(G, alg=alg)
        edge_bc = nx.edge_betweenness_centrality(self.G.to_undirected())
        print(edge_bc)
        for (s,d),bc in edge_bc.items():
            self.G[s][d]["weight"] = bc
            self.G[d][s]["weight"] = bc
        # print(self.G.edges(data=True))

class GenSpanningCDG(GenCDG):
    def __init__(self, G, root) -> None:
        super().__init__(G, root)
        self.bfs_edges = list(nx.bfs_edges(G, source=root))
        self.bfs_edge_order = dict()
        
    def gen_cdg(self):
        G = self.G
        root = self.root
        bfs_edges = self.bfs_edges
        # bfs_edges = list(nx.bfs_edges(G, source=root))

        # print("bfs_edges:", bfs_edges)
        
        bfs_tree = nx.Graph()
        bfs_tree.add_nodes_from(G.nodes)
        bfs_tree.add_edges_from(bfs_edges)
        bfs_tree = bfs_tree.to_directed()
        
        for i, e in enumerate(bfs_edges):
            self.bfs_edge_order[e] = i+1
            self.bfs_edge_order[(e[1], e[0])] = -(i+1)

        order_min = min(self.bfs_edge_order.values())
        order_max = max(self.bfs_edge_order.values())
        for node_in_G in G.nodes():
            self.bfs_edge_order[(node_in_G, SRC)] = order_min - 1
            self.bfs_edge_order[(node_in_G, DST)] = order_max + 1

        H = nx.DiGraph()
        H.add_nodes_from(map(lambda x: (x, SRC), G.nodes()))
        H.add_nodes_from(map(lambda x: (x, DST), G.nodes()))
        H.add_nodes_from(bfs_edges, direction="down")
        H.add_nodes_from([(w,v) for v,w in bfs_edges], direction="up")
        
        for node in G.nodes():
            H.add_edge((node,SRC), (node,DST))
            for pred in bfs_tree.predecessors(node):
                H.add_edge((pred, node), (node, DST))
            for succ in bfs_tree.successors(node):
                H.add_edge((node, SRC), (node, succ))
            for pred, succ in it.product(bfs_tree.predecessors(node), bfs_tree.successors(node)):
                if pred != succ:
                    H.add_edge((pred, node), (node, succ))

        if not nx.is_directed_acyclic_graph(H):
            print("cyclic")
            exit(1)
        return H
    
    def get_bfs_edge_order(self):
        return self.bfs_edge_order
    
    def get_G(self):
        return self.G

class GenSparseSpanCDG(GenSpanningCDG):
    def __init__(self, G, root) -> None:
        super().__init__(G, root)

        G_undir = G.to_undirected()
        G_undir_tmp = G_undir.copy()
        while (nx.is_connected(G_undir_tmp)):
            G_undir = G_undir_tmp.copy()
            
            maxm = nx.maximal_matching(G_undir)
            # print(maxm, len(maxm))
            G_undir_tmp.remove_edges_from(maxm)
            # print(max(dict(G_undir.degree).values()))
            # print(nx.is_connected(G_undir))

        self.bfs_edges = list(nx.bfs_edges(G_undir.to_directed(), source=root))
        self.root = nx.center(G_undir.to_directed())[0]
        
class GenFurerSpanCDG(GenSpanningCDG):
    def __init__(self, G, root) -> None:
        super().__init__(G, root)
        
        G_undir = G.to_undirected()
        T = nx.minimum_spanning_tree(G_undir)
        new_T = min_deg_spanning(G_undir, T)
        
        new_T_root = nx.center(new_T)[0]
        self.root = new_T_root
        
        self.bfs_edges = list(nx.bfs_edges(new_T, source=new_T_root))
        self.bfs_edge_order = dict()
        
        # print(self.bfs_edges)
        # exit(1)
    
class CalcSPReachable:
    def __init__(self, span_bfs, genSpanCDG, sp_tgt):
        self.span_bfs = span_bfs
        self.genSpanCDG = genSpanCDG
        self.sp_tgt = sp_tgt
        self.edge_order = genSpanCDG.get_bfs_edge_order()
        self.order_max = max(self.edge_order.values())

        self.escape_order = dict()
        self.escape_order[sp_tgt] = self.order_max + 1
        self.in_escape_order = defaultdict(list)
        for i, j in span_bfs.nodes():
            if j != SRC and j != DST:
                self.escape_order[i] = self.edge_order[(i, j)]
                self.in_escape_order[j].append(self.edge_order[(i,j)])
        # print(sorted(list(self.escape_order.items())))
        # print(sorted(list(self.in_escape_order.items())))
        # exit(1)
                
    def calc_sp_reachable(self):
        span_bfs = self.span_bfs
        genSpanCDG = self.genSpanCDG
        sp_tgt = self.sp_tgt
        edge_order = self.edge_order
        order_max = self.order_max
        escape_order = self.escape_order        
        
        G = genSpanCDG.get_G()
        G_pred = nx.predecessor(G, source=sp_tgt)
                
        # print("escape_order:")
        # print(self.escape_order)
        
        in_nodes = defaultdict(list)
        for s, nxts in G_pred.items():
            for nxt in nxts:
                in_nodes[nxt].append(s)

        visited = {sp_tgt}
        queue = deque([(sp_tgt, order_max + 1, iter(in_nodes[sp_tgt]))]) # [(31, 64, iter([11, 46, 62, 2, 16, 55]))]
        
        while queue:
            parent, tmp_order, children = queue[0]
            # print(parent, tmp_order)
            try:
                to_add = False
                child = next(children)
                # print((child, parent), tmp_order)
                
                # (IN)DIRECT DEPENDENCIES FOR ADAPTIVE PATH AND DIRECT DEPENDENCY FOR ESCAPE PATH
                # a->a, a->c->c->a
                if (child, parent) in span_bfs.nodes:
                    if edge_order[(child, parent)] < tmp_order and edge_order[(child, parent)] < escape_order[parent]:
                        queue.append((child, edge_order[(child, parent)], iter(in_nodes[child])))
                        to_add = True
                # (IN)DIRECT CROSS DEPENDENCIES FOR ADAPTIVE PATH AND DIRECT CROSS DEPENDENCY FOR ESCAPE PATH
                elif (child, parent) in edge_order:
                    # print("hoge")
                    if edge_order[(child, parent)] < tmp_order and edge_order[(child, parent)] < escape_order[parent]:
                        queue.append((child, min(escape_order[parent], tmp_order), iter(in_nodes[child])))
                        to_add = True                        
                else:
                    queue.append((child, min(escape_order[parent], tmp_order), iter(in_nodes[child])))
                    to_add = True
                    
                # if (child, parent) not in edge_order: # or edge_order[(child, parent)] < tmp_order:
                #     queue.append((child, min(escape_order[parent], tmp_order), iter(in_nodes[child])))
                #     to_add = True
                # elif edge_order[(child, parent)] < tmp_order and edge_order[(child, parent)] < escape_order[parent]:
                #     queue.append((child, edge_order[(child, parent)], iter(in_nodes[child])))
                #     to_add = True
                    
                if to_add and child not in visited:
                    visited.add(child)
                    
            except StopIteration:
                queue.popleft()
                
        return sorted(list(visited))
    
    def calc_apl_reachable(self):
        span_bfs = self.span_bfs
        genSpanCDG = self.genSpanCDG
        sp_tgt = self.sp_tgt
        edge_order = self.edge_order
        order_max = self.order_max
        escape_order = self.escape_order        
        
        G = genSpanCDG.get_G()
        # G_pred = nx.predecessor(G, source=sp_tgt)
                
        # print("escape_order:")
        # print(self.escape_order)
        
        in_nodes = defaultdict(list)
        
        # for s, nxts in G_pred.items():
        #     for nxt in nxts:
        #         in_nodes[nxt].append(s)

        for s in G.nodes:
            for nxt in G[s]:
                in_nodes[nxt].append(s)
        # print(in_nodes)

        visited = {sp_tgt:0, }
        # queue = deque([(0, sp_tgt, order_max + 1, iter(in_nodes[sp_tgt]))]) # [(0, 31, 64, iter([11, 46, 62, 2, 16, 55]))]
        queue = [QElem((0, sp_tgt, order_max + 1, iter(in_nodes[sp_tgt])))] # [(0, 31, 64, iter([11, 46, 62, 2, 16, 55]))]
        heapq.heapify(queue)
        
        cutoff = 5
        
        nexts = defaultdict(set)
        
        while queue:
            tmp_level, parent, tmp_order, children = queue[0]
            # print(parent, tmp_order)
            try:
                to_add = False
                child = next(children)
                # if child in visited:
                #     continue
                # print((child, parent), tmp_order)
                
                # (IN)DIRECT DEPENDENCIES FOR ADAPTIVE PATH AND DIRECT DEPENDENCY FOR ESCAPE PATH
                # a->a, a->c->c->a
                if (child, parent) in span_bfs.nodes:
                    if edge_order[(child, parent)] < tmp_order and edge_order[(child, parent)] < escape_order[parent]:
                        # queue.append((tmp_level+1, child, edge_order[(child, parent)], iter(in_nodes[child])))
                        heapq.heappush(queue, QElem((tmp_level+1, child, edge_order[(child, parent)], iter(in_nodes[child]))))
                        to_add = True
                # (IN)DIRECT CROSS DEPENDENCIES FOR ADAPTIVE PATH AND DIRECT CROSS DEPENDENCY FOR ESCAPE PATH
                elif (child, parent) in edge_order:
                    # print("hoge")
                    if edge_order[(child, parent)] < tmp_order and edge_order[(child, parent)] < escape_order[parent]:
                        # queue.append((tmp_level+1, child, min(escape_order[parent], tmp_order), iter(in_nodes[child])))
                        heapq.heappush(queue, QElem((tmp_level+1, child, min(escape_order[parent], tmp_order), iter(in_nodes[child]))))
                        to_add = True                        
                else:
                    # queue.append((tmp_level+1, child, min(escape_order[parent], tmp_order), iter(in_nodes[child])))
                    heapq.heappush(queue, QElem((tmp_level+1, child, min(escape_order[parent], tmp_order), iter(in_nodes[child]))))
                    to_add = True
                    
                # if (child, parent) not in edge_order: # or edge_order[(child, parent)] < tmp_order:
                #     queue.append((child, min(escape_order[parent], tmp_order), iter(in_nodes[child])))
                #     to_add = True
                # elif edge_order[(child, parent)] < tmp_order and edge_order[(child, parent)] < escape_order[parent]:
                #     queue.append((child, edge_order[(child, parent)], iter(in_nodes[child])))
                #     to_add = True
                    
                if to_add and child not in visited:
                    # visited.add(child)
                    visited[child] = tmp_level + 1

                # print(child, sp_tgt, parent)
                nexts[child].add(parent)
                    
            except StopIteration:
                # queue.popleft()
                heapq.heappop(queue)
                if tmp_level > cutoff:
                    break

        if len(visited) != len(G.nodes):
            print("error")
            exit(1)
                
        # print("for debug...")
        # print(nexts)
        # exit(1)
        
        return visited
        # return sorted(list(visited.items()))
    def get_escape_order(self):
        return sorted(list(self.escape_order.items()))
    
if __name__ == "__main__":
    # num_nodes = 64
    # degree = 6
    num_nodes = 64
    degree = 8
    seed = 1

    from collections import defaultdict
    nested_dict = lambda: defaultdict(nested_dict)
    eval_result = nested_dict()

    G = GenRandom(degree, num_nodes, seed).gen_random()
    
    # plt.rcParams["savefig.bbox"] = "tight"
    # plt.rcParams["savefig.pad_inches"] = 0.1

    # plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    # ax = plt.axis('off')
    # pos = nx.circular_layout(sorted(G.nodes()))  # compute graph layout

    # e_colors = list()
    # for i, j in G.edges():
    #     e_colors.append("black")

    # nx.draw_networkx_nodes(G, pos, node_size=100, node_color="lightgray")
    # nx.draw_networkx_labels(G, pos, font_family="serif", font_size=8)
    # nx.draw_networkx_edges(G, pos, width=0.5, edge_color=e_colors)
    # plt.show()

    center_G = nx.center(G)
    print(center_G)
    # exit(1)
    
    # bfs_root = 1
    # bfs_root = random.choice(center_G)
    bfs_root= center_G[0]
    print("bfs_root:", bfs_root)
    bfs_nodes = [bfs_root] + [w for v,w in nx.bfs_edges(G, source=bfs_root)]
    # print("bfs_edges:")
    # print(sorted(list(nx.bfs_edges(G, source=bfs_root))))
    # print("bfs_edges2:")
    # print(sorted(list(nx.bfs_edges(G, source=bfs_root)), key=lambda x: x[1]))
    # print("bfs_nodes:")    
    # print(bfs_nodes)
    G_tree = nx.bfs_tree(G, source=bfs_root)
    # print(G_tree.edges)
    bfs_idx = {n: bfs_nodes.index(n) for n in sorted(G.nodes())}
    # print(bfs_idx)
    
    
    udCDG = GenUDCDG(G, bfs_root).gen_cdg()
    # print(len(udCDG.edges))

    #### CHANGE HERE TO EVALUATE DIFFERENT SPANNING TREE ####
    genSpanCDG = GenNxMinSpanCDG(G)
    # genSpanCDG = GenNxSpanCDG(G)
    # genSpanCDG = GenSpanningCDG(G, bfs_root)
    # genSpanCDG = GenSparseSpanCDG(G, bfs_root)
    # genSpanCDG = GenFurerSpanCDG(G, bfs_root)

    spanCDG = genSpanCDG.gen_cdg()
    # print(len(spanCDG.edges))
    exit(1)
    
    offset = 1
    # aspl_tgt = 58
    # aspl_tgt = 31
    
    # aspl_tgts = [0]
    aspl_tgts = sorted(list(G.nodes))
    show_graph = False
    show_detail = False
    # show_graph = True
    # show_detail = True
    
    num_apsp_ud = list()
    num_apsp_span = list()
    
    sum_apl_shortest = list()
    sum_apl_ud = list()
    sum_apl_span = list()
        
    for aspl_tgt in aspl_tgts:
        # aspl_tgt = 9
        if show_detail:
            print("aspl_tgt:", aspl_tgt)
        
        span_bfs = nx.bfs_tree(spanCDG, source=(aspl_tgt, DST), reverse=True).reverse()
    
        lens = nx.shortest_path_length(G, target=aspl_tgt)
        count_lens = [j for i,j in sorted(list(lens.items()))]
        sum_apl_shortest.extend(count_lens)
        
        lens_udcdg = nx.shortest_path_length(udCDG, target=(aspl_tgt, DST))
        count_lens_udcdg = [j - offset for i, j in sorted(list(lens_udcdg.items())) if i[1] == SRC]
        sum_apl_ud.extend(count_lens_udcdg)

        num_shortest_ud = sum(np.equal(count_lens, count_lens_udcdg))
        num_apsp_ud.append(num_shortest_ud)
        if show_detail:
            print("SPL :", count_lens)
            print("UD  :", count_lens_udcdg)        
            print(num_shortest_ud)
    
        # lens_spancdg = nx.shortest_path_length(spanCDG, target=(aspl_tgt, DST))
        # print("SPAN:", [j - offset for i, j in sorted(list(lens_spancdg.items())) if i[1] == SRC])
        
        # lens_spanbfs = nx.shortest_path_length(span_bfs, target=(aspl_tgt, DST))
        # print("SBFS:", [j - offset for i, j in sorted(list(lens_spanbfs.items())) if i[1] == SRC])
        
        # print(span_bfs.edges)
        
        G_pred = nx.predecessor(G, source=aspl_tgt)
        G_pred_items = list(G_pred.items())
        in_nodes = defaultdict(list)
        for s, nxts in G_pred_items:
            for nxt in nxts:
                in_nodes[nxt].append(s)
                
        if show_detail:
            print("predecessor:")
            print(sorted(G_pred_items))        
            print("in_nodes:")
            print(sorted(list(in_nodes.items())))
    
        n_colors = list()
        
        bfs_edge_order = genSpanCDG.get_bfs_edge_order()
    
        if show_graph:
            for n_src, n_dst in span_bfs.nodes:
                if n_dst < 0:
                    n_colors.append("lightgray")
                elif n_src in G_pred and n_dst in G_pred[n_src]:
                    n_colors.append("lightblue")
                else:
                    n_colors.append("violet")
        
            
            n_labels = dict()
            for n in span_bfs.nodes:
                if n not in bfs_edge_order:
                    n_labels[n] = str(n) + "_*" 
                else:
                    n_labels[n] = str(n) + "_" + str(bfs_edge_order[n])
            # print(n_labels)
                    
            plt.rcParams["savefig.bbox"] = "tight"
            plt.figure(figsize=(64, 32))  # image is 8 x 8 inches64
            # axes = plt.axes()
            ax = plt.axis('off')
            gname="sbfs"
            pos = nx.nx_pydot.pydot_layout(span_bfs, prog='dot', root=(aspl_tgt, DST))
            nx.draw_networkx_nodes(span_bfs, pos, node_size=400, node_color=n_colors)
            nx.draw_networkx_labels(span_bfs, pos, labels = n_labels, font_family="serif", font_size=10)
            nx.draw_networkx_edges(span_bfs, pos)
            # plt.show()
            
            plt.savefig(gname + ".eps")
            exit(1)
    
        if show_detail:
            print("bfs_edge_order:")    
            print(sorted(list(bfs_edge_order.items())))        
            print("bfs_edge_order:")    
            print(sorted(list(bfs_edge_order.items()), key=lambda x: x[0][1]))
        
        # G_undir_preds = nested_dict()
        # G_undir = G.to_undirected()
        # for i in G_undir.nodes:
        #     tmp_pred = nx.predecessor(G, source=i)
        #     for j in G_undir.nodes:
        #         if i != j:
        #             G_undir_preds[j][i] = tmp_pred[j][0]
                    
        # print(G_undir_preds)
        
        # for s in G_undir_preds:
        #     for d in G_undir_preds[s]:
        #         n = G_undir_preds[s][d]
        #         if G_undir.has_edge(s, n):
        #             G_undir.remove_edge(s, n)
    
        # print(G_undir.edges, len(G_undir.edges))
        
        # print(nx.is_connected(G_undir))
        
        calc_sp_reachable = CalcSPReachable(span_bfs, genSpanCDG, aspl_tgt)
        reachable_nodes = calc_sp_reachable.calc_sp_reachable()
        escape_order = calc_sp_reachable.get_escape_order()
        
        num_apsp_span.append(len(reachable_nodes))
        
        if show_detail:
            print("reachable:")
            print(reachable_nodes)
            print(len(reachable_nodes))
            
            print("escape_order:")
            print(escape_order)
            
        apl = calc_sp_reachable.calc_apl_reachable()
        # print(apl)
        apl_sorted = [j for i, j in sorted(list(apl.items()))]
        sum_apl_span.extend(apl_sorted)
        
        num_eq_apl_sorted = sum(np.equal(count_lens, apl_sorted))
        print(aspl_tgt, num_eq_apl_sorted)
    
    print(np.average(num_apsp_ud))
    print(np.average(num_apsp_span))
    
    print(sum(sum_apl_shortest) / (len(G.nodes) * (len(G.nodes)-1)))
    print(sum(sum_apl_ud) / (len(G.nodes) * (len(G.nodes)-1)))
    print(sum(sum_apl_span) / (len(G.nodes) * (len(G.nodes)-1)))

