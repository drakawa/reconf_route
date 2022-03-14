# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:29:58 2021

@author: ryuut
"""
import csv
import sys
import networkx as nx
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
import copy
from multiprocessing import Pool
import argparse
import os
import json

from sys import exit

from steiner2 import *
from spanning_duato3 import *
SRC = -2
DST = -1

def rec_dd():
    return defaultdict(rec_dd)

def my_successor(DiG, target):
    return nx.predecessor(DiG.reverse(), source=target)

def bfs_iter_nxts(G_undir, tgt):
    G_undir_nxts = nx.predecessor(G_undir, tgt)
    # print(G_undir_nxts)
    G_undir_bfs_nodes = [tgt] + [w for v,w in nx.bfs_edges(G_undir, source=tgt)]
    # print(G_undir_bfs_nodes)
    
    for n in G_undir_bfs_nodes:
        for nxt in G_undir_nxts[n]:
            yield n, nxt
    

class Ddstry:
    def __init__(self, bfs_edge_order, tgt, cdst1y, cdstpcy, cdstccy):
        self.bfs_edge_order = bfs_edge_order # {(0, 15): 1, (15, 0): -1, (0, 10): 2, ...}
        self.tgt = tgt # 0
        self.cdst1y = cdst1y # {(49, 37), (63, 61), (14, 63), ...}
        self.cdstpcy = cdstpcy # {(37, 48): {(49, 37)}, (37, 38): {(49, 37)}, ...}
        self.cdstccy = cdstccy # {(51, 49): {(49, 37)}, (60, 49): {(49, 37)}, ...}
        
        self.cdstry = set()
        self.cdstry_by_tail = defaultdict(set)
        self.cdstry_by_head = defaultdict(set)

        self.min_esc = dict()
        
        self.D = nx.DiGraph()
        
        self.bfs_order_min = min(self.bfs_edge_order.values())
        self.bfs_order_max = max(self.bfs_edge_order.values())
        
    #### NOTE: assuming channels are traversed in reversed bfs order (never revisited) ####
    def update_D(self, channel):
        if channel in self.cdst1y or channel in self.D.nodes:
            return False
        
        tmp_D = self.D.copy()
        order_c = None
        if channel in self.bfs_edge_order:
            order_c = self.bfs_edge_order[channel]
            
        porder_c = {self.bfs_edge_order[c] for c in self.cdstpcy[channel]}
        corder_c = {self.bfs_edge_order[c] for c in self.cdstccy[channel]}
        
        # ADD NODE
        # print(porder_c, corder_c)
        # print("add_node:", channel)
        tmp_D.add_node(channel, order=order_c, porder=porder_c, corder=corder_c)
        
        # ADD ARC
        for out_arc in self.cdstry_by_head[channel[1]]:
            tmp_D.add_edge(channel, out_arc)
        for in_arc in self.cdstry_by_tail[channel[0]]:
            tmp_D.add_edge(in_arc, channel)
            
        # print(tmp_D.nodes(data=True))
        # CHECK IF ACYCLIC
        # if not nx.is_directed_acyclic_graph(tmp_D):
        #     print("acyclic.")
        #     return False
        
        # CHECK DEPENDENCIES FOR CHANNEL

        deps_from = set([self.bfs_order_min - 1])
        order_channel = tmp_D.nodes[channel]["order"]
        if order_channel != None:
            deps_from.add(order_channel)
        deps_from |= tmp_D.nodes[channel]["porder"]

        deps_to = set([self.bfs_order_max + 1])
        deps_to |= tmp_D.nodes[channel]["corder"]

        for out_arc in self.cdstry_by_head[channel[1]]:
            deps_to.add(self.min_esc[out_arc])

        max_deps_from = max(deps_from)
        min_deps_to = min(deps_to)
        
        # print(deps_from, deps_to)
        if not max_deps_from < min_deps_to:
            return False

        # # CHECK DEPENDENCIES FOR ALL NODE IN TMP_D
        # for node_inD in tmp_D.nodes:
        #     deps_from = set([self.bfs_order_min - 1])
        #     order_node_inD = tmp_D.nodes[node_inD]["order"]
        #     if order_node_inD != None:
        #         deps_from.add(order_node_inD)
            
        #     deps_from |= tmp_D.nodes[node_inD]["porder"]
            
        #     deps_to = set([self.bfs_order_max + 1])
            
        #     deps_to |= tmp_D.nodes[node_inD]["corder"]
            
        #     descs = nx.descendants(tmp_D, node_inD)
        #     for desc in descs:
        #         deps_to |= tmp_D.nodes[desc]["corder"]
            
        #     max_deps_from = max(deps_from)
        #     min_deps_to = min(deps_to)
            
        #     # print(deps_from, deps_to)
        #     if not max_deps_from < min_deps_to:
        #         return False
                                
        # FINALLY UPDATE CDSTRY etc
        self.cdstry.add(channel)
        self.cdstry_by_tail[channel[1]].add(channel)
        self.cdstry_by_head[channel[0]].add(channel)

        # MEMORIZE MIN_ESC
        self.min_esc[channel] = min_deps_to
        
        # UPDATE D
        self.D = tmp_D
        return True
        

def get_stats(values):
    return np.average(values), np.std(values)

def gen_ud(num_nodes, degree, seed, pri_mode, out_dir, G_undir, G, **kwargs):
    num_nodes_in_G = len(G)

    edge_outf = os.path.join(out_dir, "%d_%d_%d.edges" % (num_nodes, degree, seed))
    if os.path.exists(edge_outf) and os.path.getsize(edge_outf) > 0:
        nx.write_edgelist(G_undir, edge_outf, data=False)
    
    tp_outf = os.path.join(out_dir, "%d_%d_%d_%s.tp" % (num_nodes, degree, seed, pri_mode))
    if os.path.exists(tp_outf) and os.path.getsize(tp_outf) > 0:
        with open(tp_outf, 'w') as f: 
            writer = csv.writer(f, delimiter=" ")
            for e_s, e_d in sorted(G_undir.edges):
                # router 2 router 11 1
                writer.writerow(["router", e_s, "router", e_d, 1])
            for node in sorted(G.nodes):
                # node 9 router 9
                writer.writerow(["node", node, "router", node])

    ud_tp_outf = os.path.join(out_dir, "%d_%d_%d_%s_ud.tp" % (num_nodes, degree, seed, pri_mode))
    if os.path.exists(ud_tp_outf) and os.path.getsize(ud_tp_outf) > 0:
        with open(ud_tp_outf, 'w') as f:
     
            writer = csv.writer(f, delimiter=" ")
            for e_s, e_d in sorted(G_undir.edges):
                # router 2 router 11 1
                writer.writerow(["router", e_s, "router", e_d, 1])
    
            for node in sorted(G.nodes):
                # node 9 router 9
                writer.writerow(["node", node, "router", node])
            # writer.writerows(rt_data)

    center_G = nx.center(G)
    # print(center_G)
    if "bfs_root" in kwargs:
        bfs_root = kwargs["bfs_root"]
    else:
        bfs_root = center_G[0]
    # print(bfs_root)

    gen_udcdg = GenUDCDG(G, bfs_root)

    udCDG = gen_udcdg.gen_cdg()
    udcdg_table = gen_udcdg.gen_table()
    # print(len(udcdg_table), udcdg_table[:10])

    udrt_data = list()
    # pn, s, d, n, hops → pn, pv, s, d, n, v, pri
    for pn, s, d, n, hops in udcdg_table:
        if pri_mode == "hops":
            udrt_data.append((pn, 0, s, d, n, 0, num_nodes_in_G - hops))
        elif pri_mode == "same":
            udrt_data.append((pn, 0, s, d, n, 0, 0))

    ud_rt_outf = os.path.join(out_dir, "%d_%d_%d_%s_ud.rt" % (num_nodes, degree, seed, pri_mode))
    if os.path.exists(ud_rt_outf) and os.path.getsize(ud_rt_outf) > 0:
        with open(ud_rt_outf, 'w') as f:
     
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(udrt_data)

    return udCDG

def gen_cspany(n_srs, my_successor):
    queue = [(n_sr, SRC) for n_sr in n_srs]
    cspany = set(queue)
    while queue:
        tmp_c = queue.pop(0)
        if tmp_c[1] != DST:
            tmp_c_nxt = my_successor[tmp_c][0] 
            cspany.add(tmp_c_nxt)
            if tmp_c_nxt != DST:
                queue.append(tmp_c_nxt)

    return cspany

def min_avalilable(c_s, c_d, vc, shortest_outputs, escape_channels, span_bfs, bfs_edge_order):
    # determine channel type
    in_span_bfs = ((c_s, c_d, vc) in span_bfs.nodes)
    order_c = bfs_edge_order.get((c_s, c_d, vc))
    has_order = (order_c != None)

    if len(shortest_outputs[c_d]) == 0:
        return False, None
    elif in_span_bfs:
        # escape channel
        if order_c < max(shortest_outputs[c_d].values()) and not max(escape_channels[c_d].values()) <= order_c:
            return True, order_c
        else:
            return False, None
    elif has_order:
        # semi-escape channel
        if order_c < max(shortest_outputs[c_d].values()) and not max(escape_channels[c_d].values()) <= order_c:
            return True, min([max(shortest_outputs[c_d].values()), max(escape_channels[c_d].values())])
        else:
            return False, None
    else:
        # non-escape channel
            return True, min([max(shortest_outputs[c_d].values()), max(escape_channels[c_d].values())])

def select_channels_to_add(tmp_n, cands, escape_channels):
    if len(cands) == 0:
        return dict()
    order_parallel_escape = max(escape_channels[tmp_n].values())
    to_adds = {k:v for k,v in cands.items() if v >= order_parallel_escape}
    if len(to_adds) > 0:
        return to_adds
    else:
        order_max = max(cands.values())
        to_adds = {k:v for k,v in cands.items() if v == order_max}
        return to_adds
        # to_add = max(cands.items(), key=lambda x: x[1])
        # return dict([to_add])

def gen_rf_cnc(num_nodes, degree, seed, pri_mode, out_dir, G_undir, G, num_vcs, spanning):
    num_nodes_in_G = len(G)
    center_G = nx.center(G)
    bfs_root = center_G[0]
    # print(bfs_root)

    escape_vc = num_vcs - 1

    if spanning == "NxSpan":
        genSpanCDG = GenNxSpanCDG(G, alg="kruskal")
    elif spanning == "Span":
        genSpanCDG = GenSpanningCDG(G, bfs_root)
    elif spanning == "SparseSpan":
        genSpanCDG = GenSparseSpanCDG(G, bfs_root)
    elif spanning == "FurerSpan":
        genSpanCDG = GenFurerSpanCDG(G, bfs_root)
    else:
        print("Invalid Spanning name.")
        exit(1)
    
    spanCDG = genSpanCDG.gen_cdg()
    spanCDG_mapping =dict()
    for s,d in spanCDG.nodes():
        spanCDG_mapping[(s,d)] = (s,d,escape_vc)
    spanCDG = nx.relabel_nodes(spanCDG, spanCDG_mapping)
    # print("spanCDG.edges:", spanCDG.edges)
    bfs_edge_order = genSpanCDG.get_bfs_edge_order()
    # print("bfs_edge_order", bfs_edge_order)
    bfs_edge_order = {(s,d,escape_vc):v for ((s,d),v) in bfs_edge_order.items()}
    # print("(new) bfs_edge_order", bfs_edge_order)
    
    R_data = rec_dd()
    R1_data = rec_dd()

    num_shortests = dict()
    for tgt in sorted(G.nodes):
        R_data_dst = rec_dd()
        R1_data_dst = rec_dd()
        escape_channels = rec_dd()
        shortest_outputs = rec_dd()

        # print("current tgt:", tgt)
        succ_tgt = my_successor(spanCDG, target=(tgt,DST,escape_vc))
        # print("my_successor:", succ_tgt)
        span_bfs = nx.bfs_tree(spanCDG, source=(tgt, DST,escape_vc), reverse=True).reverse()
        # print("span_bfs:", sorted(span_bfs.edges))

        ### initialize R1(*,tgt) ###
        for c_tmp, c_nxts in succ_tgt.items():
            if c_tmp != (tgt, SRC, escape_vc):
                R1_data_dst[c_tmp] = c_nxts
            if c_tmp[1] == SRC:
                for c_nxt in c_nxts:
                    escape_channels[c_tmp[0]][c_nxt] = bfs_edge_order[c_nxt]
            # print(c_tmp, c_nxts)
        
        for eject in escape_channels[tgt]:
            shortest_outputs[tgt][eject] = bfs_edge_order[eject]
        # print("R1_data_dst:", R1_data_dst, len(R1_data_dst), len(succ_tgt))
        # print("escape_channels", escape_channels)
        # print("shortest_outputs:", shortest_outputs)

        # ### generate C_span(y) (to be C^dst_1(y)) ###
        # cspany= gen_cspany(G.nodes, succ_tgt)
        # print(sorted(list(cspany)))

        G_undir_nxts = nx.predecessor(G_undir, tgt)
        # print(G_undir_nxts)
        G_undir_bfs_nodes = [w for v,w in nx.bfs_edges(G_undir, source=tgt)]
        # print(G_undir_bfs_nodes)
        for tmp_n in G_undir_bfs_nodes:
            tmp_nxts = G_undir_nxts[tmp_n]
            # print("####################")
            # print("tmp_n, tmp_nxts:", tmp_n, tmp_nxts)
            para_escape_orders = escape_channels[tmp_n].values()
            # print("para_escape_orders:", para_escape_orders)
            cands = dict()
            
            for tmp_nxt, tmp_vc in it.product(tmp_nxts, range(num_vcs)):
                # print("tmp_n, tmp_nxt, tmp_vc:", tmp_n, tmp_nxt, tmp_vc)
                # print("(tmp_n, tmp_nxt, tmp_vc) in span_bfs.nodes:", (tmp_n, tmp_nxt, tmp_vc) in span_bfs.nodes)
                # print("edge_order:", bfs_edge_order.get((tmp_n, tmp_nxt, tmp_vc)))
                # print("shortest_outputs[tmp_nxt]:", shortest_outputs[tmp_nxt])
                # print("escape_channels[tmp_nxt]:", escape_channels[tmp_nxt])
                min_avail, owing_order = min_avalilable(tmp_n, tmp_nxt, tmp_vc, shortest_outputs, escape_channels, span_bfs, bfs_edge_order)
                # print("min_avail, owing_order:", min_avail, owing_order)
                if min_avail:
                    cands[(tmp_n, tmp_nxt, tmp_vc)] = owing_order

            # print("candidates:", cands)
            selected_channels = select_channels_to_add(tmp_n, cands, escape_channels)
            # print("selected_channels:", selected_channels)
            for sc, sc_owing_order in selected_channels.items():
                shortest_outputs[tmp_n][sc] = sc_owing_order
            # print("shortest_outputs:", shortest_outputs)

        # print("shortest_outputs:")
        num_shortests_tgt = sum((len(shortest_outputs[n]) > 0 for n in sorted(G.nodes)))
        # print(num_shortests_tgt)
        num_shortests[tgt] = num_shortests_tgt - 1

        # for n in sorted(G.nodes):
        #     print(n, dict(shortest_outputs[n]))
        
        # for c_s, c_d in bfs_iter_nxts(G_undir, tgt):
        #     print(c_s, c_d)
        #     pass

        # exit(1)

    print(num_shortests)
    print(np.average(list(num_shortests.values())), np.average(list(num_shortests.values())) / (len(G.nodes)-1))
    exit(1)


def gen_Grdst(num_nodes, degree, seed, pri_mode, out_dir, G_undir, G, spanning = "Span"):
    num_nodes_in_G = len(G)
    center_G = nx.center(G)
    # print(center_G)
    bfs_root = center_G[0]
    # print(bfs_root)

    #### CHANGE HERE TO EVALUATE DIFFERENT SPANNING TREE ####
    if spanning == "NxSpan":
        # genSpanCDG = GenNxSpanCDG(G, alg="kruskal")
        genSpanCDG = GenNxMinSpanCDG(G, alg="kruskal")
    elif spanning == "Span":
        genSpanCDG = GenSpanningCDG(G, bfs_root)
    elif spanning == "SparseSpan":
        genSpanCDG = GenSparseSpanCDG(G, bfs_root)
    elif spanning == "FurerSpan":
        genSpanCDG = GenFurerSpanCDG(G, bfs_root)
    else:
        print("Invalid Spanning name.")
        exit(1)
    
    spanCDG = genSpanCDG.gen_cdg()
    # print("spanCDG.edges:", spanCDG.edges)
    bfs_edge_order = genSpanCDG.get_bfs_edge_order()
    # print("bfs_edge_order", bfs_edge_order)

    rt_data = list()

    Grdsts, Grdst_spl_tgts = dict(), dict()
    for tgt in sorted(G.nodes):
        print("current tgt:", tgt)
        succ_tgt = my_successor(spanCDG, target=(tgt,DST))
        # print("my_successor:", succ_tgt)
        span_bfs = nx.bfs_tree(spanCDG, source=(tgt, DST), reverse=True).reverse()
        # print("span_bfs:", span_bfs.edges)
        
        cdst1y= set()
        for (i, j), (k, l) in span_bfs.edges:
            if j == SRC and l != SRC and l != DST:
                cdst1y.add((k, l))
        # print("cdst1y:")
        # print(cdst1y, len(cdst1y))
        
        G_undir_nxts = nx.predecessor(G_undir, tgt)
        # print("G_undir_nxts:", G_undir_nxts)
        r1_shortest = defaultdict(bool)
        
        for c_s, c_d in cdst1y:
            if c_s == tgt or c_d in G_undir_nxts[c_s]:
                r1_shortest[c_s] = True
            else:
                r1_shortest[c_s] = False
        # print("r1_shortest:", r1_shortest)
        
        cdst1y_by_tail = defaultdict(set)
        cdst1y_by_head = defaultdict(set)
        
        for c_s, c_d in cdst1y:
            cdst1y_by_tail[c_d].add((c_s, c_d))
            cdst1y_by_head[c_s].add((c_s, c_d))
        
        # print("cdst1y_by_tail:", sorted(list(cdst1y_by_tail.items())))
        # print("cdst1y_by_head:", sorted(list(cdst1y_by_head.items())))
        
        cdstpcy = defaultdict(set) # 
        cdstccy = defaultdict(set)
        
        for cp_s, cp_d in cdst1y:
            for out_edge in G.out_edges(cp_d):
                cdstpcy[out_edge].add((cp_s, cp_d))
                pass
            for in_edge in G.in_edges(cp_s):
                cdstccy[in_edge].add((cp_s, cp_d))
                pass
            # print(cp_s, cp_d, G[cp_s])
            # exit(1)
            pass

        # print("cdstpcy:", cdstpcy)
        # print("cdstccy:", cdstccy)
        
        
        D_dstry = Ddstry(bfs_edge_order, tgt, cdst1y, cdstpcy, cdstccy)
        
        # print("D_dstry sorted:")
        # for t in sorted(list(D_dstry.cdstry_by_head.items())):
        #     print(t)

        for c_s, c_d in bfs_iter_nxts(G_undir, tgt):
            D_dstry.update_D((c_s, c_d))
            # print("updated status:")
            # print(D_dstry.D.edges)
            # print(D_dstry.D.nodes)
            # print(D_dstry.cdstry_by_head)
            # print(D_dstry.cdstry_by_tail)
            
            # iter_count += 1
            # if iter_count > iter_max:
            #     exit(1)
            

        G_undir_nxts = nx.predecessor(G_undir, tgt)
        # print(sorted(G_undir_nxts.items()))
        G_undir_bfs_nodes = [tgt] + [w for v,w in nx.bfs_edges(G_undir, source=tgt)]
        # print(G_undir_bfs_nodes)
        
        spl_tgt = nx.shortest_path_length(G_undir, target=tgt)
        
        Grdst = nx.DiGraph()
        Grdst.add_nodes_from(G_undir.nodes)
        
        for c_s, c_d in cdst1y:
            if c_d in G_undir_nxts[c_s]:
                Grdst.add_edge(c_s, c_d, vc=0, weight=1)
            else:
                Grdst.add_edge(c_s, c_d, vc=0, weight=0)

        
        for c_s, c_d in D_dstry.cdstry:
            if c_d in G_undir_nxts[c_s]:
                Grdst.add_edge(c_s, c_d, vc=0, weight=1)
            else:
                Grdst.add_edge(c_s, c_d, vc=0, weight=0)

        Grdst_spl_tgt = nx.shortest_path_length(Grdst, target=tgt)
        # print("print about Grdst")
        # print(Grdst.nodes)
        # print(sorted(Grdst.edges(data=True)))
        # print(sorted(Grdst.edges(data="weight")))
        # print("spl: ", Grdst_spl_tgt)
        # print(cdst1y)
        # print(D_dstry.cdstry)

        if pri_mode == "hops":
            for e_i, e_j in sorted(Grdst.edges()):
                Grdst[e_i][e_j]["weight"] = num_nodes_in_G - Grdst_spl_tgt[e_j]
        elif pri_mode == "same":
            for e_i, e_j in sorted(Grdst.edges()):
                Grdst[e_i][e_j]["weight"] = 0


        # print(sorted(Grdst.edges(data="weight")))
        rt_data_dst = [(e_i, tgt, e_j, e_data["vc"], e_data["weight"]) for e_i, e_j, e_data in sorted(Grdst.edges(data=True))]

        rt_data.extend(rt_data_dst)

        # Grdst_spl_tgt = list(nx.shortest_path_length(Grdst, target=tgt).values())
        # print(Grdst_spl_tgt, len(Grdst_spl_tgt), sum(Grdst_spl_tgt))
        # Grdst_spl_tgt = nx.shortest_path_length(Grdst, target=tgt)
        # print(spl_tgt)
        # print(Grdst_spl_tgt)

        Grdsts[tgt] = Grdst
        Grdst_spl_tgts[tgt] = Grdst_spl_tgt

    rt_outf = os.path.join(out_dir, "%d_%d_%d_%s.rt" % (num_nodes, degree, seed, pri_mode))
    with open(rt_outf, 'w') as f:
 
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(rt_data)

    return Grdsts, Grdst_spl_tgts

def calc_hops_ecdg(num_nodes, degree, seed, pri_mode, out_dir, debug, num_vcs, spanning):

    offset = 1

    # nested_dict = lambda: defaultdict(nested_dict)
    # eval_result = nested_dict()

    G_undir = nx.random_regular_graph(degree, num_nodes, seed)
    G = G_undir.to_directed()

    if debug:
        gen_rf_cnc(num_nodes, degree, seed, pri_mode, out_dir, G_undir, G, num_vcs, spanning)
        exit(1)

    udCDG = gen_ud(num_nodes, degree, seed, pri_mode, out_dir, G_undir, G)
    Grdsts, Grdst_spl_tgts = gen_Grdst(num_nodes, degree, seed, pri_mode, out_dir, G_undir, G, spanning)

    lens_sp_result = list()
    lens_ud_result = list()
    lens_Grdst_result = list()

    sum_shortest_ud = 0
    sum_shortest_Grdst = 0

    for tgt in sorted(G.nodes):
        spl_tgt = nx.shortest_path_length(G_undir, target=tgt)
        ud_spl_tgt = nx.shortest_path_length(udCDG, target=(tgt, DST))
        Grdst_spl_tgt = Grdst_spl_tgts[tgt]

        lens_sp = [j for i,j in sorted(list(spl_tgt.items())) if i != tgt]
        lens_ud = [j - offset for i, j in sorted(list(ud_spl_tgt.items())) if i[1] == SRC and i[0] != tgt]
        lens_Grdst = [j for i,j in sorted(list(Grdst_spl_tgt.items())) if i != tgt]
        
        lens_sp_result.extend(lens_sp)
        lens_ud_result.extend(lens_ud)
        lens_Grdst_result.extend(lens_Grdst)

        num_shortest_Grdst = sum(np.equal(lens_sp, lens_Grdst)) 
        num_shortest_ud = sum(np.equal(lens_sp, lens_ud))
        # print(num_shortest_Grdst, num_shortest_ud)
        
        sum_shortest_ud += num_shortest_ud
        sum_shortest_Grdst += num_shortest_Grdst
    # print(lens_sp_result)

    apl_sp = sum(lens_sp_result) / (len(G.nodes) * (len(G.nodes)-1))
    apl_ud = sum(lens_ud_result) / (len(G.nodes) * (len(G.nodes)-1))
    apl_Grdst = sum(lens_Grdst_result) / (len(G.nodes) * (len(G.nodes)-1))
    frac_shortest_ud = sum_shortest_ud / (len(G.nodes) * (len(G.nodes)-1))
    frac_shortest_Grdst = sum_shortest_Grdst / (len(G.nodes) * (len(G.nodes)-1))
    print(apl_sp)
    print(apl_ud)
    print(apl_Grdst)
    print(frac_shortest_ud)
    print(frac_shortest_Grdst)
    
    return apl_sp, apl_ud, apl_Grdst, frac_shortest_ud, frac_shortest_Grdst
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs=3,
                        help='num_nodes, degree, seed')
    parser.add_argument("pri_mode", metavar="P", type=str, default="hops", help="min or hops or same")
    parser.add_argument("outdir", metavar="O", type=str, default="./", help="output directory")
    parser.add_argument('--debug', action='store_true', help="enable debug")
    parser.add_argument("--num_vcs", metavar="V", type=int, default=1, help="number of VCs")
    parser.add_argument("--spanning", metavar="S", type=str, default="Span", help="Span or SparseSpan or FurerSpan")

    args = parser.parse_args()
    print(args)
    # print(args.integers)
    # exit(1)

    num_nodes, degree, seed = args.integers
    pri_mode = args.pri_mode
    out_dir = args.outdir
    debug = args.debug
    num_vcs = args.num_vcs
    spanning = args.spanning

    print("num_nodes=%d, degree=%d, seed=%d" % (num_nodes, degree, seed))
    print("pri_mode:", pri_mode)

    print(calc_hops_ecdg(num_nodes, degree, seed, pri_mode, out_dir, debug, num_vcs, spanning))
    exit(1)
    
