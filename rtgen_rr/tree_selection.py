import math
import numpy as np
import itertools as it
import networkx as nx
import os
import csv
from validate_ecdg2_woacy_copy import gen_ud
from scipy.sparse.csgraph import shortest_path
from multiprocessing import Pool
import pickle
import random
import subprocess
import glob

from collections import defaultdict
def rec_dd():
    return defaultdict(rec_dd)

"""
    Note: G is undirected through this file
    (unless spanning_duato3.py and validate_ecdg2_woacy_copy.py)
"""

NUM_POOLS = 16
SD_OFFSET = 1

SRC = -2
DST = -1

def dst_uniform(src: int, num_nodes: int) -> int:
    dst = src
    while src == dst:
        dst = random.choice(range(num_nodes))
    return dst

def dst_transpose(src: int, num_nodes: int) -> int:
    """destination for transpose traffic.

    Example: 
        0b0010 -> 0b1000 (2 -> 8) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^(2*i))

    Returns:
        (int): destination node id

    >>> dst_transpose(2, 16)
    8
    >>> dst_transpose(0b101010, 64)
    21
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes or lg % 2 != 0:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[(lg//2):] + src_bit[:(lg//2)], 2)

def dst_reverse(src: int, num_nodes: int) -> int:
    """destination for reverse traffic.

    Example: 
        0b0010 -> 0b0100 (2 -> 4) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (int): destination node id

    >>> dst_reverse(2, 16)
    4
    >>> dst_reverse(0b001110, 64)
    28
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[::-1], 2)

def dst_shuffle(src: int, num_nodes: int) -> int:
    """destination for shuffle traffic.

    Example: 
        0b1110 -> 0b1101 (14 -> 13) for 16 nodes.

    Args:
        src (int): source node id
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (int): destination node id

    >>> dst_shuffle(14, 16)
    13
    >>> dst_shuffle(0b101110, 64)
    29
    """

    lg = int(round(math.log2(num_nodes)))

    if 1 << lg != num_nodes:
        print("Error: The number of nodes to be an even power of two!")
        exit(1)

    src_bit = "{:0{width}b}".format(src, width=lg)
    return int(src_bit[1:] + src_bit[:1], 2)

def gen_TM_from_tf(tf, num_nodes):
    """generate Traffic Matrix from traffic function

    Args:
        tf (Callable): traffic function (src: int, num_nodes: int -> int)
        num_nodes (int): # of nodes (must be 2^i)

    Returns:
        (ndarray): num_nodes x num_nodes traffic matrix 
    """
    
    tm = np.zeros((num_nodes, num_nodes))
    for src in range(num_nodes):
        dst = tf(src, num_nodes)
        if src != dst:
            tm[src, dst] = 1

    return tm

def gen_TM_uniform(num_nodes):
    return (np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)) / (num_nodes - 1)

def calc_coH_edge(inputs):
    n, H_i, H_j = inputs
    H_int = H_i.multiply(H_j)
    H_int_scsp = shortest_path(H_int)
    H_int_scsp_sd = H_int_scsp[0:n,n:2*n]
    num_posinf = sum(np.isposinf(H_int_scsp_sd))
    return sum(num_posinf) == 0

def Hs_iter(n, Hs, G):
    for i, j in it.combinations(sorted(G.nodes), 2):
        yield n, Hs[i], Hs[j]

def calc_Hs_sp(inputs):
    H, n = inputs
    return shortest_path(H)[0:n,n:2*n] - SD_OFFSET

def gen_intersect_table(H1, H2, G):
    """
    Note: G is undirected
    """
    H = nx.intersection(H1, H2)
    result_table = list()
    for dst in G.nodes():
        dst_channel = (dst, DST)
        H_spl_tgt = nx.shortest_path_length(H, target=dst_channel)
        for in_edge_H in H.nodes():
            for out_edge_H in H[in_edge_H]:
                if out_edge_H in H_spl_tgt:
                    if out_edge_H[1] == DST:
                        continue
                    else:
                        result_table.append((in_edge_H[0], out_edge_H[0], dst, out_edge_H[1], H_spl_tgt[out_edge_H]))
    return result_table

def gen_intersect_ud(n, d, s, pri_mode, G, Hs, i, j):
    """
    Note: G is undirected
    """

    out_dir = "edgefiles"
    is_rt_outf = os.path.join(out_dir, "%d_%d_%d_int_%d-%d_%s_ud.rt" % (n, d, s, i, j, pri_mode))

    print("out: %s" % is_rt_outf)
    
    if os.path.exists(is_rt_outf) and os.path.getsize(is_rt_outf) > 0:
        return None

    intersect_table = gen_intersect_table(Hs[i], Hs[j], G)
    num_nodes_in_G = len(G)

    isrt_data = list()
    # pn, s, d, n, hops → pn, pv, s, d, n, v, pri
    for pn, s, d, n, hops in intersect_table:
        if pri_mode == "hops":
            isrt_data.append((pn, 0, s, d, n, 0, num_nodes_in_G - hops))
        elif pri_mode == "same":
            isrt_data.append((pn, 0, s, d, n, 0, 0))
    
    if not (os.path.exists(is_rt_outf) and os.path.getsize(is_rt_outf) > 0):
        with open(is_rt_outf, 'w') as f:
     
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(isrt_data)
    pass
class CompGraph:
    """compatibility graph

    Attributes:
        n (int): # of nodes
        d (int): degree
        s (int): random seed
        G (nx.Graph): random regular graph
        Hs (list of nx.DiGraph): channel dependency graphs
        coH (nx.Graph): compatibility graph among CDGs
        Hs_sp (list of ndarrays): shortest path length for CDGs
    """

    def __init__(self, n, d, s, edgefile=None):
        self.n, self.d, self.s = n, d, s
        if edgefile == None:
            self.G = nx.random_regular_graph(d, n, s)
            self.edgefile = None
        else:
            self.G = nx.read_edgelist(edgefile, nodetype=int)
            # print(self.G.nodes, self.G.edges)
            self.edgefile = edgefile
        self.Hs = None
        self.coH = None
        self.Hs_sp = None
    
    def comp_graph(self):
        n, d, s = self.n, self.d, self.s
        G = self.G
        G_dir = G.to_directed()

        if self.edgefile == None:
            Hs = [gen_ud(n, d, s, "hops", "./edgefiles", G, G_dir, bfs_root=i) for i in sorted(G.nodes)]
        else:
            Hs = [gen_ud(n, d, s, "hops", "./edgefiles", G, G_dir, bfs_root=i, edgefile=self.edgefile) for i in sorted(G.nodes)]

        self.Hs = Hs

        H_sorted_nodes = sorted(Hs[0].nodes, key=lambda x: (x[1],x[0]))
        Hs = [nx.to_scipy_sparse_matrix(H, nodelist=H_sorted_nodes) for H in Hs]
            
        coH = nx.Graph()
        coH.add_nodes_from(sorted(G.nodes))

        p = Pool(NUM_POOLS)
        result = p.map(calc_coH_edge, Hs_iter(n, Hs, G))
        coH_edges_p = [(i,j) for idx, (i, j) in enumerate(it.combinations(sorted(G.nodes), 2)) if result[idx]]

        coH.add_edges_from(coH_edges_p)
        self.coH = coH

        largest_cc = max(nx.connected_components(coH), key=len)

        # print(largest_cc)
        # print(self.Hs[0])

        p = Pool(NUM_POOLS)
        Hs_sp = p.map(calc_Hs_sp, [(H, n) for H in Hs])

        self.Hs_sp = Hs_sp
        # print(Hs_sp[0], Hs_sp[0].size)

def get_tree_tmopt(comp_graph: CompGraph, tm: np.ndarray, term: int):
    for i, H_sp in enumerate(comp_graph.Hs_sp):
        weight_sp = np.sum(H_sp * tm) * term
        # print(i, ":", weight_sp)

def get_weighted_sps(comp_graph: CompGraph, tm: np.ndarray):
    weight_sps = list()
    for i, H_sp in enumerate(comp_graph.Hs_sp):
        weight_sp = np.sum(H_sp * tm)
        weight_sps.append(weight_sp)
    return weight_sps

def get_comp_graph(n, d, s, edgefile=None, load_pickle=True):
    if edgefile == None:
        cg_bin = os.path.join("binfiles", "comp_graph_{}_{}_{}.bin".format(n, d, s))
    else:
        cg_bin = os.path.join("binfiles", "comp_graph_{}.bin".format(edgefile))
    if load_pickle and os.path.exists(cg_bin):
        with open(cg_bin, "rb") as f:
            compGraph = pickle.load(f)
    else:
        compGraph = CompGraph(n, d, s, edgefile)
        compGraph.comp_graph()

        with open(cg_bin, "wb") as f:
            pickle.dump(compGraph, f)

    return compGraph

def gen_trace(ij_rate, traffics, end_cycles, num_nodes, seed=None):
    """generate tracefile

    Args:
        ij_rate (float): injection rate between 0.0 and 1.0
        traffics (list of string): traffics either in "uniform", "transpose", "reverse", "shuffle"
        end_cycles (list of int): end cycle for every traffic
        num_nodes (int): # of nodes
        seed (int, optional): random seed

    Returns:
        None ("IJRATE_TF1-EC1_TF2-EC2_...[_SEED]_NUMNODES.tr" generated)
    Usage:
        gen_trace(0.1, ["uniform", "transpose"], [10,20], 64, seed=1)
    
    """
    if len(traffics) != 0 and len(traffics) != len(end_cycles):
        print("invalid length of traffic or end_cycles")
        exit(1)
    for traffic in traffics:
        if traffic not in ["uniform", "transpose", "reverse", "shuffle"]:
            print("invalid traffic name: ", traffic)
            exit(1)
    if ij_rate < 0 or ij_rate >= 1:
        print("invalid ij_rate:", ij_rate)
        exit(1)
    if sorted(end_cycles) != end_cycles:
        print("invalid end_cycles:", end_cycles)
        exit(1)

    if seed != None:
        random.seed(seed)

    outf_tr = "{:.4f}".format(ij_rate)
    for traffic, end_cycle in zip(traffics, end_cycles):
        outf_tr += "_{}-{}".format(traffic, end_cycle)
    if seed != None:
        outf_tr += "_{}".format(seed)
    outf_tr += "_{}.tr".format(num_nodes)
    outf_tr_tmp = outf_tr + ".tmp"
    outf_tr_head_tmp = outf_tr + ".head.tmp"
    print(outf_tr, outf_tr_tmp)

    start_cycle = 1
    num_packets = 0
    with open(outf_tr_tmp, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for traffic, end_cycle in zip(traffics, end_cycles):
            tf = eval("dst_" + traffic)

            for time in range(start_cycle, end_cycle+1):
                for src in range(num_nodes):
                    if random.random() < ij_rate:
                        dst = tf(src, num_nodes)
                        if src != dst:
                            writer.writerow([time, src, dst, 1])
                            num_packets += 1

            start_cycle = time + 1

    with open(outf_tr_head_tmp, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow([num_packets])

    with open(outf_tr, "w") as f:
        subprocess.run(["cat", outf_tr_head_tmp, outf_tr_tmp], stdout=f)

    os.remove(outf_tr_tmp)
    os.remove(outf_tr_head_tmp)

class SplittedTMs:
    def __init__(self, trace, num_nodes, num_split) -> None:
        self.trace = trace
        self.num_nodes = num_nodes
        self.num_split = num_split
        self.split_terms, self.tms = None, None

    def splittedTMs_from_trace(self):
        trace, num_nodes, num_split = self.trace, self.num_nodes, self.num_split

        tms = [np.zeros((num_nodes, num_nodes)) for _ in range(num_split)]

        trace_header = subprocess.run(["head", "-n1", trace], stdout=subprocess.PIPE, text=True)
        num_packets = int(trace_header.stdout.strip().split()[0])
        # print(num_packets)

        trace_tail = subprocess.run(["tail", "-n1", trace], stdout=subprocess.PIPE, text=True)
        num_cycles = int(trace_tail.stdout.strip().split()[0])
        # print(num_cycles)

        split_terms = [1] + [int(round(num_cycles / num_split * (i+1))) + 1 for i in range(num_split)]
        # print(split_terms)

        def _get_TMid(terms, cycle):
            for i, t in enumerate(terms):
                if t > cycle:
                    return i-1

        with open(trace, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            is_head = True
            for row in reader:
                if is_head:
                    is_head = False
                    continue
                cycle, src, dst, size = list(map(int, row))
                tm_id = _get_TMid(split_terms, cycle)
                # print(cycle, src, dst, size, tm_id)
                if src != dst:
                    # tms[tm_id][src, dst] += 1
                    tms[tm_id][src, dst] += size

        # print(tms, [np.sum(tms[i]) for i in range(num_split)])

        self.split_terms, self.tms = split_terms, tms

def gen_splittedTMs_from_trace(trace, num_nodes, num_split, load_pickle=True):
    """generate Traffic Matrix from trace file

    Args:
        trace (string): trace filepath
        num_nodes (int): # of nodes (must be 2^i)
        num_split (int): # of TMs to be generated (splitted by the same time interval)
        load_pickle (optional, bool, default=True): load pickle object if exists

    Returns:
        (list), (list of ndarray): list split_terms, num_nodes x num_nodes traffic matrices
    """

    splitted_tms_bin = os.path.join("binfiles", "splittedTMs_{}_{}_{}.bin".format(os.path.basename(trace), num_nodes, num_split))

    if load_pickle and os.path.exists(splitted_tms_bin):
        with open(splitted_tms_bin, "rb") as f:
            splittedTMs = pickle.load(f)
    else:
        splittedTMs = SplittedTMs(trace, num_nodes, num_split)
        splittedTMs.splittedTMs_from_trace()

        with open(splitted_tms_bin, "wb") as f:
            pickle.dump(splittedTMs, f)

    return splittedTMs.split_terms, splittedTMs.tms

class TransitionGraph:
    TG_SRC = (-1, -1)
    TG_DST = (-2, -2)
    def __init__(self, trace, num_nodes, degree, seed, num_split, edgefile=None, load_pickle=True, trans_margin=1):
        self.trace = trace
        self.num_nodes = num_nodes
        self.degree = degree
        self.seed = seed
        self.num_split = num_split
        self.edgefile = edgefile
        self.trans_margin = trans_margin

        self.cg = get_comp_graph(num_nodes, degree, seed, edgefile, load_pickle)

        self.split_terms, self.tms_from_trace =  gen_splittedTMs_from_trace(self.trace, self.num_nodes, self.num_split, load_pickle)

    def gen_Rint(self):
        coH = self.cg.coH
        for i, j in coH.edges():
            i, j = sorted([i,j])
            gen_intersect_ud(self.num_nodes, self.degree, self.seed, "hops", self.cg.G, self.cg.Hs, i, j)

    def gen_tg(self):
        TG = nx.DiGraph()
        TG.add_node(self.TG_SRC, weight=0.0)
        TG.add_node(self.TG_DST, weight=0.0)

        weights = [get_weighted_sps(self.cg, tm) for tm in self.tms_from_trace]

        for term_id in range(self.num_split):
            for n in range(self.num_nodes):
                TG.add_node((term_id, n), weight=weights[term_id][n])

        # TG_SRC -> T_n in term 0
        for n in range(self.num_nodes):
            tmp_dst = (0, n)
            tmp_weight = TG.nodes[self.TG_SRC]["weight"] + TG.nodes[tmp_dst]["weight"]
            TG.add_edge(self.TG_SRC, tmp_dst, weight=tmp_weight)

        # term term_id -> term term_id+1
        for term_id in range(self.num_split - 1):
            # tree unchanged between terms
            for n in range(self.num_nodes):
                tmp_src, tmp_dst = (term_id, n), (term_id+1, n)
                tmp_weight = TG.nodes[tmp_src]["weight"] + TG.nodes[tmp_dst]["weight"]
                TG.add_edge(tmp_src, tmp_dst, weight=tmp_weight)

            # tree changed between terms
            # NOTE: self.cg.coH is undirected graph
            for i_undir, j_undir in self.cg.coH.edges:
                for i, j in ((i_undir, j_undir), (j_undir, i_undir)):
                    tmp_src, tmp_dst = (term_id, i), (term_id+1, j)
                    tmp_weight = TG.nodes[tmp_src]["weight"] + TG.nodes[tmp_dst]["weight"]
                    # tmp_weight = TG.nodes[tmp_src]["weight"] * 100 + TG.nodes[tmp_dst]["weight"] * 100
                    TG.add_edge(tmp_src, tmp_dst, weight=tmp_weight)
        
        # T_n in term (self.num_split-1) -> TG_DST
        for n in range(self.num_nodes):
            tmp_src = (self.num_split-1, n)
            tmp_weight = TG.nodes[tmp_src]["weight"] + TG.nodes[self.TG_DST]["weight"]
            TG.add_edge(tmp_src, self.TG_DST, weight=tmp_weight)

        self.TG = TG

    def shortest_transition(self):
        self.st = nx.shortest_path(self.TG, self.TG_SRC, self.TG_DST, weight="weight")
        return self.st

    def gen_txt(self):
        # trans_margin = 100
        # trans_margin = 1
        trans_margin = self.trans_margin

        trace_tail = subprocess.run(["tail", "-n1", self.trace], stdout=subprocess.PIPE, text=True)
        num_cycles = int(trace_tail.stdout.strip().split()[0])
        print(num_cycles)

        split_terms = [int(round(num_cycles / self.num_split * (i+1))) + 1 for i in range(self.num_split)]
        print(split_terms, len(split_terms))

        st_roots = [j for i,j in self.st[1:-1]]
        print(st_roots, len(st_roots))

        # term_roots = list()
        # tmp_split_term, tmp_st_root = 0, -1
        # prev_split_term, prev_st_root = 0, -1
        # for split_term, st_root in zip(split_terms, st_roots):
        #     if tmp_st_root == -1:
        #         tmp_split_term = split_term
        #         tmp_st_root = st_root
        #     elif st_root == tmp_st_root:
        #         tmp_split_term += (split_term - prev_split_term)
        #     else:
        #         term_roots.append((tmp_split_term, tmp_st_root))
        #         tmp_split_term = (split_term - prev_split_term)
        #         tmp_st_root = st_root
        #     prev_split_term, prev_st_root = split_term, st_root

        term_roots = list()
        tmp_split_term, tmp_st_root = 0, -1
        prev_split_term, prev_st_root = 0, -1
        for split_term, st_root in zip(split_terms, st_roots):
            if tmp_st_root == -1:
                tmp_split_term = split_term
                tmp_st_root = st_root
            elif st_root == tmp_st_root:
                tmp_split_term = split_term
            else:
                term_roots.append((tmp_split_term, tmp_st_root))
                tmp_split_term = split_term
                tmp_st_root = st_root
            prev_split_term, prev_st_root = split_term, st_root

        term_roots.append((-1, st_root))
        print(term_roots)
        print(tmp_split_term, tmp_st_root)
        ud_rt_outf = "{}_{}_{}_%s_{}_ud.rt".format(self.num_nodes, self.degree, self.seed, "hops")
        print(ud_rt_outf)

        outf_txt = os.path.join("edgefiles", "%s_%d_%d_%d_%d.txt" % (os.path.basename(self.trace), self.num_split, self.degree, self.seed, self.trans_margin))
        with open(outf_txt, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            for id in range(len(term_roots)-1):
                term, root, nxt_root = term_roots[id][0], term_roots[id][1], term_roots[id+1][1]
                writer.writerow([ud_rt_outf % str(root), term])
                writer.writerow([ud_rt_outf % ("int_%d-%d" % tuple(sorted([root, nxt_root]))), trans_margin])
                print("%s %d" % (ud_rt_outf % str(root), term))
                print("%s %d" % (ud_rt_outf % ("int_%d-%d" % tuple(sorted([root, nxt_root]))), trans_margin))
                # print(term, root)

            term, root = term_roots[-1]
            writer.writerow([ud_rt_outf % str(root), -1])
            # print(term, root)

            # writer.writerow([num_packets])

        # for id in range(len(term_roots)-1):
        #     term, root, nxt_root = term_roots[id][0], term_roots[id][1], term_roots[id+1][1]
        #     print("%s %d" % (ud_rt_outf % str(root), term))
        #     print("%s %d" % (ud_rt_outf % ("int_%d-%d" % tuple(sorted([root, nxt_root]))), trans_margin))
        #     # print(term, root)

        # term, root = term_roots[-1]
        # print("%s %d" % (ud_rt_outf % str(root), -1))
        # # print(term, root)

    def path_weight(self, path):
        return nx.path_weight(self.TG, path, weight="weight")

if __name__ == "__main__":

    import doctest
    doctest.testmod()
    
    trfiles = sorted(glob.glob("trfiles/*.tr"))
    # num_splits = [1,2,4,8,16,32,64]
    num_splits = [2**i for i in range(14)]

    input_edges = [
        ("n64d4k4l286.20150726-0h9mgg.edges",64,4,1),
        ("n64d8k3l192.20160529-pbosi4.edges",64,8,1),
        ("n64d16k2l174.20150625-ugkdx5.edges",64,16,1),
    ]

    print(input_edges)
    num_node = 64
    degrees = [4,8]
    num_seeds = range(10)

    split_ones = rec_dd()

    for trfile, num_split, degree, num_seed in it.product(trfiles, num_splits, degrees, num_seeds):
    # for trfile, num_split, (edgefile, num_node, degree, num_seed) in it.product(trfiles, num_splits, input_edges):
        transition_graph2 = TransitionGraph(trfile, num_node, degree, num_seed, num_split)
        # transition_graph2 = TransitionGraph(trfile, num_node, degree, num_seed, num_split, edgefile=edgefile)
        transition_graph2.gen_tg()
        # print(transition_graph2.TG.nodes(data=True))
        # print(transition_graph2.TG.edges(data=True))
        st = transition_graph2.shortest_transition()
        # print(st)
        set_st = set([root for term, root in st])
        
        if num_split == 1 or len(set_st) > 3:
            tmp_path_weight = transition_graph2.path_weight(st)
            if num_split == 1:
                split_ones[trfile][degree][num_seed] = tmp_path_weight
            # print(trfile, num_split, edgefile)
            print(trfile, num_split, degree, num_seed)
            print(set_st)
            print(tmp_path_weight)
            if num_split > 1:
                print(tmp_path_weight / split_ones[trfile][degree][num_seed])
    exit(1)


    # n, d, s = 64, 3, 1
    # print(gen_TM_from_tf(dst_shuffle, n))
    # print(np.nonzero(gen_TM_from_tf(dst_shuffle, n)))
    # print(gen_TM_uniform(n))

    # compGraph = get_comp_graph(n, d, s)
    # get_tree_tmopt(compGraph, gen_TM_uniform(n), 1000)
    # get_tree_tmopt(compGraph, gen_TM_from_tf(dst_shuffle, n), 1000)

    # # gen_trace(0.1, ["uniform", "transpose"], [1000,2000], 256, seed=1)
    # gen_trace(0.1, ["uniform", "transpose"], [1000,2000], 256)
    # split_terms, tms_from_trace = gen_splittedTMs_from_trace("0.1000_uniform-1000_transpose-2000_1_256.tr", 256, 2)
    # print(split_terms)
    # print(tms_from_trace)

    # transition_graph = TransitionGraph("0.1000_uniform-10_transpose-20_1_64.tr", 64, 3, 1, 2)
    # transition_graph.gen_tg()
    # print(transition_graph.TG.nodes(data=True))
    # print(transition_graph.TG.edges(data=True))
    # print(transition_graph.shortest_transition())

    # # trace_csv/crossbar_64_bt.W.64_trace_1.00e+09_64_620085_62590300.tr

    transition_graph2 = TransitionGraph("crossbar_64_is.W.64_trace_1.00e09_8_53946_4909650.tr", 64, 4, 1, 8)
    transition_graph2.gen_tg()
    # print(transition_graph2.TG.nodes(data=True))
    # print(transition_graph2.TG.edges(data=True))
    st = transition_graph2.shortest_transition()
    # print(st)
    print(set([root for term, root in st]))

    # transition_graph2 = TransitionGraph("crossbar_64_cg.S.64_trace_1.00e+09_64_267003_32932100.tr", 64, 4, 1, 10000, "n64d4k4l286.20150726-0h9mgg.edges")
    # transition_graph2.gen_tg()
    # # print(transition_graph2.TG.nodes(data=True))
    # # print(transition_graph2.TG.edges(data=True))
    # st = transition_graph2.shortest_transition()
    # print(st)
    # print(set([root for term, root in st]))
