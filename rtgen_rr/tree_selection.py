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

NUM_POOLS = 16
SD_OFFSET = 1

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

    def __init__(self, n, d, s):
        self.n, self.d, self.s = n, d, s
        self.G = nx.random_regular_graph(d, n, s)
        self.Hs = None
        self.coH = None
        self.Hs_sp = None
    
    def comp_graph(self):
        n, d, s = self.n, self.d, self.s
        G = self.G
        G_dir = G.to_directed()

        Hs = [gen_ud(n, d, s, "hops", "./edgefiles", G, G_dir, bfs_root=i) for i in sorted(G.nodes)]

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

        print(largest_cc)
        print(self.Hs[0])

        p = Pool(NUM_POOLS)
        Hs_sp = p.map(calc_Hs_sp, [(H, n) for H in Hs])

        self.Hs_sp = Hs_sp
        print(Hs_sp[0], Hs_sp[0].size)

def get_tree_tmopt(comp_graph: CompGraph, tm: np.ndarray, term: int):
    for i, H_sp in enumerate(comp_graph.Hs_sp):
        weight_sp = np.sum(H_sp * tm) * term
        print(i, ":", weight_sp)

def get_weighted_sps(comp_graph: CompGraph, tm: np.ndarray):
    weight_sps = list()
    for i, H_sp in enumerate(comp_graph.Hs_sp):
        weight_sp = np.sum(H_sp * tm)
        weight_sps.append(weight_sp)
    return weight_sps

def get_comp_graph(n, d, s):
    cg_bin = "comp_graph_{}_{}_{}.bin".format(n, d, s)
    if os.path.exists(cg_bin):
        with open(cg_bin, "rb") as f:
            compGraph = pickle.load(f)
    else:
        compGraph = CompGraph(n, d, s)
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

def gen_splittedTMs_from_trace(trace, num_nodes, num_split):
    """generate Traffic Matrix from trace file

    Args:
        trace (string): trace filepath
        num_nodes (int): # of nodes (must be 2^i)
        num_split (int): # of TMs to be generated (splitted by the same time interval)

    Returns:
        (list), (list of ndarray): list split_terms, num_nodes x num_nodes traffic matrices
    """
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
            print(cycle, src, dst, size, tm_id)
            if src != dst:
                tms[tm_id][src, dst] += 1

    print(tms, [np.sum(tms[i]) for i in range(num_split)])
    return split_terms, tms

class TransitionGraph:
    def __init__(self, trace, num_nodes, degree, seed, num_split):
        self.trace = trace
        self.num_nodes = num_nodes
        self.degree = degree
        self.seed = seed
        self.num_split = num_split

        self.cg = get_comp_graph(num_nodes, degree, seed)

        self.split_terms, self.tms_from_trace =  gen_splittedTMs_from_trace(self.trace, self.num_nodes, self.num_split)

    def gen_tg(self):
        TG_src, TG_dst = (-1, -1), (-2, -2)
        TG = nx.DiGraph()
        TG.add_node(TG_src, weight=0.0)
        TG.add_node(TG_dst, weight=0.0)

        weights = [get_weighted_sps(self.cg, tm) for tm in self.tms_from_trace]

        for term_id in range(self.num_split):
            for n in range(self.num_nodes):
                TG.add_node((term_id, n), weight=weights[term_id][n])

        # TG_src -> T_n in term 0
        for n in range(self.num_nodes):
            tmp_dst = (0, n)
            tmp_weight = TG.nodes[TG_src]["weight"] + TG.nodes[tmp_dst]["weight"]
            TG.add_edge(TG_src, tmp_dst, weight=tmp_weight)

        # term term_id -> term term_id+1
        for term_id in range(self.num_split - 1):
            # tree unchanged between terms
            for n in range(self.num_nodes):
                tmp_src, tmp_dst = (term_id, n), (term_id+1, n)
                tmp_weight = TG.nodes[tmp_src]["weight"] + TG.nodes[tmp_dst]["weight"]
                TG.add_edge(tmp_src, tmp_dst, weight=tmp_weight)

            # tree changed between terms
            for i, j in self.cg.coH.edges:
                tmp_src, tmp_dst = (term_id, i), (term_id+1, j)
                tmp_weight = TG.nodes[tmp_src]["weight"] + TG.nodes[tmp_dst]["weight"]
                TG.add_edge(tmp_src, tmp_dst, weight=tmp_weight)
        
        # T_n in term (self.num_split-1) -> TG_dst
        for n in range(self.num_nodes):
            tmp_src = (self.num_split-1, n)
            tmp_weight = TG.nodes[tmp_src]["weight"] + TG.nodes[TG_dst]["weight"]
            TG.add_edge(tmp_src, TG_dst, weight=tmp_weight)

        self.TG = TG

if __name__ == "__main__":

    import doctest
    doctest.testmod()

    n, d, s = 64, 3, 1
    print(gen_TM_from_tf(dst_shuffle, n))
    print(np.nonzero(gen_TM_from_tf(dst_shuffle, n)))
    print(gen_TM_uniform(n))

    compGraph = get_comp_graph(n, d, s)
    get_tree_tmopt(compGraph, gen_TM_uniform(n), 1000)
    get_tree_tmopt(compGraph, gen_TM_from_tf(dst_shuffle, n), 1000)

    # gen_trace(0.1, ["uniform", "transpose"], [1000,2000], 256, seed=1)
    gen_trace(0.1, ["uniform", "transpose"], [1000,2000], 256)
    split_terms, tms_from_trace = gen_splittedTMs_from_trace("0.1000_uniform-1000_transpose-2000_1_256.tr", 256, 2)
    print(split_terms)
    print(tms_from_trace)

    transition_graph = TransitionGraph("0.1000_uniform-10_transpose-20_1_64.tr", 64, 3, 1, 2)
    transition_graph.gen_tg()
    print(transition_graph.TG.nodes)
    print(transition_graph.TG.edges(data=True))
