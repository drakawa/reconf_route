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

NUM_POOLS = 16
SD_OFFSET = 1

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
    return shortest_path(H)[0:n,n:2*n] - 1

class CompGraph:
    """compatibility graph

    Attributes:
        n (int): # of nodes
        d (int): degree
        s (int): random seed
        G (nx.Graph): random regular graph
        Hs (list of nx.DiGraph): channel dependency graphs
        coH (nx.Graph): compatibility graph among CDGs
        Hs_sp (list of spmat): shortest path length for CDGs
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
    for H_sp in comp_graph.Hs_sp:
        weight_sp = np.sum(H_sp * tm) * term
        print(weight_sp)

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

if __name__ == "__main__":

    import doctest
    doctest.testmod()

    n, d, s = 64, 8, 1
    print(gen_TM_from_tf(dst_shuffle, n))
    print(np.nonzero(gen_TM_from_tf(dst_shuffle, n)))
    print(gen_TM_uniform(n))

    compGraph = get_comp_graph(n, d, s)
    get_tree_tmopt(compGraph, gen_TM_uniform(n), 1000)
    get_tree_tmopt(compGraph, gen_TM_from_tf(dst_shuffle, n), 1000)
