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

from tree_selection import *

if __name__ == "__main__":

    import doctest
    doctest.testmod()
    
    trfiles = ["trfiles/crossbar_64_is.W.64_trace_1.00e09_8_53946_4909650.tr"]

    num_splits = [1,2,4,8,16,32,64,128,256]
    # num_splits = [2**i for i in range(14)]
    # num_splits = [2**i for i in range(10)]
    # num_splits = [1,8192]

    num_node = 64

    # degrees = [4,8]
    degrees = [4]

    # num_seeds = range(10)
    num_seeds = [1]

    split_ones = rec_dd()

    for trfile, num_split, degree, num_seed in it.product(trfiles, num_splits, degrees, num_seeds):
    # for trfile, num_split, (edgefile, num_node, degree, num_seed) in it.product(trfiles, num_splits, input_edges):
        transition_graph2 = TransitionGraph(trfile, num_node, degree, num_seed, num_split)
        # transition_graph2 = TransitionGraph(trfile, num_node, degree, num_seed, num_split, edgefile=edgefile)
        transition_graph2.gen_Rint()
        transition_graph2.gen_tg()
        # print(transition_graph2.TG.nodes(data=True))
        # print(transition_graph2.TG.edges(data=True))
        st = transition_graph2.shortest_transition()
        print(st)
        print(transition_graph2.gen_txt())
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
