# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:56:35 2021

@author: ryuut
"""

# import csv
# import sys
import networkx as nx
# import numpy as np
# import scipy
# from scipy.sparse.csgraph import shortest_path as scsp
# from scipy.sparse.csgraph import shortest_path
# import itertools as it
# import collections
# from collections import deque
# import random
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib

# from collections import defaultdict as dd

from sys import exit

from collections import defaultdict
nested_dict = lambda: defaultdict(nested_dict)

class CC:
    def __init__(self, G):
        self.F = list(nx.connected_components(G))
        self.cc_id = dict()
        for i, nodes in enumerate(self.F):
            for n in nodes:
                self.cc_id[n] = i
            pass
    
    def _update_ccid(self):
        for i, nodes in enumerate(self.F):
            for n in nodes:
                self.cc_id[n] = i
            pass
        
    def is_in_diff_cc(self, u, v):
        return u in self.cc_id and v in self.cc_id and self.cc_id[u] != self.cc_id[v]
    
    def update_cc(self, C, new_vertices, T):
        to_combine = set()
        left_uncombined = set(range(len(self.F)))
        for node_in_C in C:
            if node_in_C in self.cc_id:
                to_combine.add(self.cc_id[node_in_C])
                left_uncombined.discard(self.cc_id[node_in_C])
                
        for nv in new_vertices:
            for nbr in T[nv]:
                if nbr in self.cc_id:
                    to_combine.add(self.cc_id[nbr])
                    left_uncombined.discard(self.cc_id[nbr])
                    
        combined_cc = set()
        for i in to_combine:
            combined_cc |= self.F[i]
        combined_cc |= set(new_vertices)
        
        new_F = list()
        new_F.append(combined_cc)
        for lu in left_uncombined:
            new_F.append(self.F[lu])
        
        # print("new_F", new_F)
        self.F = new_F
        self._update_ccid()
        # print("self.F", self.F)
        pass
    

def gen_marks(T, G, k):
    marks = dict()
    is_degree_k = dict()
    is_degree_km1 = dict()
    for n in G.nodes:
        if T.degree[n] == k:# or T.degree[n] == (k-1):
            marks[n] = False
            is_degree_k[n] = True
            is_degree_km1[n] = False
        elif T.degree[n] == (k-1):
            marks[n] = False
            is_degree_k[n] = False
            is_degree_km1[n] = True
        else:
            marks[n] = True
            is_degree_k[n] = False
            is_degree_km1[n] = False
            
    return marks, is_degree_k, is_degree_km1

def all_k_bad(marks, is_degree_k):
    for key, val in is_degree_k.items():
        if val and marks[key]:
            return False
    return True
    pass
              
def find_edge(F, G):
    for u, v in G.edges:
        if F.is_in_diff_cc(u, v):
            return True, u, v
    return False, None, None

def draw_dot(T):
    T_root = nx.center(T)[0]
    
    plt.figure(figsize=(16, 8))  # image is 8 x 8 inches
    ax = plt.axis('off')

    pos = nx.nx_pydot.pydot_layout(T, prog='dot', root=T_root)
    # pos = nx.nx_pydot.pydot_layout(T, prog='dot')
    # print(pos)
    nx.draw_networkx_nodes(T, pos, node_size=400, node_color="lightblue")
    nx.draw_networkx_labels(T, pos, font_family="serif", font_size=15)
    nx.draw_networkx_edges(T, pos)
    
    plt.show()
    plt.clf()

def min_deg_spanning(G, T_orig):
    T = T_orig.copy()
    
    k = max(dict(T.degree).values())
    print("T_orig:")
    print("max_degree:", k)
    
    # print("T")
    # print(T.edges)
    print("centers:", nx.center(T))
    # print("degrees")
    # print(sorted(T.degree, key=lambda x: x[1]))
    
    # draw_dot(T)

    while True:
        marks, is_degree_k, is_degree_km1 = gen_marks(T, G, k)
        
        bad_vertices = sorted([v for v in marks if marks[v] == False])
        forest = T.copy()
        forest.remove_nodes_from(bad_vertices) #.connected_components()
        F = CC(forest)
        
        # edge_found, edge_u, edge_v = find_edge(F, G)
    
        oblig = nested_dict()
        num_iter = 0
        k_marked = False
        k_marked_nodes = list()
        
        marked_iter = dict()
        for n in G.nodes:
            if marks[n]:
                marked_iter[n] = -1
        
        while find_edge(F, G)[0] and all_k_bad(marks, is_degree_k):
            edge_found, edge_u, edge_v = find_edge(F, G)
            
            # print("CC:", list(F.F))
            # print("num_iter:", num_iter)
            # print(edge_u, edge_v)
            if len(nx.cycle_basis(T)) != 0:
                print("error")
                exit(1)
            T_added = T.copy()
            T_added.add_edge(edge_u, edge_v)
            # print(sorted(T_added.edges))
            cycle = nx.find_cycle(T_added)
            cycle_nodes = [edge[0] for edge in cycle]
            # print("cycle_nodes:", cycle_nodes)
            
            bad_in_cycle = [node for node in cycle_nodes if not marks[node]]
            for node in bad_in_cycle:
                marks[node] = True
                marked_iter[node] = num_iter

                if is_degree_k[node]:
                    k_marked = True
                    k_marked_nodes.append(node)
    
            F.update_cc(cycle_nodes, bad_in_cycle, T)
            

            # km1_in_cycle = [node for node in cycle_nodes if is_degree_km1[node] == True]
            k_in_cycle = [node for node in cycle_nodes if is_degree_k[node] == True]
            # print("km1_in_cycle:", km1_in_cycle)
            # print("k_in_cycle:", k_in_cycle)
            for cn_idx, cn in enumerate(cycle_nodes):
                if is_degree_km1[cn]:
                    nbr_p1 = cycle_nodes[(cn_idx+1)%len(cycle_nodes)]
                    nbr_m1 = cycle_nodes[(cn_idx-1)%len(cycle_nodes)]
                    # print(cn, nbr_p1, nbr_m1)
                    
                    if (cn == edge_u and nbr_p1 == edge_v) or (nbr_p1 == edge_u and cn == edge_v):
                        oblig[num_iter][cn] = (nbr_m1, (edge_u, edge_v))
                    else:
                        oblig[num_iter][cn] = (nbr_p1, (edge_u, edge_v))
                        
                
            # for km1 in km1_in_cycle:
            #     oblig[num_iter][km1] = (cycle_nodes[(cycle_nodes.index(km1)+1)%len(cycle_nodes)], (edge_u, edge_v))

            num_iter += 1
            # edge_found, edge_u, edge_v = find_edge(F, G)
            pass
        
        # print("CC2:", list(nx.connected_components(forest)))
        # print("num_iter:", num_iter)
        # print("k_marked:", k_marked)
        # print("k_marked_nodes:", k_marked_nodes)
        # print("oblig:", oblig)
        # print("marked_iter:", sorted(marked_iter.items()))

        num_iter -= 1
        if k_marked:
            propagate = [(num_iter, (k_in_cycle[0], cycle_nodes[(cycle_nodes.index(k_in_cycle[0])+1)%len(cycle_nodes)]), (edge_u, edge_v))]
            while propagate:
                # print("propagate:", propagate)
                if len(nx.cycle_basis(T)) != 0:
                    print("error2")
                    exit(1)
                prop_iter, (rem_s, rem_d), (add_s, add_d) = propagate.pop(0)
                T.remove_edge(rem_s, rem_d)
                T.add_edge(add_s, add_d)
                to_add = set([add_s, add_d])- set([rem_s, rem_d])
                for ta in to_add:
                    # print("ta:", ta)
                    # if ta in oblig[prop_iter-1]:
                    if is_degree_km1[ta]:
                        ta_marked_iter = marked_iter[ta]
                        # propagate.append((prop_iter-1, (ta, oblig[prop_iter-1][ta][0]), oblig[prop_iter-1][ta][1]))
                        propagate.append((prop_iter-1, (ta, oblig[ta_marked_iter][ta][0]), oblig[ta_marked_iter][ta][1]))

            k = max(dict(T.degree).values())
            continue
        else:
            break

        # if not edge_found:
        #     break
            
    # print(oblig)
    # print(k_marked)
    # print(sorted(T.edges))

    # print(sorted(T.degree))
    print("new_T:")
    k = max(dict(T.degree).values())
    print("max_degree:", k)
    print("centers:", nx.center(T))
    
    return T

    # draw_dot(T)
        
if __name__ == "__main__":
    # G = nx.random_regular_graph(d=12, n=64, seed=18)
    G = nx.random_regular_graph(d=12, n=64, seed=18)
    T = nx.minimum_spanning_tree(G)
    draw_dot(T)
    
    new_T = min_deg_spanning(G, T)
    draw_dot(new_T)
