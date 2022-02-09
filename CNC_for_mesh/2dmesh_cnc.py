import argparse
import networkx as nx
import csv
import os
import itertools as it

SRC = -2
DST = -1

class Gen2DMesh:
    def __init__(self, x, y, vc):
        self.x, self.y, self.vc = x, y, vc

        self.tmp_G = nx.grid_2d_graph(x, y)
        self.tmp_G_mapping = dict([(e, i) for i, e in enumerate(list(sorted(self.tmp_G.nodes())))])
        # print(self.tmp_G_mapping)
        self.G = nx.relabel_nodes(self.tmp_G, self.tmp_G_mapping)
        # print(self.G.nodes, self.G.edges)

        tp_outf = os.path.join("./", "2dmesh_%d_%d.tp" % (x, y))
        with open(tp_outf, 'w') as f: 
            writer = csv.writer(f, delimiter=" ")
            for e_s, e_d in sorted(self.G.edges):
                # router 2 router 11 1
                writer.writerow(["router", e_s, "router", e_d, 1])
            for node in sorted(self.G.nodes):
                # node 9 router 9
                writer.writerow(["node", node, "router", node])
        # self.tmp_G_spl = nx.shortest_path_length(self.tmp_G)
        self.tmp_G_spl = dict(nx.shortest_path_length(self.tmp_G))
        self.tmp_G_nxts = {src:dict() for src in self.tmp_G.nodes()} # {tgt:nx.predecessor(self.tmp_G, tgt) for tgt in self.tmp_G.nodes()}
        for tgt in self.tmp_G.nodes():
            tmp_preds = nx.predecessor(self.tmp_G, tgt)
            for src in self.tmp_G.nodes():
                self.tmp_G_nxts[src][tgt] = tmp_preds[src]
        # print(self.tmp_G_spl)
        # print(self.tmp_G_nxts[(2,2)][(3,3)])

    def _data_append(self, rt, tup):
        pn,pv,s,d,n,v,pri = tup
        print(pn,pv,s,d,n,v,pri)
        pn,pv,s,d,n,v,pri = self.tmp_G_mapping[pn], pv, self.tmp_G_mapping[s], self.tmp_G_mapping[d], self.tmp_G_mapping[n], v, pri
        rt.append((pn,pv,s,d,n,v,pri))

    def xyroute(self):
        rt_data = list()

        tp_outf = os.path.join("./", "2dmesh_%d_%d_xy.tp" % (self.x, self.y))
        with open(tp_outf, 'w') as f: 
            writer = csv.writer(f, delimiter=" ")
            for e_s, e_d in sorted(self.G.edges):
                # router 2 router 11 1
                writer.writerow(["router", e_s, "router", e_d, 1])
            for node in sorted(self.G.nodes):
                # node 9 router 9
                writer.writerow(["node", node, "router", node])

        def _neighbors(n_coords):
            return self.tmp_G[n_coords]
        def _nexthop(n_coords, d_coords):
            cands = self.tmp_G_nxts[n_coords][d_coords]
            # print("cands:", cands)
            if len(cands) == 1:
                return cands[0]
            elif n_coords[1] == cands[0][1]:
                # move along x
                return cands[0]
            else:
                return cands[1]
            pass

        for s,d in it.permutations(self.tmp_G.nodes(),2):
            n = _nexthop(s,d)
            pri = len(self.tmp_G) - self.tmp_G_spl[n][d]
            # Todo: inject
            for pv,v in it.product(range(self.vc), range(self.vc)):
                # print(s,pv,s,d,n,v,pri)
                self._data_append(rt_data, (s,pv,s,d,n,v,pri))
                # rt_data.append((s,pv,s,d,n,v,pri))
            for pn,pv,v in it.product(_neighbors(s), range(self.vc), range(self.vc)):
                if pn == n or pn == d or _nexthop(pn,d) != s:
                    continue
                # print(pn,pv,s,d,n,v,pri)
                self._data_append(rt_data, (pn,pv,s,d,n,v,pri))
                # rt_data.append((pn,pv,s,d,n,v,pri))
                pass

        # print(_neighbors((0,0)))
        # print(_nexthop((0,0),(1,1)))

        xy_rt_outf = os.path.join("./", "2dmesh_%d_%d_xy.rt" % (self.x, self.y))
        with open(xy_rt_outf, 'w') as f:
    
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rt_data)
        
        pass

    def First_route(self, first_dir):
        # < => >,^,v (except v => ^)
        def _is_WF(pred, node, succ):
            is_west_first = (pred[1] == node[1] and pred[0] > node[0])
            is_down_first = (pred[0] == node[0] and pred[1] > node[1])
            is_west_last =  (node[1] == succ[1] and node[0] > succ[0])
            is_up_last =    (node[0] == succ[0] and node[1] < succ[1])

            # (< => <) or (< => {>,^,v})
            if is_west_first:
                return True
            # ({>,^,v} => <)
            elif is_west_last:
                return False
            # (v => ^)
            elif is_down_first and is_up_last:
                return False
            # ({>,^,v} => {>,^,v} except v => ^)
            else:
                return True

        # > => <,^,v (except v => ^)
        def _is_EF(pred, node, succ):
            is_east_first = (pred[1] == node[1] and pred[0] < node[0])
            is_down_first = (pred[0] == node[0] and pred[1] > node[1])
            is_east_last =  (node[1] == succ[1] and node[0] < succ[0])
            is_up_last =    (node[0] == succ[0] and node[1] < succ[1])

            # (> => >) or (> => {<,^,v})
            if is_east_first:
                return True
            # ({<,^,v} => >)
            elif is_east_last:
                return False
            # (v => ^)
            elif is_down_first and is_up_last:
                return False
            # ({<,^,v} => {<,^,v} except v => ^)
            else:
                return True

        if first_dir == "West":
            _is_First = _is_WF
            dir_suffix = "wf"
        elif first_dir == "East":
            _is_First = _is_EF
            dir_suffix = "ef"
        else:
            print("first_dir should be either West or East.")
            exit(1)

        tp_outf = os.path.join("./", "2dmesh_%d_%d_%s.tp" % (self.x, self.y, dir_suffix))
        with open(tp_outf, 'w') as f: 
            writer = csv.writer(f, delimiter=" ")
            for e_s, e_d in sorted(self.G.edges):
                # router 2 router 11 1
                writer.writerow(["router", e_s, "router", e_d, 1])
            for node in sorted(self.G.nodes):
                # node 9 router 9
                writer.writerow(["node", node, "router", node])

        H = nx.DiGraph()
        H.add_nodes_from(self.tmp_G.edges())
        H.add_nodes_from(map(lambda x: (x, SRC), self.tmp_G.nodes()))
        H.add_nodes_from(map(lambda x: (x, DST), self.tmp_G.nodes()))

        tmp_G_dir = self.tmp_G.to_directed()
        for node in tmp_G_dir.nodes():
            H.add_edge((node,SRC), (node,DST))
            for pred in tmp_G_dir.predecessors(node):
                H.add_edge((pred, node), (node, DST))
            for succ in tmp_G_dir.successors(node):
                H.add_edge((node, SRC), (node, succ))
            for pred, succ in it.product(tmp_G_dir.predecessors(node), tmp_G_dir.successors(node)):
                if _is_First(pred, node, succ):
                    H.add_edge((pred, node), (node, succ))
        
        if not nx.is_directed_acyclic_graph(H):
            print("cyclic")
            exit(1)
        for H_edge in H.edges():
            print(H_edge)
        print("len(H.edges()):", len(H.edges()))

        rt_data = list()

        num_nodes_in_G = len(tmp_G_dir)

        for dst in tmp_G_dir.nodes():
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
                            for pv, v in it.product(range(self.vc), range(self.vc)):
                                pn, s, d, n, hops = in_edge_H[0], out_edge_H[0], dst, out_edge_H[1], H_spl_tgt[out_edge_H]

                                # if pn == n or pn == d: # or _nexthop(pn,d) != s:
                                #     continue

                                self._data_append(rt_data, (pn,pv,s,d,n,v,num_nodes_in_G-hops))

                        # print(in_edge_H, out_edge_H, H_spl_tgt[out_edge_H])
                    pass
        # print(len(result_table), result_table[:10])
        print(rt_data, len(rt_data))

        first_rt_outf = os.path.join("./", "2dmesh_%d_%d_%s.rt" % (self.x, self.y, dir_suffix))
        with open(first_rt_outf, 'w') as f:
    
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rt_data)

        return rt_data

    def WFroute(self):
        return self.First_route("West")

    def EFroute(self):
        return self.First_route("East")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('x', metavar='x', type=int, help='x')
    parser.add_argument('y', metavar='y', type=int, help='y')
    parser.add_argument('vc', metavar='vc', type=int, help='vc')
    args = parser.parse_args()
    print(args)

    gen_2dMesh = Gen2DMesh(args.x, args.y, args.vc)
    gen_2dMesh.xyroute()
    gen_2dMesh.WFroute()
    gen_2dMesh.EFroute()
