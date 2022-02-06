import argparse
import networkx as nx
import csv
import os
import itertools as it

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

    def xyroute(self):
        rt_data = list()

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
        def _data_append(rt, tup):
            pn,pv,s,d,n,v,pri = tup
            print(pn,pv,s,d,n,v,pri)
            pn,pv,s,d,n,v,pri = self.tmp_G_mapping[pn], pv, self.tmp_G_mapping[s], self.tmp_G_mapping[d], self.tmp_G_mapping[n], v, pri
            rt.append((pn,pv,s,d,n,v,pri))

        for s,d in it.permutations(self.tmp_G.nodes(),2):
            n = _nexthop(s,d)
            pri = len(self.tmp_G) - self.tmp_G_spl[n][d]
            # Todo: inject
            for pv,v in it.product(range(self.vc), range(self.vc)):
                # print(s,pv,s,d,n,v,pri)
                _data_append(rt_data, (s,pv,s,d,n,v,pri))
                # rt_data.append((s,pv,s,d,n,v,pri))
            for pn,pv,v in it.product(_neighbors(s), range(self.vc), range(self.vc)):
                if pn == n or pn == d or _nexthop(pn,d) != s:
                    continue
                # print(pn,pv,s,d,n,v,pri)
                _data_append(rt_data, (pn,pv,s,d,n,v,pri))
                # rt_data.append((pn,pv,s,d,n,v,pri))
                pass

        # print(_neighbors((0,0)))
        # print(_nexthop((0,0),(1,1)))

        xy_rt_outf = os.path.join("./", "2dmesh_%d_%d_xy.rt" % (self.x, self.y))
        with open(xy_rt_outf, 'w') as f:
    
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rt_data)
        
        pass

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('x', metavar='x', type=int, help='x')
    parser.add_argument('y', metavar='y', type=int, help='y')
    parser.add_argument('vc', metavar='vc', type=int, help='vc')
    args = parser.parse_args()
    print(args)

    gen_2dMesh = Gen2DMesh(args.x, args.y, args.vc)
    gen_2dMesh.xyroute()
