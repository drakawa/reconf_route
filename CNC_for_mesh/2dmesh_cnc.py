import argparse
import networkx as nx

class Gen2DMeshCDG:
    def __init__(self, x, y):
        tmp_G = nx.grid_2d_graph(x, y)
        tmp_G_mapping = dict([(i, e) for i, e in enumerate(list(sorted(tmp_G.nodes())))])
        print(tmp_G_mapping)
        self.G = nx.grid_2d_graph(x, y)
        print(self.G.nodes, self.G.edges)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('x', metavar='x', type=int, help='x')
    parser.add_argument('y', metavar='y', type=int, help='y')
    parser.add_argument('vc', metavar='vc', type=int, help='vc')
    args = parser.parse_args()
    print(args)
    print(args.x)

    gen_2dMeshCDG = Gen2DMeshCDG(args.x, args.y)

    # num_plots = args.num_plots
    # ij_rate, end_lat = args.ij_rate, args.end_lat
    # src_dir, topo_dir, topo, traffic = args.src_dir, args.topo_dir, args.topo, args.traffic
    # num_vcs = args.num_vcs
    # log_dir = args.log_dir
    # net_name = args.net_name
