# cat *_hops_uniform* > 256_6_hops_uniform.dat

num_sim = 1
ij_rates = [i * 0.0005 for i in range(1,41)]
# print(ij_rates)
max_lat = 100.0
src_path = "../src_nnc_calc1"
topo_dir = "rtgen"
topo_nets = [("256_6_1_hops", "nncnet"), ("256_6_1_hops_ud", "cncnet")]
traffics = ["uniform", "transpose", "bitrev", "shuffle"]
num_vc = 1
log_dir = "log_hotspa4"

import itertools as it
for (topo, net), traffic in it.product(topo_nets, traffics):
    print("cat *_%s_%s* > %s_%s.out" % (topo, traffic, topo, traffic))
