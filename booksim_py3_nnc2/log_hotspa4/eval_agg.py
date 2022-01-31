# cat *_hops_uniform* > 256_6_hops_uniform.dat

num_sim = 1
ij_rates = [i * 0.002 for i in range(1,46)]
# print(ij_rates)
max_lat = 100.0
src_path = "../src_nnc_calc1"
topo_dir = "rtgen"
topo_nets = [("256_16_1_hops", "nncnet"), ("256_16_1_hops_ud", "cncnet")]
traffics = ["uniform", "transpose", "bitrev", "shuffle"]
num_vc = 1
log_dir = "log_hotspa4"

import itertools as it
for ij_rate, (topo, net), traffic in it.product(ij_rates, topo_nets, traffics):
    print("cat %d_%.5f*_%s_%s* >> %s_%s.out" % (num_vc, ij_rate, topo, traffic, topo, traffic))
