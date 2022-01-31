filename = "mysim_oneplot.py"
num_sim = 1
# ij_rates = [i * 0.005 for i in range(7,41)]
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
import os

for ij_rate, (topo, net), traffic in it.product(ij_rates, topo_nets, traffics):
    rlog_name = "%d_%.15f_%f_%s_%s_%d.rlog" % (num_sim, ij_rate, max_lat, topo, traffic, num_vc)
    rlog_path = os.path.join(log_dir, rlog_name)
    if os.path.exists(rlog_path) and os.path.getsize(rlog_path) > 0:
        continue
    print("python %s %d %.5f %.5f %s %s %s %s %d %s %s" % (filename, num_sim, ij_rate, max_lat, src_path, topo_dir, topo, traffic, num_vc, log_dir, net))

# python mysim_oneplot.py 1 0.00500 100.000000 ../src_nnc_calc1 rtgen 256_16_1_hops_ud uniform 1 log_hotspa cncnet