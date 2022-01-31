import sys
import subprocess
from socket import gethostname
import time
import math
from mysim_h_seq import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('num_plots', metavar='NUM_PLOTS', type=int, help='# of plots')
    parser.add_argument('ij_rate', metavar='IJ_RATE', type=float, help='injection rate')
    parser.add_argument('end_lat', metavar='END_LAT', type=float, help='maximum cycles in latency')
    parser.add_argument('src_dir', metavar='SRC_DIR', type=str, help='booksim source directory')
    parser.add_argument('topo_dir', metavar='TOPO_DIR', type=str, help='topology files directory')
    parser.add_argument('topo', metavar='TOPO', type=str, help='topology name')
    parser.add_argument('traffic', metavar='TRAFFIC', type=str, help='traffic name')
    parser.add_argument('num_vcs', metavar='NUM_VCS', type=int, help='# of virtual channels')
    parser.add_argument('log_dir', metavar='LOG_DIR', type=str, help='output log files directory')
    parser.add_argument('net_name', metavar='NET_NAME', type=str, help='network name (e.g. anynet)')
    args = parser.parse_args()
    # print(args.integers)

    num_plots = args.num_plots
    ij_rate, end_lat = args.ij_rate, args.end_lat
    src_dir, topo_dir, topo, traffic = args.src_dir, args.topo_dir, args.topo, args.traffic
    num_vcs = args.num_vcs
    log_dir = args.log_dir
    net_name = args.net_name

    # num_nodes, degree, seed = args.integers
    # pri_mode, = args.string

    # if len(sys.argv) < 10:
    #     print("python THISFILE num_plots ij_rate end_lat src_dir topo_dir topo traffic num_vcs log_dir")
    #     exit(1)
    # num_plots = int(sys.argv[1])
    # ij_rate, end_lat = float(sys.argv[2]), float(sys.argv[3])
    # src_dir, topo_dir, topo, traffic = sys.argv[4:8]
    # num_vcs = int(sys.argv[8])
    # log_dir = sys.argv[9]

#    topo_dir = "topology/lcrandom/"
#    topo = "64_r4_2_s1_e18_n4"
#    traffic = "shuffle"
#    num_vcs = 5

#    num_iter = 15
#    init_interval = 0.1
#    end_lat = 150.0
#
#    repeat_sim = RepeatSim(num_iter, init_interval, end_lat, topo_dir, topo, traffic, num_vcs)
#    max_ij_rate, results = repeat_sim.repeatsim()
#    print "max_ij_rate", max_ij_rate
#
#    midlog_name = "%s_%s_%d.midlog" % (topo, traffic, num_vcs)
#    log_path = "%s/%s" % (log_dir, midlog_name)
#    outf = open(log_path, "w")
#    num_plots = 30
#    ij_rate = max_ij_rate / float(num_plots)
#    results_str = ""
#    for k in xrange(num_plots):
#        tmp_ij_rate = ij_rate * (k + 1)
##        results_str += "%.15f %.15f\n" % (k, results[k])
#        results_str += "python2.7 mysim_oneplot.py 1 %.15f %.15f %s %s %s %d\n" % (ij_rate, end_lat, topo_dir, topo, traffic, num_vcs)
#    outf.write(results_str)
#    outf.close()
#    exit(1)

    repeat_sim = RepeatSim(net_name, num_plots, ij_rate, end_lat, src_dir, topo_dir, topo, traffic, num_vcs)
    max_ij_rate, results = repeat_sim.repeatsim()
    print("max_ij_rate", max_ij_rate)

    rlog_name = "%d_%.15f_%f_%s_%s_%d.rlog" % (num_plots, ij_rate, end_lat, topo, traffic, num_vcs)
    log_path = "%s/%s" % (log_dir, rlog_name)
    outf = open(log_path, "w")
    results_str = ""
    for k in sorted(results.keys()):
        results_str += "%.15f %.15f\n" % (k, results[k])
    if not list(results.keys()):
        results_str += "%.15f %.15f\n" % (max_ij_rate, end_lat * 2)
    outf.write(results_str)
    outf.close()
    
