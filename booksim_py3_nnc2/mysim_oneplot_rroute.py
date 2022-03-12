import sys
import subprocess
from socket import gethostname
import time
import math
from mysim_h_one_rroute import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
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

    ij_rate, end_lat = args.ij_rate, args.end_lat
    src_dir, topo_dir, topo, traffic = args.src_dir, args.topo_dir, args.topo, args.traffic
    num_vcs = args.num_vcs
    log_dir = args.log_dir
    net_name = args.net_name

    exec_sim = ExecSim(net_name, src_dir, topo_dir, topo, traffic, ij_rate, num_vcs)
    acc_ij_rate, avg_lat = exec_sim.execsim()
    print("acc_ij_rate", acc_ij_rate)

    rlog_name = "%.15f_%f_%s_%s_%d.rlog" % (ij_rate, end_lat, topo, traffic, num_vcs)
    log_path = "%s/%s" % (log_dir, rlog_name)
    outf = open(log_path, "w")

    if acc_ij_rate != None:
        result_str = "%.15f %.15f\n" % (acc_ij_rate, avg_lat)
    else:
        result_str = "%.15f %.15f\n" % (ij_rate, end_lat * 2)

    outf.write(result_str)
    outf.close()
    
