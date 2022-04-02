import sys
import subprocess
from socket import gethostname
import time
import math
import itertools as it
import os
import re
import argparse

# nw_file, rt_file, traffic, ij_rate(f), num_vcs(d)
CFG_TEMPLATE = '''// Topoogy
topology = %s;
network_file = %s;

// Routing
routing_function = min;
routing_table_file = %s;
use_vc = 1;

// Traffic
traffic = %s;
injection_rate = %.15f;

// Flow control
num_vcs     = %d;
vc_buf_size = 8;
wait_for_tail_credit = 1;

// Allocator
vc_allocator = select; // islip, select, separable_input_first
vc_alloc_arb_type = round_robin; // round_robin, matrix

sw_allocator = islip; // islip, select 
sw_alloc_arb_type = round_robin; // round_robin, matrix

alloc_iters  = 1;

// Router speedup
input_speedup     = 1;
output_speedup    = 1;
internal_speedup  = 1.0;

// Latency (4-cycle)
credit_delay   = 0;
routing_delay  = 1;
vc_alloc_delay = 1;
sw_alloc_delay = 1;
st_final_delay = 0;

// Simulation
sim_type = latency;
// warmup_periods = 1;
warmup_periods = 0;
sample_period  = 100000;
max_samples    = %d;
sim_count = 1;

converged_periods = %d;

// trace_file
use_trace_file = %d;
trace_file = %s;

latency_thres = 3000.0;

// print_activity = 1;
'''

class ExecSim:
    def __init__(self, net_name, src_dir, tp_dir, tp, rt_dir, rt, trace_dir, trace, traf, ijr, nvc, use_trace_file=False):
        """
        hoge
        """
        self.net_name = net_name
        self.src_dir = src_dir
        self.topo_path = os.path.join(tp_dir, tp)
        self.rt_path = os.path.join(rt_dir, rt)
        self.trace_path = os.path.join(trace_dir, trace)
        self.traffic = traf
        self.ij_rate = ijr
        self.num_vcs = nvc
        self.use_trace_file = use_trace_file

    def execsim(self):
        bp = BsimOpen(self.net_name, self.src_dir, self.topo_path, self.rt_path, self.trace_path, self.traffic, self.ij_rate, self.num_vcs, self.use_trace_file)
        acc_ij_rate, avg_lat = None, None

        bp.bsim_open()
        bp_gen = bp.get_line()
        rl = [None, None]

        for line in bp_gen:
            if "Overall average accepted rate" in line:
                rl[0] = float(line.split()[5])
            elif "Overall average latency" in line:
                rl[1] = float(line.split()[4])

            # bp.terminate()

        if rl[0] != None and rl[1] != None and rl[1] <= 100.0:
            acc_ij_rate, avg_lat = rl
                
        return acc_ij_rate, avg_lat
        
class BsimOpen:
    def __init__(self, nn, sd, tpp, rtp, trp, traf, ijr, nvc, utrace):
        self.net_name = nn
        self.src_dir = sd
        self.topo_path = tpp
        self.rt_path = rtp
        self.trace_path = trp
        self.traffic = traf
        self.ij_rate = ijr
        self.num_vcs = nvc
        self.utrace = utrace

        self.proc = None

    def bsim_open(self):        
        if self.utrace: 
            utrace_int = 1
        else:
            utrace_int = 0

        topo = os.path.basename(self.topo_path)
        rt = os.path.basename(self.rt_path)
        trace = os.path.basename(self.trace_path)
        
        self.cfg_name = "%s_%s_%s_%s.cfg" % (self.net_name, topo, rt, trace)
        cfg_name = self.cfg_name
        
        outf = open(cfg_name, "w")
        # nw_file, rt_file, traffic, ij_rate(f), num_vcs(d)
        # topo_path = "%s%s" % (topo_dir, topo)
        tmp_stdout = subprocess.run(["tail", "-n1", self.trace_path], stdout = subprocess.PIPE).stdout.split()
        # trace_cycles = int(re.split("_|\.", trace)[-2])
        trace_cycles = int(tmp_stdout[0])

        sample_period = 100000
        cycle_margin = 10000
        max_samples = (trace_cycles + cycle_margin) // sample_period + 1
        # print(trace)
        # print(trace_cycles)
        # exit(1)
        cfg_to_write = CFG_TEMPLATE % (self.net_name, self.topo_path, self.rt_path, self.traffic, self.ij_rate, self.num_vcs, max_samples, max_samples, utrace_int, self.trace_path)
        print(cfg_to_write)
        outf.write(cfg_to_write)
        outf.close()

        # cmd=["../src2_%s/booksim" % gethostname(), cfg_name]  
        cmd=[os.path.join(src_dir, "booksim"), cfg_name]  
        print("cmd:", *cmd)
        # self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf8")
        print("pid:", self.proc.pid)

    def get_line(self):
        while True:
            if self.proc.poll() != None:
                break

            line = self.proc.stdout.readline()

            if "WARNING: Possible network deadlock." in line:
                continue
            elif "Trace: Packet is injected" in line:
                continue
            elif "in_flight_Rold.size" in line:
                continue
            
            print(line, end="")

            yield line
            # else:
            #     continue
            #     # print("break!")
            #     # break

    # def terminate(self):
    #     self.proc.terminate()
    #     # subprocess.call(["rm", "-f", self.cfg_name])
    #     while True:
    #         if self.proc.poll() is not None:
    #             break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('src_dir', metavar='SRC_DIR', type=str, help='booksim source directory')
    parser.add_argument('topo_dir', metavar='TOPO_DIR', type=str, help='topology files directory')
    parser.add_argument('topo', metavar='TOPO', type=str, help='topology name')
    parser.add_argument('rt_dir', metavar='RT_DIR', type=str, help='routing table files directory')
    parser.add_argument('rt', metavar='RT', type=str, help='routing table name')
    parser.add_argument('trace_dir', metavar='TRACE_DIR', type=str, help='trace files directory')
    parser.add_argument('trace', metavar='TRACE', type=str, help='trace name')
    parser.add_argument('log_dir', metavar='LOG_DIR', type=str, help='output log files directory')
    parser.add_argument('net_name', metavar='NET_NAME', type=str, help='network name (e.g. anynet)')
    args = parser.parse_args()
    # print(args.integers)

    src_dir = args.src_dir

    topo_dir = args.topo_dir
    topo = args.topo
    rt_dir = args.rt_dir
    rt = args.rt
    trace_dir = args.trace_dir
    trace = args.trace

    log_dir = args.log_dir
    net_name = args.net_name
    
    traffic = "uniform" # actually not used
    ij_rate = 0.1 # actually not used
    num_vcs = 1 # fixed

    end_lat = 300.0 # fixed

    exec_sim = ExecSim(net_name, src_dir, topo_dir, topo, rt_dir, rt, trace_dir, trace, traffic, ij_rate, num_vcs, True)
    acc_ij_rate, avg_lat = exec_sim.execsim()
    print("acc_ij_rate", acc_ij_rate)

    rlog_name = "%s_%s_%s.rlog" % (topo, rt, trace)
    log_path = "%s/%s" % (log_dir, rlog_name)
    outf = open(log_path, "w")

    if acc_ij_rate != None:
        result_str = "%.15f %.15f\n" % (acc_ij_rate, avg_lat)
    else:
        result_str = "%.15f %.15f\n" % (ij_rate, end_lat * 2)

    outf.write(result_str)
    outf.close()
    
