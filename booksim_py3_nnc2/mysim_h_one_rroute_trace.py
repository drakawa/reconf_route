import sys
import subprocess
from socket import gethostname
import time
import math
import itertools as it
import os

import re

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
warmup_periods = 1;
sample_period  = 100000;
max_samples    = %d;
sim_count = 1;

converged_periods = %d;

// trace_file
use_trace_file = %d;
trace_file = %s;

// print_activity = 1;
'''

class ExecSim:
    def __init__(self, net_name, src_dir, tp_dir, tp, traf, ijr, nvc, use_trace_file=False, trace_file="nan"):
        self.net_name = net_name
        self.src_dir = src_dir
        self.topo_dir = tp_dir
        self.topo = tp
        self.traffic = traf
        self.ij_rate = ijr
        self.num_vcs = nvc
        self.use_trace_file = use_trace_file
        self.trace_file = trace_file

    def execsim(self):
        net_name = self.net_name
        src_dir = self.src_dir
        topo_dir = self.topo_dir
        topo = self.topo
        traffic = self.traffic
        ij_rate = self.ij_rate
        num_vcs = self.num_vcs
        use_trace_file = self.use_trace_file
        trace_file = self.trace_file

        bp = BsimOpen(net_name, src_dir, topo_dir, topo, traffic, ij_rate, num_vcs, use_trace_file, trace_file, "hogehoge", 300)
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
    def __init__(self, nn, sd, td, tp, tr, ijr, nvc, utrace, trace, term, tmax):
        self.net_name = nn
        self.src_dir = sd
        self.topo_dir = td
        self.topo = tp
        self.traffic = tr
        self.ij_rate = ijr
        self.num_vcs = nvc
        self.utrace = utrace
        self.trace = trace
        self.term_str = term
        self.term_max = tmax
        self.cfg_name = ""

        self.proc = None

    def bsim_open(self):
        net_name = self.net_name
        src_dir = self.src_dir
        topo_dir = self.topo_dir
        topo = self.topo
        traffic = self.traffic
        ij_rate = self.ij_rate
        num_vcs = self.num_vcs
        utrace = self.utrace
        trace = self.trace
        
        if utrace: 
            utrace_int = 1
        else:
            utrace_int = 0

        self.cfg_name = "%s_%s_%s_%.15f_%d.cfg" % (net_name, topo, trace, ij_rate, num_vcs)
        cfg_name = self.cfg_name
        
        outf = open(cfg_name, "w")
        # nw_file, rt_file, traffic, ij_rate(f), num_vcs(d)
        # topo_path = "%s%s" % (topo_dir, topo)
        topo_path = os.path.join(topo_dir, topo)
        tmp_stdout = subprocess.run(["tail", "-n1", trace], stdout = subprocess.PIPE).stdout.split()
        # trace_cycles = int(re.split("_|\.", trace)[-2])
        trace_cycles = int(tmp_stdout[0])

        sample_period = 100000
        cycle_margin = 10000
        max_samples = (trace_cycles + cycle_margin) // sample_period + 1
        # print(trace)
        # print(trace_cycles)
        # exit(1)
        cfg_to_write = CFG_TEMPLATE % (net_name, "%s.tp" % topo_path, "%s.txt" % topo_path, traffic, ij_rate, num_vcs, max_samples, max_samples, utrace_int, trace)
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
        term_max = self.term_max
        term_str = self.term_str
        term_count = 0
        while True:
            if self.proc.poll() != None:
                break

            line = self.proc.stdout.readline()
            print(line, end="")

            if term_str in line:
                term_count += 1
                if term_count > term_max:
                    break

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
