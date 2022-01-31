import sys
import subprocess
from socket import gethostname
import time
import math
import itertools as it
import os

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
max_samples    = 100000;
sim_count = 1;

// print_activity = 1;
'''

class ExecSim:
    def __init__(self, net_name, src_dir, tp_dir, tp, traf, ijr, nvc):
        self.net_name = net_name
        self.src_dir = src_dir
        self.topo_dir = tp_dir
        self.topo = tp
        self.traffic = traf
        self.ij_rate = ijr
        self.num_vcs = nvc

    def execsim(self):
        net_name = self.net_name
        src_dir = self.src_dir
        topo_dir = self.topo_dir
        topo = self.topo
        traffic = self.traffic
        ij_rate = self.ij_rate
        num_vcs = self.num_vcs

        ij_rate_n = ij_rate * 0.99
        ij_rate_p = ij_rate * 1.01

        # ij_rates = [ij_rate, ij_rate_n, ij_rate_p]
        ij_rates = [ij_rate]

        bprocs = [BsimOpen(net_name, src_dir, topo_dir, topo, traffic, ijr, num_vcs, "deadlock", 300) for ijr in ij_rates]
        acc_ij_rate, avg_lat = None, None

        for bp in bprocs:
            bp.bsim_open()
            bp_gen = bp.get_line()
            rl = [None, None]

            for line in bp_gen:
#    
#                sys.stdout.write(line)
#    
                if "Overall average accepted rate" in line:
                    rl[0] = float(line.split()[5])
                elif "Overall average latency" in line:
                    rl[1] = float(line.split()[4])

            bp.terminate()

            if rl[0] != None and rl[1] != None and rl[1] <= 100.0:
                acc_ij_rate, avg_lat = rl
                break
                
        return acc_ij_rate, avg_lat
        
class BsimOpen:
    def __init__(self, nn, sd, td, tp, tr, ijr, nvc, term, tmax):
        self.net_name = nn
        self.src_dir = sd
        self.topo_dir = td
        self.topo = tp
        self.traffic = tr
        self.ij_rate = ijr
        self.num_vcs = nvc
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
        
        self.cfg_name = "%s_%s_%s_%.15f_%d.cfg" % (net_name, topo, traffic, ij_rate, num_vcs)
        cfg_name = self.cfg_name
        
        outf = open(cfg_name, "w")
        # nw_file, rt_file, traffic, ij_rate(f), num_vcs(d)
        # topo_path = "%s%s" % (topo_dir, topo)
        topo_path = os.path.join(topo_dir, topo)
        outf.write(CFG_TEMPLATE % (net_name, "%s.tp" % topo_path, "%s.rt" % topo_path, 
                                   traffic, ij_rate, num_vcs))
        outf.close()

        # cmd=["../src2_%s/booksim" % gethostname(), cfg_name]  
        cmd=[os.path.join(src_dir, "booksim"), cfg_name]  
        # self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf8")
        print("pid:", self.proc.pid)

    def get_line(self):
        term_max = self.term_max
        term_str = self.term_str
        term_count = 0
        while True:
            line = self.proc.stdout.readline()
            if line != '':
                # print(term_str)
                print(line, end="")
                if term_str in line:
                    term_count += 1
                    if term_count > term_max:
                        break
                yield line
            else:
                break

    def terminate(self):
        self.proc.terminate()
        subprocess.call(["rm", "-f", self.cfg_name])
        while True:
            if self.proc.poll() is not None:
                break
        
class RepeatSim:
    def __init__(self, nn, n_it, init_df, thr, src_dir, tp_dir, tp, traf, nvc):
        self.net_name = nn
        self.num_iter = n_it
        self.init_diff = init_df
        self.threshold = thr

        self.src_dir = src_dir
        self.topo_dir = tp_dir
        self.topo = tp
        self.traffic = traf
        self.num_vcs = nvc

    def repeatsim(self):
        net_name = self.net_name
        num_iter = self.num_iter
        tmp_diff = self.init_diff 
        threshold = self.threshold

        src_dir = self.src_dir
        topo_dir = self.topo_dir
        topo = self.topo
        traffic = self.traffic
        ij_rate = 0.0
        num_vcs = self.num_vcs

        prev_succ = True
        ever_failed = False

        results = dict()
        ij_rate_failmin = 100.0 # sorry for irresponsibility
        for _ in range(num_iter):
            ij_rate += tmp_diff
            print("tmp_ij_rate:", ij_rate)
            exec_sim = ExecSim(net_name, src_dir, topo_dir, topo, traffic, ij_rate, num_vcs)
            acc_ij_rate, avg_lat = exec_sim.execsim()
            print("RESULT:ij_rate:", acc_ij_rate, "avg_lat:", avg_lat)
            tmp_succ = (acc_ij_rate != None)
            if prev_succ != tmp_succ:
                tmp_diff *= -0.5
            elif ever_failed:
                tmp_diff *= 0.5

            if not tmp_succ and ij_rate_failmin > ij_rate:
                ij_rate_failmin = ij_rate

            prev_succ = tmp_succ
            ever_failed |= (not tmp_succ)

            if tmp_succ:
                results[acc_ij_rate] = avg_lat

            if math.fabs(tmp_diff) * 120 < ij_rate:
                break    

            if avg_lat == None or avg_lat > threshold:
                break

#        return ij_rate, results
        return ij_rate_failmin, results
