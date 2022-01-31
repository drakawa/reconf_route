import sys
import subprocess
from socket import gethostname
import time
import math
import itertools as it

# nw_file, rt_file, traffic, ij_rate(f), num_vcs(d)
CFG_TEMPLATE = '''// Topoogy
topology = anynet;
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
vc_allocator = islip; 
sw_allocator = islip;
alloc_iters  = 2;

// Router speedup
input_speedup     = 1;
output_speedup    = 1;
internal_speedup  = 1.0;

// Latency (3-cycle)
credit_delay   = 0;
routing_delay  = 1;
vc_alloc_delay = 1;
sw_alloc_delay = 0;
st_final_delay = 0;

// Simulation
sim_type = latency;
warmup_periods = 1;
sample_period  = 100000;
max_samples    = 100000;
sim_count = 1;
'''

class ExecSim:
    def __init__(self, tp_dir, tp, traf, ijr, nvc):
        self.topo_dir = tp_dir
        self.topo = tp
        self.traffic = traf
        self.ij_rate = ijr
        self.num_vcs = nvc

    def execsim(self):
        topo_dir = self.topo_dir
        topo = self.topo
        traffic = self.traffic
        ij_rate = self.ij_rate
        num_vcs = self.num_vcs

        ij_rate_n = ij_rate * 0.99
        ij_rate_p = ij_rate * 1.01

        ij_rates = [ij_rate_n, ij_rate, ij_rate_p]

        bprocs = [BsimOpen(topo_dir, topo, traffic, ijr, num_vcs, "deadlock", 300) for ijr in ij_rates]
        for bp in bprocs:
            bp.bsim_open()

        bp_gens = [bp.get_line() for bp in bprocs]
        rls = [[None, None] for _ in range(len(bp_gens))]
        
        acc_ij_rate, avg_lat = None, None
        for lines in it.izip_longest(*bp_gens, fillvalue = ""):
            for l_idx, line in enumerate(lines):
#    
#                sys.stdout.write(line)
#    
                if "Overall average accepted rate" in line:
                    rls[l_idx][0] = float(line.split()[5])
                elif "Overall average latency" in line:
                    rls[l_idx][1] = float(line.split()[4])

            for rl in rls:
                if rl[0] != None and rl[1] != None and rl[1] <= 100.0:
                    acc_ij_rate, avg_lat = rl
                    break
                
            if acc_ij_rate != None and avg_lat != None:
                break

        for bp in bprocs:
            bp.terminate()

        print(rls)
        return acc_ij_rate, avg_lat
        
class BsimOpen:
    def __init__(self, td, tp, tr, ijr, nvc, term, tmax):
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
        topo_dir = self.topo_dir
        topo = self.topo
        traffic = self.traffic
        ij_rate = self.ij_rate
        num_vcs = self.num_vcs
        
        self.cfg_name = "%s_%s_%.15f_%d.cfg" % (topo, traffic, ij_rate, num_vcs)
        cfg_name = self.cfg_name
        
        outf = open(cfg_name, "w")
        # nw_file, rt_file, traffic, ij_rate(f), num_vcs(d)
        topo_path = "%s%s" % (topo_dir, topo)
        outf.write(CFG_TEMPLATE % ("%s.tp" % topo_path, "%s.rt" % topo_path, 
                                   traffic, ij_rate, num_vcs))
        outf.close()

        cmd=["../src2_%s/booksim" % gethostname(), cfg_name]        
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("pid:", self.proc.pid)

    def get_line(self):
        term_max = self.term_max
        term_str = self.term_str
        term_count = 0
        while True:
            line = self.proc.stdout.readline()
            if line != '':
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
    def __init__(self, n_it, init_df, thr, tp_dir, tp, traf, nvc):
        self.num_iter = n_it
        self.init_diff = init_df
        self.threshold = thr

        self.topo_dir = tp_dir
        self.topo = tp
        self.traffic = traf
        self.num_vcs = nvc

    def repeatsim(self):
        num_iter = self.num_iter
        tmp_diff = self.init_diff 
        threshold = self.threshold

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
            exec_sim = ExecSim(topo_dir, topo, traffic, ij_rate, num_vcs)
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

            if avg_lat > threshold:
                break

#        return ij_rate, results
        return ij_rate_failmin, results

if __name__ == '__main__':
    print("hoge")
