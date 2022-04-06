script_name = "mysim_rroute_trace.py"
script_dir = "../src_nnc_calc1"
tp_dir = "rtgen_rr_edgefiles"
txt_dir = "rtgen_rr_edgefiles"
tr_dir = "trfiles"
log_dir = "log_rroute"
net_name = "reconfroute"

degree_seeds = [(4,1),(8,4)]
trans_margins = [1,5,10]
num_splits = [2**i for i in range(14)]

tr_names = [
    "crossbar_64_bt.W.64_trace_1.00e09_4096_620085_62590300.tr",
    "crossbar_64_cg.W.64_trace_1.00e09_4096_267004_41961500.tr",
    "crossbar_64_ep.W.64_trace_1.00e09_4096_2655_9327740.tr",
    "crossbar_64_ft.W.64_trace_1.00e09_4096_18223_2379540.tr",
    "crossbar_64_is.W.64_trace_1.00e09_4096_53946_4909650.tr",
    "crossbar_64_lu.W.64_trace_1.00e09_4096_2163744_243721000.tr",
    "crossbar_64_mg.W.64_trace_1.00e09_4096_72726_5670700.tr",
    "crossbar_64_sp.W.64_trace_1.00e09_4096_1234359_134421000.tr"
]


num_node = 64
degree_seeds = [(4,1),(8,4)]

trans_margins = [1,5,10]

import itertools as it
import os
for tr_name, num_split, (degree, num_seed), trans_margin in it.product(tr_names, num_splits, degree_seeds, trans_margins):
    tp_name = "{}_{}_{}_hops_ud.tp".format(num_node, degree, num_seed)
    txt_name = "{}_{}_{}_{}_{}.txt".format(tr_name, num_split, degree, num_seed, trans_margin)
    # python mysim_rroute_trace.py ../src_nnc_calc1 rtgen_rr_edgefiles 64_8_4_hops_ud.tp rtgen_rr_edgefiles crossbar_64_is.W.64_trace_1.00e09_4096_53946_4909650.tr_8192_8_4_10.txt trfiles crossbar_64_is.W.64_trace_1.00e09_4096_53946_4909650.tr log_rroute reconfroute
    cmd_str = "python {} {} {} {} {} {} {} {} {} {}".format(script_name, script_dir, tp_dir, tp_name, txt_dir, txt_name, tr_dir, tr_name, log_dir, net_name)
    # print(cmd_str)
    # print("python %s %s")
    # print(trfile, num_split, (degree, num_seed), trans_margin)
    rlog_name = "%s_%s_%s.rlog" % (tp_name, txt_name, tr_name)
    print(os.path.exists(os.path.join(log_dir, rlog_name)), rlog_name)
