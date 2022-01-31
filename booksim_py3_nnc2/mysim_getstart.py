from mysim_h import *
if __name__ == '__main__':
    if len(sys.argv) < 6:
        print "python THISFILE topo_dir topo traffic num_vcs log_dir"
        exit(1)
    topo_dir, topo, traffic = sys.argv[1:4]
    num_vcs = int(sys.argv[4])
    log_dir = sys.argv[5]

#    topo_dir = "topology/lcrandom/"
#    topo = "64_r4_2_s1_e18_n4"
#    traffic = "shuffle"
#    num_vcs = 5

    num_iter = 15
    init_interval = 0.1
    end_lat = 100.0

    repeat_sim = RepeatSim(num_iter, init_interval, end_lat, topo_dir, topo, traffic, num_vcs)
    max_ij_rate, results = repeat_sim.repeatsim()
    print "max_ij_rate", max_ij_rate

    midlog_name = "%s_%s_%d.midwtflog" % (topo, traffic, num_vcs)
    log_path = "%s/%s" % (log_dir, midlog_name)
    outf = open(log_path, "w")
    num_plots = 30
    ij_rate = max_ij_rate / float(num_plots)
    results_str = ""
    for k in xrange(num_plots):
        tmp_ij_rate = ij_rate * (k + 1)
#        results_str += "%.15f %.15f\n" % (k, results[k])
        results_str += "python2.7 mysim_oneplot.py 1 %.15f %f %s %s %s %d %s\n" % (tmp_ij_rate, end_lat, topo_dir, topo, traffic, num_vcs, log_dir)
    outf.write(results_str)
    outf.close()
    exit(1)

#    num_plots = 50
#    ij_rate = max_ij_rate / float(num_plots)
#
#    repeat_sim = RepeatSim(num_plots, ij_rate, end_lat, topo_dir, topo, traffic, num_vcs)
#    max_ij_rate, results = repeat_sim.repeatsim()
#
#    rlog_name = "%s_%s_%d.rlog" % (topo, traffic, num_vcs)
#    log_path = "%s/%s" % (log_dir, rlog_name)
#    outf = open(log_path, "w")
#    results_str = ""
#    for k in sorted(results.keys()):
#        results_str += "%.15f %.15f\n" % (k, results[k])
#    outf.write(results_str)
#    outf.close()
    
