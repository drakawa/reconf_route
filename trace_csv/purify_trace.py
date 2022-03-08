import csv
import argparse
import glob
import math
import subprocess

def get_csvfiles(prefix):
    searching = "{:s}_[0-9]*.csv".format(prefix)
    # print(searching)
    csvfiles = glob.glob(searching)
    return sorted(csvfiles)

class PurifyCSV:
    def __init__(self, csvfiles, sample_period, flit_size):
        self.csvfiles = csvfiles
        self.sample_period = sample_period
        self.flit_size = flit_size

    def get_num_packets_num_cycles(self):
        num_packets = 0
        num_cycles = 0
        for clk, src, dst, packet_size in self.purify_csv():
            if src != dst and packet_size > 0:
                num_packets += 1
                num_cycles = clk

        return num_packets, num_cycles

        # num_packets = 0
        # for csvfile in self.csvfiles:
        #     tmp_stdout = subprocess.run(["wc", "-l", csvfile], stdout = subprocess.PIPE).stdout.split()
        #     # print(tmp_stdout)
        #     num_lines = int(tmp_stdout[0])
        #     # print(num_lines)
        #     num_packets += num_lines
        #     # print("run:", subprocess.run(["wc", "-l", csvfile], stdout = subprocess.PIPE).stdout)
        # # print(num_packets)

        # tmp_stdout = subprocess.run(["tail", "-n1", csvfiles[-1]], stdout = subprocess.PIPE).stdout.decode("utf_8").split(",")
        # # print(tmp_stdout)
        # num_cycles = round(float(tmp_stdout[4]) * self.sample_period)
        # # print(num_cycles)

        # return num_packets, num_cycles

    def purify_csv(self):
        for csvfile in self.csvfiles:
            # print(csvfile)
            with open(csvfile) as f:
                is_head = True
                reader = csv.reader(f)
                for row in reader:
                    if is_head:
                        is_head = False
                        continue
                    # print(row)
                    src, dst, size, start = int(row[0][1:])-1, int(row[1][1:])-1, int(row[2]), float(row[4])
                    # print(src, dst, size, start)
                    packet_size = math.ceil(size / self.flit_size)
                    clk = round(start * self.sample_period)
                    # print(packet_size)
                    # print(clk)
                    yield clk, src, dst, packet_size
                    # exit(1)

if __name__ == "__main__":
    """
input
-t trace: suffix (_*.csv) 前のファイル名
例: crossbar_256_is.A.256_trace
-s sample_period: 
-f flit_size

output
filename: trace_s_f_p_c.tr
p, c: num_packets, cycles
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('trace', help='trace file name prefix (before _*.csv)')
    parser.add_argument('sample_period', type=float)
    parser.add_argument('flit_size', type=int)

    args = parser.parse_args()
    trace, sample_period, flit_size = args.trace, args.sample_period, args.flit_size

    print("args:", args)

    csvfiles = get_csvfiles(trace)
    print("csvfiles:", csvfiles)

    purify_csv = PurifyCSV(csvfiles, sample_period, flit_size)
    num_packets, num_cycles = purify_csv.get_num_packets_num_cycles()
    print("num_packets, num_cycles:", num_packets, num_cycles)
    
    # debugcount = 0
    # for clk, src, dst, packet_size in purify_csv.purify_csv():
    #     print(clk,src,dst,packet_size)
    #     debugcount += 1
    #     if debugcount > 100:
    #         exit(1)

    outf_name = "{:s}_{:.2e}_{:d}_{:d}_{:d}.tr".format(trace, sample_period, flit_size, num_packets, num_cycles)
    print("outf_name:", outf_name)

    with open(outf_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([num_packets])
        for clk, src, dst, packet_size in purify_csv.purify_csv():
            if src != dst and packet_size > 0:
                writer.writerow([clk, src, dst, packet_size])