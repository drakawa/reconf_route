import csv

class CsvWrite:
    def __init__(self, seq, fname):
        self.seq = seq
        self.fname = fname

    def csv_write(self):
        f = open(self.fname, 'w')
        dataWriter = csv.writer(f, delimiter=' ')

        print("out:", self.fname)
        dataWriter.writerows(self.seq)
        f.close()
        
