import csv

########### For reading CSV #############
def list_elem_strtoint(lname):
    result = []
    for e in lname:
        result.append(int(e))
    return result

def mycsvread(arg):
    with open(arg, 'rt') as f:
        dialect = csv.Sniffer().sniff(f.read(), delimiters=' \t') # 
        f.seek(0)
        reader = csv.reader(f, dialect)
        
        for row in reader:
            yield(list_elem_strtoint(row))

