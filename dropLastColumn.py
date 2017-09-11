import sys
import csv
from utility.loadDataSet import transpose

if __name__ == '__main__':

    filename = sys.argv[1]
    with open(filename, 'r') as file_o:
        file_r = csv.reader(file_o, delimiter="\t")
        rows = [row for row in file_r]
        rows = transpose(rows)
        rows = rows[:-1]
        rows = transpose(rows)

    with open(filename+"2.csv", 'w') as file_w:
        writer_csv = csv.writer(file_w, delimiter='\t')
        for row in rows:
            writer_csv.writerow(row)
