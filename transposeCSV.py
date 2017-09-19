import sys
import csv
from utility.loadDataSet import transpose

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f_o:
        csv_reader = csv.reader(f_o, delimiter=',')
        data = [row for row in csv_reader]

    data = transpose(data)
    with open("./new_csv.csv", "w") as f_w:
        csv_writer = csv.writer(f_w, delimiter=',')
        for row in data:
            csv_writer.writerow(row)
