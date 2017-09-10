import sys
import csv

if __name__ == '__main__':
    dataset_file = sys.argv[1]
    with open(dataset_file, 'r') as file_o:
        file_r = csv.reader(file_o, delimiter='\t')
        rows = [row for row in file_r]
    features = rows[0] # features
    X = [row[1:] for row in rows[1:]] #instances
    
