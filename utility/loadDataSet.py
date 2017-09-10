import csv
from numpy import array

def loadDataSet(filename):
    with open(filename, 'r') as file_o:
        file_r = csv.reader(file_o, delimiter='\t')
        rows = [row for row in file_r]
    features = rows[0] # features
    X = [row[1:] for row in rows[1:]] #instances
    return (array(features), array(X))
