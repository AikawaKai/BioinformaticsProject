import csv
import os
from numpy import array

def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))]
            for j in range(len(matrix[0]))]

def loadRowsFromCsv(filename):
    with open(filename, 'r') as file_o:
        file_r = csv.reader(file_o, delimiter='\t')
        rows = [row for row in file_r]
    return rows

def loadDataSet(filename):
    rows = loadRowsFromCsv(filename)
    features = rows[0] # features
    X = [row[1:] for row in rows[1:]] #instances
    X = [[float(val) for val in row] for row in X]
    return (array(features), array(X))

def loadClasses(dirname):
    files = os.listdir(dirname)
    all_classes_name = []
    all_y = []
    for file_ in files:
        rows = loadRowsFromCsv(dirname+"/"+file_)
        classes = rows[0][1:]
        all_classes_name+=classes
        rows = rows[1:]
        rows = [row[1:] for row in rows]
        rows = [[int(val) for val in row] for row in rows]
        all_y+=transpose(rows)
    print(len(all_y[0]))
    return(array(all_classes_name), array(all_y))
