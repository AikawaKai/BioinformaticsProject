from utility.loadDataSet import loadDataSet
import sys

if __name__ == '__main__':
    dataset_dir = sys.argv[1]
    (features, X) = loadDataSet(dataset_dir)
    print(features)
