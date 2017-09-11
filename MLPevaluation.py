from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
import sys

if __name__ == '__main__':
    print("[INFO] Loading dataset.")
    dataset_dir = sys.argv[1]
    annotations = sys.argv[2]
    (features, X) = loadDataSet(dataset_dir)
    print(features)
    print(X)
    print("[INFO] Dataset loaded (Features, X).")
    print("[INFO] Loading classes.")
    (classes, Y) = loadClasses(annotations)
    print(len(classes))
    print(len(Y))
    print("[INFO] Classes loaded (Y).")
    print(len(Y[0]), len(features))
