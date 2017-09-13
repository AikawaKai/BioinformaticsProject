from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
from utility.loadDataSet import transpose
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
import sys
from numpy import array

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
    print(Y)
    print("[INFO] Classes loaded (Y).")
    print(len(Y[0]), len(features))
    print("[INFO] MLP Training Started.")
    clf = MLPClassifier(hidden_layer_sizes=tuple([len(features) for i in range(2)]),
                        early_stopping=True)
    OneVsRestClassifier(clf).fit(X, Y).predict(X)
