from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
from utility.loadDataSet import transpose
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from utility.scorer import *
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
import csv
import sys
from numpy import array

if __name__ == '__main__':
    print("[INFO] Loading dataset.")
    dataset_dir = sys.argv[1]
    annotations = sys.argv[2]
    (features, X) = loadDataSet(dataset_dir)
    # print(features)
    # print(X)
    print("[INFO] Dataset loaded (Features, X).")
    print("[INFO] Loading classes.")
    (classes, Y) = loadClasses(annotations)
    # print(classes)
    # print(Y)
    print("[INFO] Classes loaded (Y).")
    print(len(Y[0]), len(features))
    print("[INFO] MLP Training Started.")
    clf = MLPClassifier(hidden_layer_sizes=tuple([100 for i in range(2)]),
                        early_stopping=True)
    results = []
    sme = SMOTEENN(random_state=10)
    kf = StratifiedKFold(n_splits=2)
    res = []
    for y in Y:
        X_res, y_res = sme.fit_sample(X, y)
        auc = 0
        for train_index, test_index in kf.split(X_res, y_res):
            X_train, X_test = X_res[train_index], X_res[test_index]
            y_train, y_test = y_res[train_index], y_res[test_index]
            clf.fit(X_train, y_train)
            auc_i = my_scorer(clf, X_test, y_test)
            auc+=auc_i
        cur_auc = auc/5
        res.append(cur_auc)
        print(cur_auc)
        auc_ = 0

    with open("./results/MLPresults.csv", "w") as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(classes)
        csv_writer.writerow(res)
