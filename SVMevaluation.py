from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
from utility.loadDataSet import transpose
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from utility.scorer import *
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
import csv
import sys
from numpy import array
from collections import Counter

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
    print(len(classes))
    # print(Y)
    print("[INFO] Classes loaded (Y).")
    print(len(Y[0]), len(features))
    print("[INFO] SVM Training Started.")
    kf = StratifiedKFold(n_splits=5)
    clf = svm.SVC(class_weight="balanced")
    res = [""]
    counter_confusion_matrix = [[] for i in range(len(features))]
    for y in Y:
        auc = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            auc_i, diff_ = my_scorer(clf, X_test, y_test)
            auc+=auc_i
            i=0
            for index in test_index:
                counter_confusion_matrix[index].append(diff_[i])
                i+=1
            # print(counter_confusion_matrix)
        cur_auc = auc/5
        res.append(cur_auc)
        print(cur_auc)
        auc_ = 0

    with open("./results/SVM_AUC_results.csv", "w") as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(["AUC"]+list(classes))
        csv_writer.writerow(res)

    with open("./results/SVM_Precision_Recall_multilabel_results.csv") as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(["Precision", "Recall"])
        precision = 0
        recall = 0
        for inst in counter_confusion_matrix:
            dict_ = Counter(inst)
            tp = dict_["TP"]
            tn = dict_["TN"]
            fn = dict_["FN"]
            fp = dict_["FP"]
            precision+=tp/(tp+fp)
            recall+=tp/(tp+fn)
        csv_writer.writerow([precision/len(counter_confusion_matrix), recall/len(counter_confusion_matrix)])
