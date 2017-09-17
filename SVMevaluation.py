from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
from utility.loadDataSet import transpose
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from utility.scorer import *
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
    clf = svm.SVC(class_weight="balanced", probability=True)
    res1 = ["ROC"]
    res2 = ["PRC"]
    counter_confusion_matrix = [[] for i in range(len(features))]
    for y in Y:
        auc1 = 0
        auc2 = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            auc_roc_i, auc_pr_i, diff_ = my_scorer(clf, X_test, y_test)
            auc1+=auc_roc_i
            auc2+=auc_pr_i
            i=0
            for index in test_index:
                counter_confusion_matrix[index].append(diff_[i])
                i+=1
            # print(counter_confusion_matrix)
        auc1 = auc1/5
        auc2 = auc2/5
        res1.append(auc1)
        res2.append(auc2)
        print(auc1)
        print(auc2)

    with open("./results/SVM_AUC_results.csv", "w") as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(["AUC"]+list(classes))
        csv_writer.writerow(res1)
        csv_writer.writerow(res2)

    with open("./results/SVM_Precision_Recall_multilabel_results.csv", 'w') as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(["Precision", "Recall"])
        precision = 0
        recall = 0
        len_div1 = len(counter_confusion_matrix)
        len_div2 = len_div1
        for inst in counter_confusion_matrix:
            dict_ = Counter(inst)
            try:
                tp = dict_["TP"]
            except:
                tp = 0
            try:
                tn = dict_["TN"]
            except:
                tn = 0
            try:
                fn = dict_["FN"]
            except:
                fn = 0
            try:
                fp = dict_["FP"]
            except:
                fp = 0
            try:
                precision+=tp/(tp+fp)
            except:
                len_div1 = len_div1-1
            try:
                recall+=tp/(tp+fn)
            except:
                len_div2 = len_div2-1

        if len_div1>0:
            csv_writer.writerow([precision/len_div1, recall/len_div2])
        else:
            csv_writer.writerow([0, 0])
