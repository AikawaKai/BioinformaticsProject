from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
from utility.loadDataSet import transpose
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from utility.scorer import *
from utility.resultPrinter import *
from utility.fmeasure import runFscore
import sys
from numpy import array
from collections import Counter

if __name__ == '__main__':
    # Caricamento delle istanze e delle etichette\annotazioni
    print("[INFO] Loading dataset.")
    dataset_dir = sys.argv[1]
    annotations = sys.argv[2]
    (features, X) = loadDataSet(dataset_dir)
    print("[INFO] Dataset loaded (Features, X).")
    print("[INFO] Loading classes.")
    (classes, Y) = loadClasses(annotations)
    print("[INFO] Classes loaded (Y).")

    # Preparazione della cross validation e del classificatore
    print("[INFO] SVM Training Started.")
    kf = StratifiedKFold(n_splits=5)
    clf = svm.SVC(class_weight="balanced", probability=True)

    # popolamento della matrice di confusione necessaria al calcolo delle
    # metriche per example e cross validation con fold stratificate
    res1 = ["ROC"]
    res2 = ["PRC"]
    threesholds = [i/100 for i in range(0,102,2)]
    counter_confusion_matrix = {t : [{"TP" : 0, "FP" : 0, "TN" : 0 , "FN" : 0} for i in range(len(features))] for t in threesholds}
    for y in Y:
        auc1 = 0
        auc2 = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            auc_roc_i, auc_pr_i, diff_ = threesholdExplorerScorer(clf, X_test, y_test, threesholds)
            auc1+=auc_roc_i
            auc2+=auc_pr_i
            for t in threesholds:
                i=0
                for index in test_index:
                    count = Counter(diff_[t][i])
                    tp = count["TP"] if "TP" in count else 0
                    tn = count["TN"] if "TN" in count else 0
                    fn = count["FN"] if "FN" in count else 0
                    fp = count["FP"] if "FP" in count else 0
                    counter_confusion_matrix[t][index]["TP"] += tp
                    counter_confusion_matrix[t][index]["TN"] += tn
                    counter_confusion_matrix[t][index]["FN"] += fn
                    counter_confusion_matrix[t][index]["FP"] += fp
                    i+=1
        auc1 = auc1/5
        auc2 = auc2/5
        res1.append(auc1)
        res2.append(auc2)

    # scrittura su file dei risultati di AUC per ROC e PRC
    printAUCROC("./results/SVM_AUC_results.csv", classes, res1, res2)

    # scrittura su file dei risultati per example di precision e recall
    "./results/SVM_Precision_Recall_multilabel_results_test.csv"
    printPrecisionRecall("./results/SVM_Precision_Recall_multilabel_results_test.csv",
                         "./results/SVM_Precision_Recall_multilabel_results_old.csv",
                         threesholds, counter_confusion_matrix)

    # calcolo e scrittura su file dell' F-measure per threshold
    runFscore("./results/SVM_Precision_Recall_multilabel_results_test.csv",
              "./results/SVM_F-measure_multilabel_results.csv")
    runFscore("./results/SVM_Precision_Recall_multilabel_results_old.csv",
              "./results/SVM_F-measure_multilabel_results_old.csv")
