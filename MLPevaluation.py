from utility.loadDataSet import loadDataSet
from utility.loadDataSet import loadClasses
from utility.loadDataSet import transpose
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import precision_recall_curve
from utility.scorer import *
import csv
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
    print("[INFO] MLP Training Started.")
    kf = StratifiedKFold(n_splits=5)
    clf = MLPClassifier(hidden_layer_sizes=(500, 500),
                        early_stopping=True)

    # popolamento della matrice di confusione necessaria al calcolo delle
    # metriche per example e cross validation con fold stratificate
    res1 = ["ROC"]
    res2 = ["PRC"]
    counter_confusion_matrix = [[] for i in range(len(features))]
    for y in Y:
        auc1 = 0
        auc2 = 0
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            cc = ClusterCentroids(random_state=0) # undersampling
            X_train, y_train = cc.fit_sample(X_train, y_train)
            clf.fit(X_train, y_train)
            auc_roc_i, auc_pr_i, diff_ = my_scorer(clf, X_test, y_test)
            auc1+=auc_roc_i
            auc2+=auc_pr_i
            i=0
            for index in test_index:
                counter_confusion_matrix[index].append(diff_[i])
                i+=1
        auc1 = auc1/5
        auc2 = auc2/5
        res1.append(auc1)
        res2.append(auc2)

    # scrittura su file dei risultati di AUC per ROC e PRC
    with open("./results/MLP_AUC_results.csv", "w") as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(["AUC"]+list(classes))
        csv_writer.writerow(res1)
        csv_writer.writerow(res2)

    # scrittura su file dei risultati per example di precision e recall
    with open("./results/MLP_Precision_Recall_multilabel_results.csv",
              'w') as f_i:
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

        #csv_writer.writerow([precision/50, recall/50])
        if len_div1>0:
            csv_writer.writerow([precision/len_div1, recall/len_div2])
        else:
            csv_writer.writerow([0, 0])
