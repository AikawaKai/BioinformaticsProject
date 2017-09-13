from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
auc_res = []

def getScores(estimator, x, y):
    global auc_res
    yPred = estimator.predict(x)
    fpr, tpr, thresholds = roc_curve(y, yPred, pos_label=1)
    #print(classification_report(y, yPred))
    auc_res.append(auc(fpr, tpr))
    return (precision_score(y, yPred, average='binary'),
            confusion_matrix(y, yPred), auc(fpr, tpr))

def my_scorer(estimator, x, y):
    p, confMatrix, roc_area = getScores(estimator, x, y)
    return roc_area
