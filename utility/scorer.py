from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, classification_report, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from collections import Counter

def checkPredict(c):
    res = ''
    if c[0]==c[1]:
        res+='T'
        if c[0] == 0:
            res+='N'
            return res
        res+='P'
        return res
    res+='F'
    if c[0] == 0:
        res+='P'
        return res
    return res+'N'

def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    yScores = estimator.predict_proba(x)[:,1]
    print(yScores)
    # print(yScores)
    fpr, tpr, thresholds = roc_curve(y, yScores, pos_label=1)
    auc_ro = auc(fpr, tpr)
    #precision, recall, thresholds = precision_recall_curve(y, yScores, pos_label=1)
    # print(fpr, tpr)
    #precision_recall_curve
    auc_pr = average_precision_score(y, yScores)
    # print(thresholds)
    # print(classification_report(y, yPred))
    print(confusion_matrix(y, yPred))
    #auc_pr = auc(recall, precision)
    #print(auc_pr, auc(recall, precision))

    #auc_ro = roc_auc_score(y, yScores)
    return (precision_score(y, yPred, average='binary'), auc_ro, auc_pr, yPred)

def my_scorer(estimator, x, y):
    p, roc_area, prec_recall_area, yPred = getScores(estimator, x, y)
    diffs_ = [checkPredict(c) for c in zip(y, yPred)]
    print(Counter(diffs_))
    return (roc_area, prec_recall_area, diffs_)
