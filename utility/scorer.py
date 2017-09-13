from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
auc_res = []

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
    global auc_res
    yPred = estimator.predict(x)
    yScores = estimator.decision_function(x)
    # print(yScores)
    fpr, tpr, thresholds = roc_curve(y, yScores, pos_label=1)
    # print(thresholds)
    # print(classification_report(y, yPred))
    print(confusion_matrix(y, yPred))
    auc_ = auc(fpr, tpr)
    auc_res.append(auc_)
    return (precision_score(y, yPred, average='binary'), auc_, yPred)

def my_scorer(estimator, x, y):
    p, roc_area, yPred = getScores(estimator, x, y)
    diffs_ = [checkPredict(c) for c in zip(y, yPred)]
    print(Counter(diffs_))
    return (roc_area, diffs_)
