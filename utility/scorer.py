from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, classification_report, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from collections import Counter
import matplotlib.pyplot as plt

# metodo per plottare la curva ROC
def plotCurve(fpr, tpr, auc, labely, labelx):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# metodo per controllare se il risultato TP, TN, FP, o FN
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

# metodo custom per area roc e area prc
def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    yScores = estimator.predict_proba(x)[:,1]
    fpr, tpr, thresholds = roc_curve(y, yScores, pos_label=1)
    precision, recall, thresholds = precision_recall_curve(y, yScores, pos_label=1)
    auc_ro = auc(fpr, tpr)
    auc_pr = average_precision_score(y, yScores)
    print(confusion_matrix(y, yPred))
    return (precision_score(y, yPred, average='binary'), auc_ro, auc_pr, yPred)

def my_scorer(estimator, x, y):
    p, roc_area, prec_recall_area, yPred = getScores(estimator, x, y)
    diffs_ = [checkPredict(c) for c in zip(y, yPred)]
    print(Counter(diffs_))
    return (roc_area, prec_recall_area, diffs_)
