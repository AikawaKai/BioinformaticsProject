import csv
from collections import Counter

def printAUCROC(filename, classes, res1, res2):
    with open(filename, "w") as f_i:
        csv_writer = csv.writer(f_i, delimiter=",")
        csv_writer.writerow(["AUC"] + list(classes))
        csv_writer.writerow(res1)
        csv_writer.writerow(res2)

def __prepareCSV__(fout):
    csv_writer = csv.writer(fout, delimiter=",")
    csv_writer.writerow(["Threshold", "Precision", "Recall"])
    return csv_writer

def printPrecisionRecall(filename, filenameOld, threesholds, counter_confusion_matrix):
    fout = open(filename, "w")
    foutOld = open(filenameOld, "w")
    csv_writer = __prepareCSV__(fout)
    csv_writer_old = __prepareCSV__(foutOld)

    for threeshold in threesholds:
        precision = 0
        recall = 0
        len_div1 = len(counter_confusion_matrix[threeshold])
        len_div2 = len_div1
        len_div3 = len_div1
        len_div4 = len_div1
        for inst in counter_confusion_matrix[threeshold]:
            try:
                precision+=inst["TP"]/(inst["TP"]+inst["FP"])
            except:
                len_div3 = len_div3 -1

            try:
                recall+=inst["TP"]/(inst["TP"]+inst["FN"])
            except:
                len_div4 = len_div4 -1

        #csv_writer.writerow([precision/50, recall/50])
        csv_writer.writerow([threeshold, precision / len_div1, recall / len_div2])

        if len_div3>0:
            csv_writer_old.writerow([threeshold, precision / len_div3, recall / len_div4])
        else:
            csv_writer_old.writerow([threeshold, 0, 0])

    foutOld.close()
    fout.close()
