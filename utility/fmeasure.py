import csv

def loadObjects(inFile):
    with open(inFile, "r") as f:
        fileReader = csv.reader(f)
        return [{"threshold" : row[0], "precision" : row[1], "recall" : row[2]} for row in fileReader][1:]

def toFloat(el):
    el["precision"] = float(el["precision"])
    el["recall"] = float(el["recall"])
    el["threshold"] = float(el["threshold"])
    return el

def fscore(el):
    el["fscore"] = (2 * el["precision"] * el["recall"])/(el["precision"] + el["recall"])
    return el

if __name__ == "__main__":
    from sys import argv
    fscores = sorted([fscore(toFloat(e)) for e in loadObjects(argv[1])],key = lambda x: x["fscore"])
    with open(argv[2], "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Threshold, F-score, Precision, Recall"])
        for el in fscores:
            writer.writerow([el["threshold"], el["fscore"], el["precision"], el["recall"]])

