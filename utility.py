from csv import DictReader, writer

from matplotlib.pyplot import figure, barh, yticks, title, savefig
from numpy import argsort, array
from datetime import datetime

# Прочитать данные из файла CSV в справочник
# можно было бы читать и в list, но PyCharm начинает подчёркивать как ошибку обращения к элементам списка по индексу.
# Да, мне известно, что это можно сделать иными 100500 способами.
def readFromCSV(filename) -> dict:
    result = {}
    idRow = 0
    with open(filename, "r", encoding = "utf8") as csvFile:
        reader = DictReader(csvFile, delimiter = ';')
        for row in reader:
            result[idRow] = row
            idRow += 1

    return result


# сохранить значения из справочника в файл
def writeToCSV(filename, columns, listOfRows):
    with open(filename, "w", encoding="utf8", newline="") as csvFile:
        csvWriter = writer(csvFile, delimiter=";")
        csvWriter.writerow(columns)
        for row in listOfRows:
            csvWriter.writerow(row)


# отобразить на графике значимость параметров обученной модели
# и сохранить в файл, имя которого будет содержать метку времени
def drawFeatures(model, columns):
    feature_importance = model.feature_importances_
    sorted_idx = argsort(feature_importance)
    figure(figsize=(12, 12))
    barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    yticks(range(len(sorted_idx)), array(columns)[sorted_idx])
    title('Feature Importance')
    savefig("train_features_" + datetime.now().strftime("%Y-%m-%d-%H-%M") + ".png")
