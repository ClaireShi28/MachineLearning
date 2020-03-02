import pandas as pd

def load_wine_quality_data():

    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                       sep=';')
    #print("Wine Quality Data Description")
    print(data.describe())
    data = data.dropna()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y_original=data.iloc[:,-1]
    y = y.values
    y = y.astype(int)
    y[y < 6] = 0
    y[y >= 6] = 1

    return X, y

def load_ozone_data():
    data = pd.read_csv('https://raw.githubusercontent.com/ClaireShi28/MachineLearning/master/Assignment1/Data/PM2.5.csv',
                       error_bad_lines=False, sep=",")
    print("Air Pollution Data Description")
    print(data.describe())
    data = data.dropna()
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y_original = data.iloc[:, -1]
    y = y.values
    #y = y.astype(float)
    y[y <= 22] = 0 #22 for PM2.5
    y[y > 22] = 1

    return X, y




