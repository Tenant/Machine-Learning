from sklearn.model_selection import train_test_split
import numpy as np

def read_input(filepath):
    data = []
    labels = []
    with open(filepath) as ifile:
        print("="*40)
        print("Loading data from " + str(filepath))
        for line in ifile:
            tokens = line.strip().split('	')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(float(tokens[-1]))
    x = np.array(data)
    y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print("Data loaded")
    return x_train, x_test, y_train, y_test