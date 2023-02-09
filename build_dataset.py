import numpy as np

def build_dataset_train(df, seq_length):
    dataX = []
    dataY = []
    for i in range(len(df)-seq_length):
        _x = df[i:i+seq_length, :]
        _y = df[i+seq_length, [-1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)


def build_dataset_test(df, seq_length):
    dataX = []
    for i in range(len(df)-seq_length+1):
        _x = df[i:i+seq_length, :]
        dataX.append(_x)
    
    return np.array(dataX)