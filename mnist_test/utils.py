import numpy as np


def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    # np.random.permutation: np.arange(인자) 또는 정수 배열에 대한 shuffle 반환

    # trainX & trainY or testX & testY 같이 2쌍이거나 trainX, trainY, testX, testY처럼 네쌍이거나
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t
