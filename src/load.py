import numpy as np


def load_data(train_data):

    max_user = max(train_data[:, 0])
    max_object = max(train_data[:, 1])

    print("max_user , max_objects = %d , %d "%(max_user,max_object))
    print train_data
    M = np.empty([int(max_user), int(max_object)])
    M[:]=-1
    for i in range(0, len(train_data)):
        M[int(train_data[i, 0]) - 1, int(train_data[i, 1]) - 1] = train_data[i, 2]
    return M
