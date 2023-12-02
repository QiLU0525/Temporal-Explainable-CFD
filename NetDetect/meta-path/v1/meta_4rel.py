import numpy as np


def avg_sim(listAB, listDE, listCoordinate, listCoordinate2):
    listLen = []

    listBC = [item[1] for item in listCoordinate if item[0] in listAB]
    listLen.append(len(set(listBC)))

    listCD = [item[1] for item in listCoordinate2 if item[0] in listBC]
    listLen.append(len(set(listCD)))

    if not listLen or max(listLen) == 0:
        return 0
    else:
        return len(set(listCD) & set(listDE)) / max(listLen)

def cal_metapath(matrix_a, matrix_b, matrix_c, matrix_d, user_group_pair):

    arr = np.where(matrix_a == 1)
    listOfCoordinates = list(zip(arr[0], arr[1]))

    arr2 = np.where(matrix_b == 1)
    listOfCoordinates2 = list(zip(arr2[0], arr2[1]))

    arr3 = np.where(matrix_c == 1)
    listOfCoordinates3 = list(zip(arr3[0], arr3[1]))

    arr4 = np.where(matrix_d == 1)
    listOfCoordinates4 = list(zip(arr4[0], arr4[1]))

    users = np.size(matrix_a, 0)
    groups = np.size(matrix_d, 1)
    matrix_m = np.zeros((users,groups))
    for u, g, _ in user_group_pair:
        listAB = [item[1] for item in listOfCoordinates if item[0] == u]
        listFE = [item[0] for item in listOfCoordinates4 if item[1] == g]
        if not listAB:
             matrix_m[u,g]=0
        else:
            rw = avg_sim(listAB, listFE, listOfCoordinates2, listOfCoordinates3)
            rw_reverse = avg_sim(listFE, listAB, listOfCoordinates2, listOfCoordinates3)
            matrix_m[u,g] = 1 / 2 * (rw + rw_reverse)
    return matrix_m