import numpy as np
from scipy.sparse import coo_matrix
# meta_2rel

def generate_ngh(listOfCoordinates, listOfCoordinates2):
    dictAB, dictBA, dictBC, dictCB = {}, {}, {}, {}
    for i0, i1 in listOfCoordinates:
        # A 的正向出度：A->B
        if i0 in dictAB:
            dictAB[i0].append(i1)
        else:
            dictAB[i0] = [i1]
        # B 的反向出度: B->A
        if i1 in dictBA:
            dictBA[i1].append(i0)
        else:
            dictBA[i1] = [i0]

    for i0, i1 in listOfCoordinates2:
        # B 的正向出度: B->C
        if i0 in dictBC:
            dictBC[i0].append(i1)
        else:
            dictBC[i0] = [i1]
        
        # C 的反向出度: C->B
        if i1 in dictCB:
            dictCB[i1].append(i0)
        else:
            dictCB[i1] = [i0]

    return dictAB, dictBA, dictBC, dictCB


def cal_metapath(matrix_a, matrix_b, del_diagnoal=False):
    # matrix_b 是倒的，列标（col）是所有上市公司
    listOfCoordinates = list(zip(matrix_a.row, matrix_a.col))
    listOfCoordinates2 = list(zip(matrix_b.row, matrix_b.col))

    dictAB, dictBA, dictBC, dictCB = generate_ngh(listOfCoordinates, listOfCoordinates2)

    '''outdegree = np.array([list(matrix_b.row).count(i) for i in range(matrix_a.shape[1])])
    outdegree_reverse = np.array([list(matrix_a.col).count(i) for i in range(matrix_b.shape[0])])
    '''
    _row, _col, _data = [], [], []
    for f_i in range(matrix_a.shape[0]):
        for f_j in range(matrix_b.shape[1]):
            if del_diagnoal and f_i==f_j:
                continue
            
            if f_i not in dictAB or f_j not in dictCB:
                continue

            listAB = dictAB[f_i]
            listCB = dictCB[f_j]
            
            # print(f_i,f_j,listAB,listBC)
            if not listCB or not listAB or not list(set(listAB) & set(listCB)): # listBC是空集
                continue

            # listLen = [len(listAB),len(listBC)]
            # rw, rw_reverse 得到一个向量
            rw =  1/len(listAB) * 1 / np.array([len(dictBC[i]) for i in set(listAB) & set(listCB)])
            # outdegree[list(set(listAB) & set(listBC))]
            rw_reverse = 1/len(listCB) * 1 / np.array([len(dictBA[i]) for i in set(listAB) & set(listCB)])
            
            _row.append(f_i)
            _col.append(f_j)
            _data.append(1/2 * (np.sum(rw) + np.sum(rw_reverse)))

    # print(_row, _col, _data)
    return coo_matrix((_data, (_row, _col)), shape = (matrix_a.shape[0], matrix_b.shape[1]), dtype=np.float32)
