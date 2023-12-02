import numpy as np
from scipy.sparse import coo_matrix

def generate_ngh(listOfCoordinates, listOfCoordinates2, listOfCoordinates3, listOfCoordinates4):
    dictAB, dictBA, dictBC, dictCB, dictCD, dictDC, dictDE, dictED= {}, {}, {}, {}, {}, {}, {}, {}
    for i0, i1 in listOfCoordinates:
        if i0 in dictAB:
            dictAB[i0].append(i1)
        else:
            dictAB[i0] = [i1]
        if i1 in dictBA:
            dictBA[i1].append(i0)
        else:
            dictBA[i1] = [i0]

    for i0, i1 in listOfCoordinates2:
        if i0 in dictBC:
            dictBC[i0].append(i1)
        else:
            dictBC[i0] = [i1]
        if i1 in dictCB:
            dictCB[i1].append(i0)
        else:
            dictCB[i1] = [i0]

    for i0, i1 in listOfCoordinates3:
        if i0 in dictCD:
            dictCD[i0].append(i1)
        else:
            dictCD[i0] = [i1]
        
        if i1 in dictDC:
            dictDC[i1].append(i0)
        else:
            dictDC[i1] = [i0]
    
    for i0, i1 in listOfCoordinates4:
        if i0 in dictDE:
            dictDE[i0].append(i1)
        else:
            dictDE[i0] = [i1]
        
        if i1 in dictED:
            dictED[i1].append(i0)
        else:
            dictED[i1] = [i0]

    return dictAB, dictBA, dictBC, dictCB, dictCD, dictDC, dictDE, dictED

def avg_sim(listAB, listED, dictBC, dictCD, dictDE):
    value = 0
    for node_b in listAB:
        for node_d in listED:
            if node_b in dictBC:
                for node_c in dictBC[node_b]:
                    if node_c in dictCD and node_d in dictCD[node_c]:
                        value += 1 / (len(dictBC[node_b]) * len(dictCD[node_c]) * len(dictDE[node_d]))
    return 1/len(listAB) * value

def cal_metapath(matrix_a, matrix_b, matrix_c, matrix_d, del_diagnoal=False):

    listOfCoordinates = list(zip(matrix_a.row, matrix_a.col))
    listOfCoordinates2 = list(zip(matrix_b.row, matrix_b.col))
    listOfCoordinates3 = list(zip(matrix_c.row, matrix_c.col))
    listOfCoordinates4 = list(zip(matrix_d.row, matrix_d.col))

    dictAB, dictBA, dictBC, dictCB, dictCD, dictDC, dictDE, dictED = generate_ngh(listOfCoordinates, listOfCoordinates2, listOfCoordinates3, listOfCoordinates4)

    _row, _col, _data = [], [], []

    for f1 in range(matrix_a.shape[0]):
        for f2 in range(matrix_d.shape[1]):
            if del_diagnoal and f1==f2:
                continue
            if f1 not in dictAB or f2 not in dictED:
                continue
            listAB = dictAB[f1]
            listED = dictED[f2]
            
            rw = avg_sim(listAB, listED, dictBC, dictCD, dictDE)
            rw_reverse = avg_sim(listED, listAB, dictDC, dictCB, dictBA)
            if rw==0 and rw_reverse==0:
                continue
            _row.append(f1)
            _col.append(f2)
            _data.append( 0.5 * (rw + rw_reverse))
    return coo_matrix((_data, (_row, _col)), shape = (matrix_a.shape[0], matrix_d.shape[1]), dtype=np.float32)

