import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
# meta_2rel

def generate_ngh(listOfCoordinates, listOfCoordinates2):
    dictAB, dictCB = {}, {}
    for i0, i1 in listOfCoordinates:
        if i0 in dictAB:
            dictAB[i0].append(i1)
        else:
            dictAB[i0] = [i1]
    for i0, i1 in listOfCoordinates2:
        if i1 in dictCB:
            dictCB[i1].append(i0)
        else:
            dictCB[i1] = [i0]
    return dictAB, dictCB

def cal_metapath(matrix_a, matrix_b, del_diagnoal=False):
    # matrix_b 是倒的，列标（col）是所有上市公司
    listOfCoordinates = list(zip(matrix_a.row, matrix_a.col))
    listOfCoordinates2 = list(zip(matrix_b.row, matrix_b.col))
    
    # temp = pd.DataFrame(columns=['_row','_col','_data'])
    # temp.to_csv(path,encoding='utf-8',index=None)
    
    dictAB, dictBC = generate_ngh(listOfCoordinates, listOfCoordinates2)
    _row, _col, _data = [], [], []

    for f_i in range(matrix_a.shape[0]):
        # print(f_i)
        for f_j in range(matrix_b.shape[1]):
            if del_diagnoal and f_i==f_j:
                # del_diagnoal 是否删除对角线元素
                continue
            if f_i not in dictAB or f_j not in dictBC:
                continue
            # listAB = [item[1] for item in listOfCoordinates if item[0] == f_i]
            listAB = dictAB[f_i]
            # listBC = [item[0] for item in listOfCoordinates2 if item[1] == f_j]
            listBC = dictBC[f_j]
            if not listBC or not listAB or not list(set(listAB) & set(listBC)): # listBC是空集
                continue
            
            listLen = [len(listAB),len(listBC)]
            rw =  len(set(listAB) & set(listBC)) / max(listLen)
            _row.append(f_i)
            _col.append(f_j)
            _data.append(rw)

            '''temp = pd.DataFrame({
                '_row':f_i,
                '_col':f_j,
                '_data':rw}, index=[0])
            temp.to_csv(path,index=None,encoding='utf-8',header=None, mode='a')'''
    # return 0
    return coo_matrix((_data, (_row, _col)), shape = (matrix_a.shape[0], matrix_b.shape[1]), dtype=np.float32)