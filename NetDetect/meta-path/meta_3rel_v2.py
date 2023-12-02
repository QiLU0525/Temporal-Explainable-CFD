import numpy as np
from scipy.sparse import coo_matrix

'''
    改进后的 #relation=3 meta path相似度计算
    For example,
    AB 矩阵为 user - paper, shape = [2,7]
        a = np.array([
          [1,0,0,0,1,0,0],
          [1,1,0,0,0,0,1],
        ])
    BC 矩阵为 paper - topic, shape = [7, 7]
        b = np.array([
          [1,1,0,0,0,0,0],
          [1,1,0,0,0,0,0],
          [0,0,1,0,0,0,0],
          [0,0,0,1,0,0,0],
          [0,0,0,0,1,0,1],
          [0,0,0,0,0,1,0],
          [0,0,0,0,1,0,1]
        ])
    CD 矩阵为 topic - group, shape = [7, 2]
        c = np.array([
          [1,1],
          [0,1],
          [0,0],
          [0,0],
          [1,0],
          [0,0],
          [0,1]
        ])

    user-group 相似度矩阵 m: shape= [2, 2]
        array([[0.5  , 0.625],
                [0.625, 0.75 ]])
'''

def generate_ngh(listOfCoordinates, listOfCoordinates2, listOfCoordinates3):
    dictAB, dictBA, dictBC, dictCB, dictCD, dictDC= {}, {}, {}, {}, {}, {}
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

    return dictAB, dictBA, dictBC, dictCB, dictCD, dictDC


def avg_sim(listAB, listDC, dictBC, dictCD):
        value = 0
        for node_b in listAB:
            for node_c in listDC:
                if node_b in dictBC and node_c in dictBC[node_b]:
                    value += 1 / len(dictBC[node_b]) / len(dictCD[node_c])
        return 1/len(listAB) * value


def cal_metapath(matrix_a, matrix_b, matrix_c, del_diagnoal=False):
    
    # matrix_a, matrix_b, matrix_c 都是 coo_matrix 对象
    # 得到 矩阵 a 中值为 1 的行标、列标和值 三个列表
    
    listOfCoordinates = list(zip(matrix_a.row, matrix_a.col))
    # listOfCoordinates: 矩阵 a 中值为 1 的行标和列标 [(0, 0), (0, 4), (1, 0), (1, 1), (1, 6)]

    listOfCoordinates2 = list(zip(matrix_b.row, matrix_b.col))
    # listOfCoordinates2: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 6), (5, 5), ()]

    listOfCoordinates3 = list(zip(matrix_c.row, matrix_c.col))
    dictAB, dictBA, dictBC, dictCB, dictCD, dictDC = generate_ngh(listOfCoordinates, listOfCoordinates2, listOfCoordinates3)

    _row, _col, _data = [], [], []

    for f1 in range(matrix_a.shape[0]):
        for f2 in range(matrix_c.shape[1]):
            if del_diagnoal and f1==f2:
                continue
            if f1 not in dictAB or f2 not in dictDC:
                continue
            #listAB = [body[1] for body in listOfCoordinates if body[0] == f1]
            listAB = dictAB[f1]
            # 矩阵 AB * BC 的不为0的列标 要和 CD 的行标去对应
            #listCD = [body[0] for body in listOfCoordinates3 if body[1] == f2]
            listDC = dictDC[f2]
            if not listAB and not listDC:
                continue

            rw = avg_sim(listAB, listDC, dictBC=dictBC, dictCD=dictCD)
            rw_reverse = avg_sim(listDC, listAB, dictBC=dictCB, dictCD=dictBA)
            
            if rw==0 and rw_reverse==0:
                continue
            _row.append(f1)
            _col.append(f2)
            _data.append( 0.5 * (rw + rw_reverse) )

    return coo_matrix((_data, (_row, _col)), shape = (matrix_a.shape[0], matrix_c.shape[1]), dtype=np.float32)

'''
# [2,7]
a = np.array([
  [1,0,0,0,1,0,0],
  [1,1,0,0,0,0,1],
])

# [7, 7]
b = np.array([
  [1,1,0,0,0,0,0],
  [1,1,0,0,0,0,0],
  [0,0,1,0,0,0,0],
  [0,0,0,1,0,0,0],
  [0,0,0,0,1,0,1],
  [0,0,0,0,0,1,0],
  [0,0,0,0,1,0,1]
])

# [7, 3]
c = np.array([
  [1,1],
  [0,1],
  [0,0],
  [0,0],
  [1,0],
  [0,0],
  [0,1]
])

m = cal_metapath(a,b,c)
m
'''