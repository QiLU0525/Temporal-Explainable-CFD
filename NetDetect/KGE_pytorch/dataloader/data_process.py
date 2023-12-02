import os
import json
import pandas as pd
import random

def generate_entity2id(all_body):

    # 用f.write写入的话要把双引号去掉，因为有些实体的名字带双引号，会使得编码出问题，最后entity2id.txt读取错误
    with open('../data/CorpNet/entity2id.txt', 'w', encoding='gb18030') as f:
        for i in range(len(all_body)):
            item = all_body[i]
            if type(item)==str:
                # 因为带双引号的名字会使得后面读取出错，比如3133行 "SungrowPower(HongKong)CompanyLimited
                item = item.replace('"','')
            f.write('{}\t{}\r\n'.format(item, i))
    '''df = pd.DataFrame(columns=['name','id'])
    df.to_csv('../data/CorpNet/entity2id.txt',encoding='gb18030', index=None, header=None, sep='\t')
    for i in range(len(all_body)):
        temp = pd.DataFrame({
                'name':all_body[i],
                'id':i},index=[0])
        temp.to_csv('../data/CorpNet/entity2id.txt',encoding='gb18030', mode='a', header=None, index=None, sep='\t')'''

    print('总共有%d个实体：%d家上市公司，%d家非上市公司和%d名自然人' % (len(all_body), len(listed_corp), len(unlisted_corp), len(people)))
    return all_body

def generate_relation():
    all_relation = ['SubsidiaryOf','KinshipOf','ManagerOf','ShareholderOf','HasTransaction']
    with open('../data/CorpNet/relation2id.txt', 'w') as f:
        for item in all_relation:
            f.write('{}\t{}\r\n'.format(item, all_relation.index(item)))
    print('总共有%d个关系类型' % len(all_relation))
    return all_relation

def generate_triple(all_body):
    def trans_to_int(item):
        try:
            return int(item)
        except:
            return item
    # 生成 triple2id.txt
    subfirm = pd.read_csv('../../上市公司子公司情况表/subsidiary_fullname_matched.csv', encoding='gb18030')
    qinshu = pd.read_csv('../../股权及亲属关系/qinshu_no_dup_person.csv', encoding='gb18030')
    gaoguan = pd.read_csv('../../高管关系/gaoguan_no_dup_person.csv', encoding='gb18030')
    gudong = pd.read_csv('../../十大流通股股东/gudong_not_fin.csv', encoding='gb18030')
    guanlian = pd.read_csv('../../关联方交易/guanlian_no_dup_person.csv', encoding='gb18030')

    subfirm = subfirm.set_index(['interval'])
    qinshu = qinshu.set_index(['interval'])
    gaoguan = gaoguan.set_index(['interval'])
    gudong = gudong.set_index(['interval'])
    guanlian = guanlian.set_index(['interval'])

    # 先生成文件夹
    for year in range(2003, 2022):
        if not os.path.exists(os.path.join('../data/CorpNet', str(year))):
            os.mkdir(os.path.join('../data/CorpNet', str(year)))

    triple_num = 0
    for year in range(2003, 2022):
        with open('../data/CorpNet/{}/triple2id.txt'.format(year), 'w') as f:
            # SubsidiaryOf
            for Stkcd, sub in subfirm.loc[year,:].values:
                f.write('{}\t{}\t{}\n'.format(all_body.index(trans_to_int(sub)), 0 ,all_body.index(int(Stkcd))))
                triple_num += 1

            # KinshipOf
            for person1, person2 ,_ ,_ in qinshu.loc[year,:].values:
                f.write('{}\t{}\t{}\n'.format(all_body.index(person1), 1, all_body.index(person2)))
                f.write('{}\t{}\t{}\n'.format(all_body.index(person2), 1, all_body.index(person1)))
                triple_num += 2

            # ManagerOf
            for Stkcd, manager ,_ ,_ in gaoguan.loc[year,:].values:
                f.write('{}\t{}\t{}\n'.format(all_body.index(manager), 2, all_body.index(int(Stkcd))))
                triple_num += 1

            # ShareholderOf
            for holded, holder ,_ in gudong.loc[year,:].values:
                f.write('{}\t{}\t{}\n'.format(all_body.index(trans_to_int(holder)), 3, all_body.index(trans_to_int(holded))))
                triple_num += 1

            # HasTransaction
            for seller, buyer ,_  in guanlian.loc[year,:].values:
                f.write('{}\t{}\t{}\n'.format(all_body.index(trans_to_int(seller)), 4, all_body.index(trans_to_int(buyer))))
                triple_num += 1

        print('%d年有%d个三元组' % (year, triple_num))
        triple_num = 0

def train_test_valid_split():
    random.seed(0)
    dir = '../data/CorpNet/{}'
    for year in range(2003,2022):
        triple = pd.read_csv(os.path.join(dir.format(year),'triple2id.txt'), sep='\t',names=['head','relation','tail'])
        n_row = triple.shape[0]
        random_indices =  random.sample(range(0,n_row),n_row)
        train = triple.loc[random_indices[:int(n_row * 0.8)], :]
        valid = triple.loc[random_indices[int(n_row * 0.8):int(n_row * 0.9)], :]
        # test 数据集，用于每个几轮验证模型，而valid数据集才是在最后去测试模型
        test = triple.loc[random_indices[int(n_row * 0.9):], :]
        train.to_csv(os.path.join(dir.format(year),'train.txt'), sep='\t', header=None, index=None)
        valid.to_csv(os.path.join(dir.format(year), 'valid.txt'), sep='\t', header=None, index=None)
        test.to_csv(os.path.join(dir.format(year), 'test.txt'), sep='\t', header=None, index=None)
        print('%d年的数据已划分:训练集%d条，验证集%d条，测试集%d条' % (year, train.shape[0], valid.shape[0], test.shape[0]))


if __name__ == '__main__':

    with open('../../meta-path/listed_firm.json', 'r') as f:
        listed_corp = json.load(f)
    with open('../../meta-path/unlisted_firm.json', 'r') as f:
        unlisted_corp = json.load(f)
    with open('../../meta-path/people.json', 'r') as f:
        people = json.load(f)
    all_body = listed_corp + unlisted_corp + people

    # 生成标准文件entity2id.txt: entity \t index \n
    # generate_entity2id(all_body)
    # 生成标准文件relation2id.txt: relation \t index \n
    # generate_relation()
    '''entity2id.txt, relation2id.txt所有年份共用'''
    # 生成标准文件triple2id.txt head \t relation \t tail \n 格式
    # generate_triple(all_body)

    train_test_valid_split()
