from transformers import BertConfig, BertModel, AutoTokenizer, AutoModel
import pandas as pd
import torch

def call_bert(model):
    all_mda_embs = torch.tensor([]).to(DEVICE)
    for i in mda.index:
        Stkcd = mda.Scode[i]
        text = mda.BusDA[i]
        # inputs 是分成一段一段的文本，因为bert最大只能处理 512 个token
        inputs = []
        cur_sentence = ''
        for s in text.split(' '):
            if len(cur_sentence + s) <= MAX_SUBSEN_LEN:
                cur_sentence += s
            else:
                inputs.append(cur_sentence)
                cur_sentence = s
        
        # 手动加上最后一段
        if cur_sentence != '':
            inputs.append(cur_sentence)
        
        # 开始调用 tokenizer，把汉字转成token，再调用model，提出CLS token的embedding
        embs = []
        for section in inputs:
            encoded_input = tokenizer(section, return_tensors='pt',padding=True).to(DEVICE)
            output = model(**encoded_input)
            # 提取出每一个句子的 CLS 向量
            # output shape: [1, n_token, emb_size]
            embs.append(output.last_hidden_state[:,0,:].squeeze().tolist())
        embs = torch.tensor(embs).to(DEVICE)

        pooling_layer = torch.nn.MaxPool1d(kernel_size=embs.shape[0])
        merged = pooling_layer(embs.T).T
        
        all_mda_embs = torch.concat([all_mda_embs, merged], dim=0)
        print('{} is done'.format(Stkcd))
    return all_mda_embs.data.cpu()


if __name__=='__main__':
    global mda
    global DEVICE
    global MAX_SUBSEN_LEN

    mda = pd.read_excel('../年报/管理层讨论与分析/mda_no_number.xlsx')
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # MAX_SUBSEN_LEN: Bert 单次处理的最大文本长度
    MAX_SUBSEN_LEN = 500

    '''configuration = BertConfig()
    model = BertModel(configuration)
    model = BertModel(configuration)
    configuration = model.config'''

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese").to(DEVICE)

    print('*' * 10 + ' Bert starts inference ... '+ '*' * 10)
    all_mda_embs = call_bert(model)
    torch.save(all_mda_embs, 'mda_embs.pt')