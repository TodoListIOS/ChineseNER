# coding=utf-8
import codecs

import torch


def calculate(origin_sentence, origin_tags, x, y, id2word, id2tag, res=[]):
    """
    sentence, predict, id2word, id2tag, entityres
    Args:
        x: sentence中的word列表转化成的tensor
        y: 预测的label组成的tensor
        id2word:
        id2tag:
        res:

    Returns:

    """
    debug = 1
    entity = []
    for j in range(len(x)):
        if x[j] == 0 or y[j] == 0:  # 0是凑数的，所以要略过去
            continue
        if id2tag[y[j]][0] == 'B':  # 7 -> B_company_name
            # print(id2word[x[j]])
            # print(id2tag[y[j]])
            # id2word[[x[j]]]返回的是一个汉字
            entity = [id2word[x[j]] + '/' + id2tag[y[j]]]
        elif id2tag[y[j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
            # 如果标签的j位置对应的实体名称是以M开头的，并且前面计算中entity中已经存在以B开头，中间是M的实体了
            # 并且entity中最后一项的'/'后（如M_company）从第1个字符开始算（_company）== 标签的j位置对应的实体名称的从第一个字符开始算
            entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
        elif id2tag[y[j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
            entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
            entity.append(str(j))
            res.append(entity)
            entity = []
        else:
            entity = []
    return res


def calculate3(x, y, id2word, id2tag, res=[]):
    '''
    使用这个函数可以把抽取出的实体写到res.txt文件中，供我们查看。
    注意，这个函数每次使用是在文档的最后添加新信息，所以使用时尽量删除res文件后使用。
    '''
    with codecs.open('./res.txt', 'a', 'utf-8') as outp:
        entity = []
        for j in range(len(x)):  # for every word
            if x[j] == 0 or y[j] == 0:
                continue
            if id2tag[y[j]][0] == 'B':
                entity = [id2word[x[j]] + '/' + id2tag[y[j]]]
            elif id2tag[y[j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
                entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
            elif id2tag[y[j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[j]][1:]:
                entity.append(id2word[x[j]] + '/' + id2tag[y[j]])
                entity.append(str(j))
                res.append(entity)
                st = ""
                for s in entity:
                    st += s + ' '
                # print st
                outp.write(st + '\n')
                entity = []
            else:
                entity = []
    return res
