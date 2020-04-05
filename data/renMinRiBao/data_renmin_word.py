"""
if tag == 'nr' or tag == 'ns' or tag == 'nt':  # 如果tag是人名 地名 机构团体中的
在人民日报的文件中总共有3个实体类：nr 人名, ns 地名, nt 机构团体名, O 什么实体也不是
"""
import codecs
import re
import pdb
import pandas as pd
import numpy as np
import collections
import pickle

from sklearn.model_selection import train_test_split


def origin_handle():
    line_num = 0
    with open('./renmin.txt', 'r') as inp, open('./renmin2.txt', 'w') as outp:
        for line in inp.readlines():
            line = line.split('  ')  # txt文档中word/tag的组合是以两个空格分隔的
            i = 1  # 第0个word/tag的组合是日期，就忽略了，所以从1开始算起
            while i < len(line) - 1:  # 每一句话最后一个字符都是\n，所以处理line的时候不处理最后一个
                if line[i][0] == '[':  # 如果line中第i个单词的第0个字符是'['，表示开始一个组合词
                    outp.write(line[i].split('/')[0][1:])  # 单词部分从'['开始,把word写入文件
                    i += 1  # 继续下一个word/tag
                    while i < len(line) - 1 and line[i].find(']') == -1:  # 如果没有扫描到这个组合词的结尾
                        if line[i] != '':
                            outp.write(line[i].split('/')[0])  # 如果word/tag不为空，就把word部分写入文件
                        i += 1
                    # 扫描到了组合词的结尾('[日本/ns', '政府/n]nt')
                    outp.write(line[i].split('/')[0].strip() + '/' + line[i].split('/')[1][-2:] + ' ')  # -2表示取组合词的tag
                elif line[i].split('/')[1] == 'nr':  # 如果tag是人名
                    word = line[i].split('/')[0]  # 人名
                    i += 1  # 继续下一个word/tag
                    if i < len(line) - 1 and line[i].split('/')[1] == 'nr':  # 如果下一个word/tag的tag也代表人名('王/nr', '林昌/nr')
                        outp.write(word + line[i].split('/')[0] + '/nr ')
                    else:
                        outp.write(word + '/nr ')
                        continue
                else:
                    outp.write(line[i] + ' ')
                i += 1
            outp.write('\n')
            line_num += 1
            if line_num != 0 and line_num % 200 == 0:
                print('处理完第{}行'.format(line_num))


def origin_handle2():
    with codecs.open('./renmin2.txt', 'r', 'utf-8') as inp, codecs.open('./renmin3.txt', 'w', 'utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0  # 没有了表示时间的word/tag，i可以从0开始计算了
            while i < len(line) - 1:
                if line[i] == '':
                    i += 1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                if tag == 'nr' or tag == 'ns' or tag == 'nt':  # 如果tag是人名 地名 机构团体中的
                    outp.write(word[0] + "/B_" + tag + " ")
                    for j in word[1:len(word) - 1]:
                        if j != ' ':
                            outp.write(j + "/M_" + tag + " ")
                    outp.write(word[-1] + "/E_" + tag + " ")
                else:
                    for wor in word:
                        outp.write(wor + '/O ')  # 'O'表示不是nr ns nt中的任何一个
                i += 1
            outp.write('\n')


def sentence2split():  # renmin4.txt中有2172条数据
    with open('./renmin3.txt', 'r') as inp, codecs.open('./renmin4.txt', 'w', 'utf-8') as outp:
        # texts = inp.read().decode('utf-8')
        texts = inp.read()  # 所有文本
        # sentences = re.split('[，。！？、‘’“”:]/[O]'.decode('utf-8'), texts)
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)

        line_num = 0  # renmin4.txt中文本的行数
        for sentence in sentences:
            if sentence != " ":
                outp.write(sentence.strip() + '\n')
                line_num += 1
                if line_num >= 2000:  # 先拿2000条数据作为train和test的集合
                    break


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def check_list(x):
    for el in x:
        if not isinstance(el, str):
            print(el)


def my_data2pkl():
    # datas = []  # [[第一行的所有单字], [第二行的所有单字], [第三行的所有单字], .....]
    document_words = []  # [[第一行的所有word], [第二行的所有word], [第三行的所有word], .....]
    # labels = []  # [[第一行的所有tag], [第二行的所有tag], [第三行的所有tag], ...]
    document_tags = []  # [[第一行的所有tag], [第二行的所有tag], [第三行的所有tag], ...]
    # linedata = []
    # linelabel = []
    tags = set()  # 存储整个文本的所有tag
    # tags.add('')
    input_data = codecs.open('renmin4.txt', 'r', 'utf-8')
    line_num = 0

    for line in input_data.readlines():
        # line = line.split()  # 把一行文本切成单个字/tag的形式
        # 把一行文本切成单个字/tag的形式
        list_of_word_tag = line.split()  # ['迈/O', '向/O', '充/O', '满/O', '希/O', '望/O', '的/O', '新/O', '世/O', '纪/O', ...]

        # linedata = []  # 存放所有的单字
        # linelabel = []  # 存放所有的标签
        list_of_word_in_line = []  # 存放所有的word
        list_of_tag_in_line = []  # 存放所有的tag
        numNotO = 0  # 一行文本中非'O'的实体的数量
        for word_tag in list_of_word_tag:
            # word = word.split('/')
            _word = word_tag.split('/')[0]
            _tag = word_tag.split('/')[1]
            # linedata.append(word[0])
            # linelabel.append(word[1])
            list_of_word_in_line.append(_word)  # 把分离出来的一个word放入存储每行word的列表
            list_of_tag_in_line.append(_tag)  # # 把分离出来的一个tag放入存储每行tag的列表
            # tags.add(word[1])  # 存放所有tag的列表中加入该字对应的tag
            tags.add(_tag)  # 存放整个文本tag的set中加入分离出来的tag
            # if word[1] != 'O':
            #     numNotO += 1
            if _tag != 'O':  # 如果word对应的tag不是'O'
                numNotO += 1
        if numNotO != 0:  # 如果一行文本中存在实体
            # datas.append(linedata)
            document_words.append(list_of_word_in_line)  # 把本行的word列表加入存放整个文本word的列表
            # labels.append(linelabel)
            document_tags.append(list_of_tag_in_line)  # # 把本行的tag列表加入存放整个文本tag的列表
        line_num += 1

    input_data.close()
    print('document_words的长度是{}'.format(len(document_words)))  # 638 使用完整的renmin4.txt时是37924行
    print('document_tags的长度是{}'.format(len(document_tags)))  # 638 使用完整的renmin4.txt时是37924行
    print('renmin4.txt总共有{}行'.format(line_num))  # 2172
    print('所有文本中出现过的tag有 {}'.format(str(tags)))  # {'M_nr', 'M_ns', 'E_ns', 'B_nt', 'B_nr', 'M_nt', 'E_nt', 'O', 'B_ns', 'E_nr'}

    word_to_ix = {}  # {'中': 0, '共': 1, '央': 2, ......}
    for words in document_words:
        # words为列表，存放每行的word
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print('字典形式的word_to_ix的长度是{}'.format(len(word_to_ix)))

    ix_to_word = {}
    for key, val in word_to_ix.items():
        ix_to_word[val] = key
    print('字典形式的ix_to_word的长度是{}'.format(len(ix_to_word)))

    tag_to_ix = {}  # {'B_nt': 0, 'M_nt': 1, ......}
    for tags in document_tags:
        # tags为列表，存放每行的tag
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    # 加入开始和结束标签
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)

    print('字典形式的tag_to_ix的长度是{}'.format(len(tag_to_ix)))
    print(tag_to_ix)

    ix_to_tag = {}
    for key, val in tag_to_ix.items():
        ix_to_tag[val] = key
    print('字典形式的ix_to_tag的长度是{}'.format(len(ix_to_tag)))
    print(ix_to_tag)

    #
    X_train, X_test, y_train, y_test = train_test_split(document_words, document_tags, test_size=0.1, random_state=1)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    print('训练数据的长度: {}'.format(len(X_train)))  # 选取前2000条时：574 完整的：34131
    print('测试数据的长度: {}'.format(len(X_test)))  # 选取前2000条时：64 完整的：3793

    training_data = []
    for ix in range(len(X_train)):
        list_of_word_in_line = X_train[ix]
        list_of_tag_in_line = y_train[ix]
        training_data.append((list_of_word_in_line, list_of_tag_in_line))
    print('training_data的类型是 {} 长度是 {}'.format(str(type(training_data)), str(len(training_data))))  # list, 574

    testing_data = []
    for ix in range(len(X_test)):
        list_of_word_in_line = X_test[ix]
        list_of_tag_in_line = y_test[ix]
        testing_data.append((list_of_word_in_line, list_of_tag_in_line))
    print('testing_data的类型是 {} 长度是 {}'.format(str(type(testing_data)), str(len(testing_data))))  # list, 64

    with open('../renmindata.pkl', 'wb') as outp:
        pickle.dump(word_to_ix, outp)
        pickle.dump(ix_to_word, outp)
        pickle.dump(tag_to_ix, outp)
        pickle.dump(ix_to_tag, outp)
        pickle.dump(training_data, outp)
        pickle.dump(testing_data, outp)
    print('** Finish saving data to renmindata.pkl.')


# def data2pkl():
#     datas = list()  # [[第一行的所有单字], [第二行的所有单字], [第三行的所有单字], .....]
#     labels = list()  # [[第一行的所有tag], [第二行的所有tag], [第三行的所有tag], ...]
#     linedata = list()
#     linelabel = list()
#     tags = set()
#     tags.add('')
#     input_data = codecs.open('renmin4.txt', 'r', 'utf-8')
#     line_num = 0
#
#     for line in input_data.readlines():
#         line = line.split()  # 把一行文本切成单个字/tag的形式
#         linedata = []  # 存放所有的单字
#         linelabel = []  # 存放所有的标签
#         numNotO = 0  # 非'O'的字的数量
#         for word in line:
#             word = word.split('/')
#             linedata.append(word[0])
#             linelabel.append(word[1])
#             tags.add(word[1])  # 存放所有tag的列表中加入该字对应的tag
#             if word[1] != 'O':
#                 numNotO += 1
#         if numNotO != 0:
#             datas.append(linedata)
#             labels.append(linelabel)
#         line_num += 1
#
#     input_data.close()
#     # print('datas的长度是{}'.format(len(datas)))  # datas的长度是37924
#     # print('labels的长度是{}'.format(len(labels)))  # labels的长度是37924
#     # print('renmin4.txt总共有{}行'.format(line_num))  # renmin4.txt总共有154949行
#     # 总共有37924行至少出现ns nr nt中的一种
#
#     # flatten这个方式是python2中的，在python3中已经没有了
#     # flatten的作用:
#     # >>> flatten(["junk",["nested stuff"],[],[[]]])
#     # ['junk', 'nested stuff']
#     all_words = flatten(datas)
#     sr_allwords = pd.Series(all_words)
#     sr_allwords = sr_allwords.value_counts()  # 元素和元素出现的次数
#     """
#     的    15101
#     国    10952
#     中     7540
#     在     5192
#     １     4888
#          ...
#     碟        1
#     曜        1
#     栅        1
#     镭        1
#     怂        1
#     Length: 3916, dtype: int64
#     """
#     # print(sr_allwords)
#     set_words = sr_allwords.index  # 所有的元素
#     """
#     Index(['的', '国', '中', '在', '１', '一', '会', '大', '人', '日',
#        ...
#        '盂', '珣', '绶', '媲', '淬', '碟', '曜', '栅', '镭', '怂'],
#       dtype='object', length=3916)
#     """
#     # print(set_words)
#     set_ids = range(1, len(set_words) + 1)  # range(1, 3917)
#
#     tags = [i for i in tags]  # 把set型的tags转换为list型的tags
#     # ['', 'O', 'E_nr', 'B_nt', 'B_ns', 'M_ns', 'B_nr', 'M_nr', 'E_ns', 'E_nt', 'M_nt']
#
#     tag_ids = range(len(tags))  # range(0, 11)
#
#     word2id = pd.Series(set_ids, index=set_words)  # 数字序号对应字，从1开始
#     """
#     的       1
#     国       2
#     中       3
#     在       4
#     １       5
#          ...
#     蔷    3912
#     稔    3913
#     飒    3914
#     湄    3915
#     濉    3916
#     Length: 3916, dtype: int64
#     """
#     id2word = pd.Series(set_words, index=set_ids)  # 单字对应id，从1开始,与word2id相反
#     """
#     1       的
#     2       国
#     3       中
#     4       在
#     5       １
#            ..
#     3912    饥
#     3913    盱
#     3914    荼
#     3915    弭
#     3916    卞
#     Length: 3916, dtype: object
#     """
#     tag2id = pd.Series(tag_ids, index=tags)
#     """
#              0
#     M_nt     1
#     E_nt     2
#     E_nr     3
#     B_nr     4
#     M_ns     5
#     E_ns     6
#     B_ns     7
#     M_nr     8
#     B_nt     9
#     O       10
#     dtype: int64
#     """
#     id2tag = pd.Series(tags, index=tag_ids)
#     """
#     0
#     1     M_nt
#     2     E_nt
#     3     E_nr
#     4     B_nr
#     5     M_ns
#     6     E_ns
#     7     B_ns
#     8     M_nr
#     9     B_nt
#     10       O
#     """
#
#     word2id["unknow"] = len(word2id) + 1
#     """
#     的            1
#     国            2
#     中            3
#     在            4
#     １            5
#               ...
#     臧         3913
#     讳         3914
#     揆         3915
#     龟         3916
#     unknow    3917  -> 新加的
#     Length: 3917, dtype: int64
#     """
#     id2word[len(word2id)] = "unknow"  # 在上一行中len(word2id)已经增加1了
#     """
#     1            的
#     2            国
#     3            中
#     4            在
#     5            １
#              ...
#     3913         臧
#     3914         讳
#     3915         揆
#     3916         龟
#     3917    unknow
#     Length: 3917, dtype: object
#     """
#
#     max_len = 60
#
#     # print(word2id[['的', '国']])
#     """
#     的    1
#     国    2
#     dtype: int64
#     """
#
#     # print(list(word2id[['的', '国']]))  # [1, 2]
#
#     def x_padding(words):
#         """
#         Returns: words列表对应的ids列表
#         """
#         debug = 1
#         ids = list(word2id[words])  # 查找words列表中每一个单字的id，并把所有id存储在ids列表中
#         if len(ids) >= max_len:
#             return ids[:max_len]  # 返回前max_len个
#         ids.extend([0] * (max_len - len(ids)))  # 不足max_len，就用0填充(原本的word2id中的id是从1开始的,所以0不会发生重复）
#         return ids
#
#     def y_padding(tags):
#         """
#         Returns: tags列表对应的ids列表
#         """
#         ids = list(tag2id[tags])
#         if len(ids) >= max_len:
#             return ids[:max_len]
#         ids.extend([0] * (max_len - len(ids)))
#         return ids
#
#     df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
#     """
#                                                        words                                               tags
#     0                                  [中, 共, 中, 央, 总, 书, 记]                  [B_nt, M_nt, M_nt, E_nt, O, O, O]
#     1                                  [国, 家, 主, 席, 江, 泽, 民]                     [O, O, O, O, B_nr, M_nr, E_nr]
#     2                                  [中, 共, 中, 央, 总, 书, 记]                  [B_nt, M_nt, M_nt, E_nt, O, O, O]
#     3      [国, 家, 主, 席, 江, 泽, 民, 发, 表, １, ９, ９, ８, 年, 新, ...  [O, O, O, O, B_nr, M_nr, E_nr, O, O, O, O, O, ...
#     4                      [（, 新, 华, 社, 记, 者, 兰, 红, 光, 摄, ）]  [O, B_nt, M_nt, E_nt, O, O, B_nr, M_nr, E_nr, ...
#     ...                                                  ...                                                ...
#     37919                        [见, 过, 河, 东, 河, 西, 的, 变, 迁]            [O, O, B_ns, E_ns, B_ns, E_ns, O, O, O]
#     37920                                          [刘, 志, 辉]                                 [B_nr, M_nr, E_nr]
#     37921                                    [（, 马, 东, 生, ）]                           [O, B_nr, M_nr, E_nr, O]
#     37922                                    [（, 牛, 沛, 岩, ）]                           [O, B_nr, M_nr, E_nr, O]
#     37923                                          [段, 和, 平]                                 [B_nr, M_nr, E_nr]
#
#     [37924 rows x 2 columns]
#     """
#
#     # print(df_data['words'])
#     """
#     0                                    [中, 共, 中, 央, 总, 书, 记]
#     1                                    [国, 家, 主, 席, 江, 泽, 民]
#     2                                    [中, 共, 中, 央, 总, 书, 记]
#     3        [国, 家, 主, 席, 江, 泽, 民, 发, 表, １, ９, ９, ８, 年, 新, ...
#     4                        [（, 新, 华, 社, 记, 者, 兰, 红, 光, 摄, ）]
#                                    ...
#     37919                          [见, 过, 河, 东, 河, 西, 的, 变, 迁]
#     37920                                            [刘, 志, 辉]
#     37921                                      [（, 马, 东, 生, ）]
#     37922                                      [（, 牛, 沛, 岩, ）]
#     37923                                            [段, 和, 平]
#     Name: words, Length: 37924, dtype: object
#     """
#
#     df_data['x'] = df_data['words'].apply(x_padding)
#     """
#     0        [3, 159, 3, 118, 50, 149, 21, 0, 0, 0, 0, 0, 0...
#     1        [2, 22, 40, 160, 62, 223, 17, 0, 0, 0, 0, 0, 0...
#     2        [3, 159, 3, 118, 50, 149, 21, 0, 0, 0, 0, 0, 0...
#     3        [2, 22, 40, 160, 62, 223, 17, 42, 83, 5, 44, 4...
#     4        [13, 16, 20, 45, 21, 30, 325, 444, 340, 303, 1...
#                                    ...
#     37919    [307, 192, 177, 61, 177, 70, 1, 699, 1355, 0, ...
#     37920    [171, 106, 824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
#     37921    [13, 181, 61, 72, 14, 0, 0, 0, 0, 0, 0, 0, 0, ...
#     37922    [13, 1157, 1506, 1212, 14, 0, 0, 0, 0, 0, 0, 0...
#     37923    [512, 11, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
#     Name: x, Length: 37924, dtype: object
#     """
#
#     df_data['y'] = df_data['tags'].apply(y_padding)
#     """
#     0        [3, 2, 2, 1, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, ...
#     1        [8, 8, 8, 8, 9, 6, 10, 0, 0, 0, 0, 0, 0, 0, 0,...
#     2        [3, 2, 2, 1, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, ...
#     3        [8, 8, 8, 8, 9, 6, 10, 8, 8, 8, 8, 8, 8, 8, 8,...
#     4        [8, 3, 2, 1, 8, 8, 9, 6, 10, 8, 8, 0, 0, 0, 0,...
#                                    ...
#     37919    [8, 8, 4, 5, 4, 5, 8, 8, 8, 0, 0, 0, 0, 0, 0, ...
#     37920    [9, 6, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
#     37921    [8, 9, 6, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
#     37922    [8, 9, 6, 10, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
#     37923    [9, 6, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...
#     Name: y, Length: 37924, dtype: object
#     """
#
#     # print(list(df_data['x'].values))  # [[3, 159, 3, 118, 50, ...], [[2, 22, 40, 160, 62, 223, ...], ...]
#
#     x = np.asarray(list(df_data['x'].values))
#     """ x是二维矩阵 行数为datas的长度，每一行的内容是word对应的id
#     [[   3  159    3 ...    0    0    0]
#      [   2   22   40 ...    0    0    0]
#      [   3  159    3 ...    0    0    0]
#      ...
#      [  13  181   61 ...    0    0    0]
#      [  13 1164 1498 ...    0    0    0]
#      [ 512   11  122 ...    0    0    0]]
#     """
#     y = np.asarray(list(df_data['y'].values))
#     """ y是一个二维矩阵 shape为(37924, 60)
#     [[ 4  1  1 ...  0  0  0]
#      [10 10 10 ...  0  0  0]
#      [ 4  1  1 ...  0  0  0]
#      ...
#      [10  9  5 ...  0  0  0]
#      [10  9  5 ...  0  0  0]
#      [ 9  5  6 ...  0  0  0]]
#     """
#
#     from sklearn.model_selection import train_test_split
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
#     # print(x_train)
#     """ 
#     [[ 130   64  228 ...    0    0    0]
#      [ 128  804 1738 ...    0    0    0]
#      [   2   22  242 ...    0    0    0]
#      ...
#      [1500 1001  391 ...    0    0    0]
#      [ 177   47   69 ...    0    0    0]
#      [ 491  610  247 ...    0    0    0]]
#     """
#     # print(y_train)
#     """
#     [[9 9 9 ... 0 0 0]
#      [8 6 6 ... 0 0 0]
#      [9 9 9 ... 0 0 0]
#      ...
#      [5 2 7 ... 0 0 0]
#      [8 6 6 ... 0 0 0]
#      [3 4 4 ... 0 0 0]]
#     """
#     # print(x_test)
#     """
#     [[   6   77    3 ...    0    0    0]
#      [  16   20   45 ...    0    0    0]
#      [ 267  316  104 ...    0    0    0]
#      ...
#      [ 826  401  576 ...    0    0    0]
#      [ 114  625 1161 ...    0    0    0]
#      [  23 1322  381 ...    0    0    0]]
#     """
#     # print(y_test)
#     """
#     [[9 9 3 ... 0 0 0]
#      [8 6 1 ... 0 0 0]
#      [5 2 2 ... 0 0 0]
#      ...
#      [5 2 7 ... 0 0 0]
#      [9 9 9 ... 0 0 0]
#      [9 9 9 ... 0 0 0]]
#      """
#     x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)
#
#     import pickle
#     import os
#     with open('../renmindata.pkl', 'wb') as outp:
#         pickle.dump(word2id, outp)
#         pickle.dump(id2word, outp)
#         pickle.dump(tag2id, outp)
#         pickle.dump(id2tag, outp)
#         pickle.dump(x_train, outp)
#         pickle.dump(y_train, outp)
#         pickle.dump(x_test, outp)
#         pickle.dump(y_test, outp)
#         pickle.dump(x_valid, outp)
#         pickle.dump(y_valid, outp)
#     print('** Finished saving the data.')


origin_handle()
origin_handle2()
sentence2split()
my_data2pkl()
