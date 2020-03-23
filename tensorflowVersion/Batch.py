import pickle

import numpy as np


class BatchGenerator(object):
    """ Construct a Data generator. The input X, y should be ndarray or list like type.
    
    Example:
        Data_train = BatchGenerator(X=X_train_all, y=y_train_all, shuffle=False)
        Data_test = BatchGenerator(X=X_test_all, y=y_test_all, shuffle=False)
        X = Data_train.X
        y = Data_train.y
        or:
        X_batch, y_batch = Data_train.next_batch(batch_size)
     """ 
    
    def __init__(self, X, y, shuffle=False):
        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self._X = X
        """ _X.shape=(24271, 60), _X.shape[0]=24271
        [[ 53 753 237 ...   0   0   0]
         [ 51 523  71 ...   0   0   0]
         [169   2 392 ...   0   0   0]
         ...
         [ 61  47 302 ...   0   0   0]
         [426 201 578 ...   0   0   0]
         [180  16  20 ...   0   0   0]]
        """
        self._y = y
        """  _y.shape=(24271, 60), _y.shape[0]=24271
        [[10 10 10 ...  0  0  0]
         [10 10 10 ...  0  0  0]
         [10 10 10 ...  0  0  0]
         ...
         [10  9  4 ...  0  0  0]
         [ 2  5  8 ...  0  0  0]
         [10  3  7 ...  0  0  0]]
        """
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]  # _X.shape[0]就是矩阵的行数
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)  # 重新排列[0, self._numuber_example]，形成一个列表
            self._X = self._X[new_index]  # 重新排列_X成一个二维矩阵，每行的内容不变，行与行之间的顺序打乱
            self._y = self._y[new_index]  # 重新排列_y成一个二维矩阵，每行的内容不变，行与行之间的顺序打乱
                
    @property  # @property：把方法变成属性来使用  obj.x就行了，不用obj.x()
    def x(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def num_examples(self):  # 矩阵的行数
        return self._number_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size):
        """ Return the next 'batch_size' examples from this data set.
            batch_size为选择矩阵的行数
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            # finished epoch
            self._epochs_completed += 1
            # Shuffle the data 
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples  # 防止一次取出的batch_size大于矩阵自身的总行数
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]  # 从start行到end-1行的数据


if __name__ == '__main__':
    with open('../data/renmindata.pkl', 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
    data_train = BatchGenerator(x_train, y_train, shuffle=True)
