import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):
    """Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.

    Args:
        columns: List of names corresponding to the matrices in data.
        data: List of n-dimensional data matrices ordered in correspondence with columns.
            All matrices must have the same leading dimension.  Data can also be fed a list of
            instances of np.memmap, in which case RAM usage can be limited to the size of a
            single batch.
    """

    def __init__(self, columns, data):
        assert len(columns) == len(data), 'columns length does not match data length'
        # 验证行数相等
        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths)) == 1, 'all matrices in data must have same first dimension'
        # 行数 145063
        self.length = lengths[0]
        # data,is_nan,page_id,project,access,agent,test,test_is_nan
        self.columns = columns
        self.data = data
        # 数据字典
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length)

    def shapes(self):
        # 每列数据的shape
        return pd.Series(dict(zip(self.columns, [mat.shape for mat in self.data])))

    def dtypes(self):
        # 每列数据的dtype
        return pd.Series(dict(zip(self.columns, [mat.dtype for mat in self.data])))

    def shuffle(self):
        # 改变idx的顺序
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(10000)):
        # 划分训练集 137809 7254
        train_idx, test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)
        # 深度拷贝原矩阵
        train_df = DataFrame(copy.copy(self.columns), [mat[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat[test_idx] for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()
            # 错误 +1
            for i in range(0, self.length + 1, batch_size):
                batch_idx = self.idx[i: i + batch_size]
                # 最后一批 <batch
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    break
                yield DataFrame(columns=copy.copy(self.columns), data=[mat[batch_idx].copy() for mat in self.data])

            epoch_num += 1

    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def __iter__(self):
        # for循环
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value
