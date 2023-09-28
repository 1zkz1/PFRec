import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import torch.nn as nn
class DataProcess(nn.Module):

    def __init__(self, dataset_path: str, data_names:list, headers: list):
        super(DataProcess, self).__init__()
        self.dataset_path = dataset_path
        self.data_names = data_names
        self.headers = headers
        self.data_dict = {}
        self.data = pd.DataFrame()
        self.n_users = self.n_items = 0

        if len(data_names) != len(headers):
            print('')

    def load_raw_data(self):
        path = self.dataset_path
        data_names = self.data_names
        headers = self.headers

        for i, data_name in enumerate(data_names):
            header = headers[i]
            print('load {}'.format(data_name))
            self.data_dict[data_name] = pd.read_csv(r'{}/{}'.format(self.dataset_path, data_name), sep='::',
                                       engine='python', names=header, encoding='ISO-8859-1')
            print('shape:', self.data_dict[data_name].shape)
        print('load finish!')
        return self.data_dict

    def process_data(self):
        if self.data_dict == {}:
            self.load_raw_data()

        data = self.data_dict
        users = data['users.dat']
        ratings = data['ratings.dat']
        movies = data['movies.dat']
        data = pd.merge(users, ratings, on='UserID')
        data = data[['UserID', 'MovieID', 'Rating']]
        data = data.drop_duplicates(subset=['UserID', 'MovieID'], keep='last')
        # print('data:\n', data)

        data = data[data.groupby('UserID')['UserID'].transform('count') >= 50]
        # print('data:\n', data)
        data = data[data.groupby('MovieID')['MovieID'].transform('count') >= 50]
        # print('data:\n', data)

        unique_user = data['UserID'].unique()
        user2id = pd.Series(data=range(len(unique_user)), index=unique_user)
        data['UserID'] = data['UserID'].map(user2id)

        unique_item = data['MovieID'].unique()
        item2id = pd.Series(data=range(len(unique_item)), index=unique_item)
        data['MovieID'] = data['MovieID'].map(item2id)

        # print('data:\n', data)
        self.n_users = len(data['UserID'].unique())
        print('user_nums:', self.n_users, 'range：[{}, {}]'.format(min(data['UserID']), max(data['UserID'])))
        self.n_items = len(data['MovieID'].unique())
        print('item_nums:', self.n_items, 'range：[{}, {}]'.format(min(data['MovieID']), max(data['MovieID'])))

        data = data.rename(columns={'UserID': 'userid', 'MovieID': 'itemid', 'Rating': 'rating'})
        # print(data)
        self.data = data.reset_index(drop=True)

        print('process data finish!')
        return self.data

    def split_by_users_data(self, ratio=0.8):
        if len(self.data) <= 0:
            self.process_data()

        train = pd.DataFrame(columns=self.data.columns)
        test = pd.DataFrame(columns=self.data.columns)

        a = self.data.groupby('userid')
        for i in range(len(a)):
            temp = a.get_group(i)
            # print('temp:', temp)
            length = int(len(temp) * ratio)
            train = pd.concat([train, temp.iloc[0: length]])
            test = pd.concat([test, temp.iloc[length:]])
            # print('train.shape:', train.shape, 'test.shape', test.shape)
            # print('train:\n', train, '\n test:\n', test)

        # print('train.shape:', train.shape, 'test.shape', test.shape)
        print('split data finish!')
        return train, test

    @classmethod
    def _split_by_users_data(cls, data, ratio=0.8):
        if len(data) <= 0 :
            print('data.length is 0')
            return

        train = pd.DataFrame(columns=data.columns)
        test = pd.DataFrame(columns=data.columns)

        a = data.groupby('UserID')
        for i in range(0, len(a)):
            temp = a.get_group(i)
            # print('temp:', temp)
            length = int(len(temp) * ratio)
            train = pd.concat([train, temp.iloc[0: length]])
            test = pd.concat([test, temp.iloc[length:]])
            # print('train.shape:', train.shape, 'test.shape', test.shape)
            # print('train:\n', train, '\n test:\n', test)

        # print('train.shape:', train.shape, 'test.shape', test.shape)
        print('split data finish!')
        return train, test

    def getDataDict(self, name='userid'):
        if len(self.data) <= 0:
            self.process_data()

        dict = {}
        a = self.data.groupby(name)
        for i, v in a:
            # temp = a.get_group(i)
            dict[i] = v.reset_index().drop('index', axis=1)
        # for k, v in dict.items():
        #     print('k:', k)
        #     print('v:', v)
        return dict

    def construct_one_valued_matrix(self, processed, item_based=False):
        if self.n_users == 0 or self.n_items == 0:
            self.process_data()

        n_users = self.n_users
        n_items = self.n_items

        if item_based:
            return csr_matrix((np.ones_like(processed.rating.values),
                               (processed.itemid, processed.userid)),
                              shape=(n_items, n_users),
                              dtype='float32')

        return csr_matrix((np.ones_like(processed.rating.values),
                           (processed.userid.values, processed.itemid.values)),
                          shape=(n_users, n_items),
                          dtype='float32')

    def construct_real_matrix(self, processed, item_based=False, low=5.):
        if self.n_users == 0 or self.n_items == 0:
            self.process_data()

        n_users = self.n_users
        n_items = self.n_items

        processed.rating = processed.rating / low
        # processed.rating = round(processed.rating / low, 2)

        if item_based:
            return csr_matrix((processed.rating,
                           (processed.itemid, processed.userid)),
                          shape=(n_items, n_users),
                          dtype='float32')

        return csr_matrix((processed.rating,
                           (processed.userid, processed.itemid)),
                          shape=(n_users, n_items),
                          dtype='float32')

    def forward(self):
        self.load_raw_data()
        return self.process_data()

    def save_to_csv(self, path='.', name='data'):
        if len(self.data) <= 0:
            self.process_data()

        data = self.data
        location = '{}/{}.csv'.format(path, name)
        data.to_csv(path_or_buf=location, index=False)
        print('write data to {}.csv'.format(name))


class MyDataloader():

    def __init__(self, path=r'data', dataset_name=r'Amazon-beauty', fold=1):

        super(MyDataloader, self).__init__()
        self.path = path
        self.dataset_name = dataset_name
        self.fold = fold
        self.train_dataset = self.test_dataset = self.ratings = pd.DataFrame
        self.n_users = 0
        self.n_items = 0
        self.loader(fold)

    def loader(self, fold=1):
        print(r'load {}\{}\train_df_{}.csv'.format(self.path, self.dataset_name, fold))
        self.train_dataset = pd.read_csv(r'{}\{}\train_df_{}.csv'.format(self.path, self.dataset_name, fold))
        print(r'load {}\{}\test_df_{}.csv'.format(self.path, self.dataset_name, fold))
        self.test_dataset = pd.read_csv(r'{}\{}\test_df_{}.csv'.format(self.path, self.dataset_name, fold))
        self.ratings = pd.concat((self.train_dataset, self.test_dataset))
        self.n_users = len(self.ratings['user'].unique())
        self.n_items = len(self.ratings['item'].unique())

    def reloader(self, train_pd: pd.DataFrame, test_pd: pd.DataFrame):
        print('reload dataset')
        self.train_dataset = train_pd
        self.test_dataset = test_pd
        self.ratings = pd.concat((self.train_dataset, self.test_dataset))
        self.n_users = len(self.ratings['user'].unique())
        self.n_items = len(self.ratings['item'].unique())

    def get_dataset(self):
        return self.train_dataset, self.test_dataset

    def get_n_users(self):
        return self.n_users

    def get_n_items(self):
        return self.n_items

    def construct_one_valued_matrix(self, processed, item_based=False):
        n_users = self.n_users
        n_items = self.n_items

        if item_based:
            return csr_matrix((np.ones_like(processed.rating.values),
                               (processed.item, processed.user)),
                              shape=(n_items, n_users),
                              dtype='float32')

        return csr_matrix((np.ones_like(processed.rating.values),
                           (processed.user.values, processed.item.values)),
                          shape=(n_users, n_items),
                          dtype='float32')

    def construct_real_matrix(self, processed, item_based=False, low=5.):
        n_users = self.n_users
        n_items = self.n_items

        processed.rating = processed.rating / low

        if item_based:
            return csr_matrix((processed.rating,
                           (processed.item, processed.user)),
                          shape=(n_items, n_users),
                          dtype='float32')

        return csr_matrix((processed.rating,
                           (processed.user, processed.item)),
                          shape=(n_users, n_items),
                          dtype='float32')


def remap_id(temp, col_name: str):
    unique_user = temp[col_name].unique()
    print('len:', len(unique_user))
    user2id = pd.Series(data=range(len(unique_user)), index=unique_user)
    temp[col_name] = temp[col_name].map(user2id)
    return temp



