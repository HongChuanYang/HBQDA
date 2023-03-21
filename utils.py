# coding: UTF-8
import imp
import pickle
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import numpy as np

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def cut_sentences(content):
    # 结束符号，包含中文和英文的
        end_flag = ['?', '!', '？', '！', '。', '…','，']

        content_len = len(content)
        sentences = []
        tmp_char = ''
        for idx, char in enumerate(content):
            # 拼接字符
            tmp_char += char

            # 判断是否已经到了最后一位
            if (idx + 1) == content_len:
                sentences.append(tmp_char)
                break
                
            # 判断此字符是否为结束符号
            if char in end_flag:
                # 再判断下一个字符是否为结束符号，如果不是结束符号，则切分句子
                next_idx = idx + 1
                if not content[next_idx] in end_flag:
                    sentences.append(tmp_char)
                    tmp_char = ''
                    
        return sentences
    


    def bert_load_dataset(config):
        train_raw_data = pd.read_csv(config.train_path)
        dev_raw_data = pd.read_csv(config.dev_path)
        test_raw_data = pd.read_csv(config.test_path)
        pd_query = pd.read_csv(config.query_path).dropna()
        print('train_raw_data len = {}, dev_raw_data len = {}, test_raw_data len = {}'.format(len(train_raw_data), len(dev_raw_data), len(test_raw_data)))
        # with open (config.word2vec_dict_path, 'rb') as d:
        #     id2word = pickle.load(d)
        # word2id = dict(zip(id2word.values(), id2word.keys()))
        # char_word_2id = word2id # use dict learned by word2vec
        def bert_data_process(raw_data, pad_size, doc_pad_size):
            contents = []
            # test001 = [] # delete
            # test_i = 0 #test
            for data in raw_data.itertuples(index=False):
                lin = data[5].strip()
                if not lin:
                    continue
                content = lin.replace('\r','').replace('\n','').replace('\u3000','').replace(' ','')
                label_dict = {301:'0', 302:'1', 303:'2', 304:'3', 305:'4'}
                label = int(label_dict[data[12]])
                # #将label进行Onehot编码
                # label = [0]*num_classes
                # label[label_num] = 1
                content_cut =  cut_sentences(content)
                sen_ids = []
                sen_len = []
                sen_masks = []
                for sen in content_cut:
                    # test001.append(sen)  # delete
                    token = config.tokenizer.tokenize(sen)
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = config.tokenizer.convert_tokens_to_ids(token)
                    if pad_size: # pading sentence
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size
                    # sen_ids.append((token_ids, seq_len, mask))
                    sen_ids.append(token_ids)
                    sen_len.append(seq_len)
                    sen_masks.append(mask)

                if doc_pad_size: # pading document
                    if len(sen_ids) < doc_pad_size:
                        sen_mask = [0] * pad_size
                        # pad_sent = (sen_mask, 0, sen_mask)
                        padding = doc_pad_size - len(sen_ids)
                        for i in range(padding):
                            sen_ids.append(sen_mask)
                            sen_len.append(0)
                            sen_masks.append(sen_mask)
                    else:
                        sen_ids = sen_ids[:doc_pad_size]            
                        sen_len = sen_len[:doc_pad_size]            
                        sen_masks = sen_masks[:doc_pad_size]   
                contents.append((sen_ids, label, sen_len, sen_masks, data[-1]))
            return shuffle(contents)
        def bert_query_process(query):
            all_query_id = {}
            for q_c in query.columns:
                # q_l = self.query[q_c].values
                q_id = []
                for q_w in query[q_c].values:
                    q_token = config.tokenizer.tokenize(q_w)
                    q_token_ids = config.tokenizer.convert_tokens_to_ids(q_token)
                    q_id.extend(q_token_ids)
                    # for word in q_token:
                    #     if word not in data_dict:
                    #         data_dict[word] = len(data_dict)
                    #     q_id.append(data_dict[word])
                all_query_id[q_c] = q_id
            return all_query_id

        train_content = bert_data_process(train_raw_data, config.pad_size, config.doc_pad_size)
        dev_content = bert_data_process(dev_raw_data, config.pad_size, config.doc_pad_size)
        test_content = bert_data_process(test_raw_data, config.pad_size, config.doc_pad_size)
        query_ids = bert_query_process(pd_query)
    # # delete
    #     pd_test001 = pd.DataFrame(test001)
    #     pd_test001.to_csv('F:\\test001.csv',encoding='utf-8',header=None, index=None)
        return train_content, dev_content, test_content, query_ids

    def load_dataset(config):
        train_raw_data = pd.read_csv(config.train_path)
        dev_raw_data = pd.read_csv(config.dev_path)
        test_raw_data = pd.read_csv(config.test_path)
        pd_query = pd.read_csv(config.query_path).dropna()

        print('train_raw_data len = {}, dev_raw_data len = {}, test_raw_data len = {}'.format(len(train_raw_data), len(dev_raw_data), len(test_raw_data)))
        # with open (config.word2vec_dict_path, 'rb') as d:
        #     id2word = pickle.load(d)
        # word2id = dict(zip(id2word.values(), id2word.keys()))
        char_word_2id = {'<PAD>': 0, '<UNK>': 1}
        # char_word_2id = word2id # use dict learned by word2vec

        def data_process(raw_data, data_dict, pad_size, doc_pad_size):
            contents = []
            for data in raw_data.itertuples(index=False):
                sen_ids = []
                lin = data[5].strip()
                if not lin:
                    continue
                content = lin.replace('\r','').replace('\n','').replace('\u3000','').replace(' ','')
                label_dict = {301:'0', 302:'1', 303:'2', 304:'3', 305:'4'}
                label = int(label_dict[data[12]])
                content_cut =  cut_sentences(content)
                for sen in content_cut:
                    # test001.append(sen)  # delete
                    token = config.tokenizer.tokenize(sen)
                    token_ids = []
                    for word in token:
                        if word not in data_dict:
                            data_dict[word] = len(data_dict)
                        token_ids.append(data_dict[word])
                    if pad_size:  # pading sentence
                        if len(token_ids) < pad_size:
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            token_ids = token_ids[:pad_size]
                    sen_ids.append(token_ids)
                if doc_pad_size:  # pading document
                    if len(sen_ids) < doc_pad_size:
                        sen_mask = [0] * pad_size
                        pad_sent = (sen_mask)
                        padding = doc_pad_size - len(sen_ids)
                        for i in range(padding):
                            sen_ids.append(pad_sent)
                    else:
                        sen_ids = sen_ids[:doc_pad_size]
                contents.append((sen_ids, label, data[-1]))
            return shuffle(contents), data_dict

        def query_process(query, data_dict):
            all_query_id = {}
            for q_c in query.columns:
                # q_l = self.query[q_c].values
                q_id = []
                for q_w in query[q_c].values:
                    q_token = config.tokenizer.tokenize(q_w)
                    for word in q_token:
                        if word not in data_dict:
                            data_dict[word] = len(data_dict)
                        q_id.append(data_dict[word])
                all_query_id[q_c] = q_id
            return all_query_id, data_dict

        query_ids, char_word_2id = query_process(pd_query, char_word_2id)
        train_content, char_word_2id = data_process(train_raw_data, char_word_2id, config.pad_size, config.doc_pad_size)
        dev_content, char_word_2id = data_process(dev_raw_data, char_word_2id, config.pad_size, config.doc_pad_size)
        test_content, char_word_2id = data_process(test_raw_data, char_word_2id, config.pad_size, config.doc_pad_size)
        # id2word = {char_word_2id[i]: i for i in char_word_2id}
        print("vocabulary size:{}, including query, train, dev and test file".format(len(char_word_2id)))
        with open(config.word2id_path, 'wb') as f:
            pickle.dump(char_word_2id, f)
        return train_content, dev_content, test_content, query_ids

    def word2vec_load_dataset(train_path, dev_path, test_path, word2vec_sen_path):
        train_raw_data = pd.read_csv(train_path)
        dev_raw_data = pd.read_csv(dev_path)
        test_raw_data = pd.read_csv(test_path)

        def word2vec_data_process(train_raw_data, dev_raw_data, test_raw_data):
            raw_list = [train_raw_data, dev_raw_data, test_raw_data]
            all_content = ['UNK']
            for raw_data in raw_list:
                for data in raw_data.itertuples(index=False):
                    lin = data[5].strip()
                    if not lin:
                        continue
                    content = lin.replace('\r', '').replace('\n', '').replace('\u3000', '').replace(' ', '')
                    content_cut = cut_sentences(content)
                    for sen in content_cut:
                        # test001.append(sen)  # delete
                        token = config.tokenizer.tokenize(sen)
                        all_content.append(token)
            return shuffle(all_content)

        all_content = word2vec_data_process(train_raw_data, dev_raw_data, test_raw_data)
        # id2word = {char_word_2id[i]: i for i in char_word_2id}
        print("all content sent len:{}, including train, dev and test file".format(len(all_content)))
        # pd_all_sen = pd.DataFrame(all_content)
        # pd_all_sen.to_csv(word2vec_all_sen_path, sep='\n', index=False)
        with open(word2vec_sen_path, 'w') as f:
            for i in all_content:
                for w in i:
                    f.write(w)
                    f.write(' ')
                f.write('\n')

    # if config.train_word2vec:
    #     word2vec_load_dataset(config.train_path, config.dev_path, config.test_path, config.word2vec_sen_path)
    # else:
    if config.is_bert:
        # dev = bert_load_dataset(config.dev_path, config.pad_size, config.doc_pad_size, config.num_classes)
        # train = bert_load_dataset(config.train_path, config.pad_size, config.doc_pad_size, config.num_classes)
        # test = bert_load_dataset(config.test_path, config.pad_size, config.doc_pad_size, config.num_classes)
        train, dev, test, query_ids = bert_load_dataset(config)
        return train, dev, test, query_ids
    else:
        train, dev, test, query_ids = load_dataset(config)
        return train, dev, test, query_ids

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):

        # for c_data in datas:
        x = torch.LongTensor([[sen[0] for sen in _[0]] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([[sen[1] for sen in _[0]] for _ in datas]).to(self.device)
        mask = torch.LongTensor([[sen[2] for sen in _[0]] for _ in datas]).to(self.device)

        # # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

class NoPreDatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):

        # for c_data in datas:
        x = torch.LongTensor([[sen for sen in _[0]] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
        
        
class BertReviewDataset(Dataset):
    def __init__(self, data, device):
        super(BertReviewDataset, self).__init__()
        self.data = data
        self.device = device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        batch_data = self.data[index]
        # batch_x = torch.LongTensor([[sen for sen in _[0]] for _ in batch_data]).to(self.device)
        batch_x = torch.LongTensor(batch_data[0]).to(self.device)
        batch_x_mask = torch.LongTensor(batch_data[3]).to(self.device)
        # batch_y = torch.LongTensor([_[1] for _ in batch_data]).to(self.device) 
        y_np = np.array(batch_data[1])
        batch_y = torch.from_numpy(y_np).type(torch.LongTensor).to(self.device)
        # batch_y = torch.LongTensor(y_np).to(self.device)
        X = [batch_x, batch_y, batch_x_mask, batch_data[4]]
        return X

class ReviewDataset(Dataset):
    def __init__(self, data, device):
        super(ReviewDataset, self).__init__()
        self.data = data
        self.device = device

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        batch_data = self.data[index]
        # batch_x = torch.LongTensor([[sen for sen in _[0]] for _ in batch_data]).to(self.device)
        batch_x = torch.LongTensor(batch_data[0]).to(self.device)
        # batch_y = torch.LongTensor([_[1] for _ in batch_data]).to(self.device) 
        y_np = np.array(batch_data[1])
        batch_y = torch.from_numpy(y_np).type(torch.LongTensor).to(self.device)
        # batch_y = torch.LongTensor(y_np).to(self.device)
        X = [batch_x, batch_y, batch_data[2]]
        return X

def get_data_loader(dataset, batch_size, num_workers):
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #,worker_init_fn=worker_init_fn: waiting for set

def build_dataloader(data, device, batch_size, num_workers, is_bert):
    if is_bert:
        dataset = BertReviewDataset(data, device)
    else:
        dataset = ReviewDataset(data, device)
    dataloader = get_data_loader(dataset, batch_size, num_workers)
    return dataloader

def build_iterator(dataset, is_bert, batch_size, device):
    if is_bert:
        iter = DatasetIterater(dataset, batch_size, device)
    else:
        iter = NoPreDatasetIterater(dataset, batch_size, device)
    return iter




def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def matrix_mul(input, weight, bias=False):
    # weight_matrix = torch.repeat_interleave(weight.unsqueeze(dim=0), repeats=input.shape[0], dim=0) 
    weight_matrix = weight.expand(input.shape[0], -1,-1) 
    feature = torch.matmul(input, weight_matrix)
    if isinstance(bias, torch.nn.parameter.Parameter):
        feature = feature + bias.expand(feature.size()[0], feature.size()[1], -1)
    feature = torch.tanh(feature)
    return feature



def element_wise_mul(input1, input2):
    feature_2 = input2.unsqueeze(2).expand_as(input1)
    output = input1 * feature_2
    return torch.sum(output, 0)

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)
    
class Common_Config(object):
    def __init__(self, args):
        self.train_path = 'data/' + args.dataset + '/data/train.csv'                                # 训练集
        self.dev_path = 'data/' + args.dataset + '/data/dev.csv'                                    # 验证集
        self.test_path = 'data/' + args.dataset + '/data/test.csv'                                  # 测试集
        # processed data path
        self.processed_train_path = 'result/processed_data/processed_data_nopre/train.csv'                                # 训练集
        self.processed_dev_path = 'result/processed_data/processed_data_nopre/dev.csv'                                    # 验证集
        self.processed_test_path = 'result/processed_data/processed_data_nopre/test.csv'
        self.word2id_path = 'result/processed_data/processed_data_nopre/word2id.pkl'
        # class label list
        with open('data/' + args.dataset + '/data/class.txt') as c_l:
            self.class_list = [x.strip() for x in c_l.readlines()]
        self.num_classes = len(self.class_list)                         # 类别数
        self.device = args.device  # 设备

