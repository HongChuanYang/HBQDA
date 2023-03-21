# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from models.sent_att_model import SentAttNet



class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = 'data/' + dataset + '/data/train.csv'                                # 训练集
        self.dev_path = 'data/' + dataset + '/data/dev.csv'                                    # 验证集
        self.test_path = 'data/' + dataset + '/data/test.csv'                                  # 测试集
        # processed data path
        self.processed_train_path = 'data/' + dataset + '/processed_data/train.csv'                                # 训练集
        self.processed_dev_path = 'data/' + dataset + '/processed_data/dev.csv'                                    # 验证集
        self.processed_test_path = 'data/' + dataset + '/processed_data/test.csv'  
        self.class_list = [x.strip() for x in open(
            'data/' + dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = 'data/' + dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 8                                          # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.doc_pad_size = 24                                          # 每篇文章处理成的句子长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        # self.bert_path = './bert_pretrain'
        self.bert_path = 'pretrain/bert_pretrain' 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2
        self.sent_hidden_size = 50

        # self.max_length_sentences = 32

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        

        self.gru = nn.GRU(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.fc = nn.Linear(config.rnn_hidden, 1)
        self.document_fc = nn.Linear(config.doc_pad_size, config.num_classes)
        self.sent_att_net = SentAttNet(config.sent_hidden_size, config.hidden_size, config.num_classes)
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.sent_hidden_size = config.sent_hidden_size

        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        # self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            # self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, x):
        output_list = []
        contexts = x[0].permute(1, 0, 2)
        masks = x[2].permute(1, 0, 2)
        for i in range(len(contexts)):    
            #encoder_out (batch,seq_len,hidden_size) 最后一层 各个时刻/token对应的编码向量
            #text_cls (batch,hidden_size) 最后一层 cls token对应的编码向量,即句向量
            encoder_out, text_cls = self.bert(contexts[i], attention_mask=masks[i], output_all_encoded_layers=False)
            # sen_out = self.fc(text_cls)
            # sen_out = self.dropout(sen_out)
            output_list.append(text_cls)
        output = torch.stack(output_list)
        # outputs = output.squeeze(2)
        # outputsp = outputs.permute(1,0) 
        # out = self.document_fc(outputsp)
        outputp = output.permute(1,0,2) 
        out, _ = self.gru(outputp)
        out = self.dropout(out)
        test1= out[:, -1, :] # 看看什么样
        out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
