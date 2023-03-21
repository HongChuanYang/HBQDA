# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from models.sent_att_model import SentAttNet
from models.sent_att_model_no_pre import SentAttNet_no_pre
from utils import Common_Config



class Config(Common_Config):

    """配置参数"""
    def __init__(self, args):
        super(Config, self).__init__(args)
        self.model_name = 'hbqa'
          # query path 
        self.query_path = 'data/query/query.csv'
        self.processed_query_path = 'result/processed_data/processed_data_bert/query_id.csv'
        self.processed_train_path = 'result/processed_data/processed_data_bert/train.csv'                                
        self.processed_dev_path = 'result/processed_data/processed_data_bert/dev.csv'                                    
        self.processed_test_path = 'result/processed_data/processed_data_bert/test.csv'

        self.save_path = 'result/model_saved_dict/'  +self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = 'result/test_result/' + self.model_name + '_log.txt'    
              # 模型训练结果
        self.device = args.device  # 设备

        self.is_bert = True  # 是否为bert
        self.bert_path = 'pretrain/bert_pretrain' 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                         # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.doc_pad_size = 24  
        # self.learning_rate = 0.003                                    # 学习率
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2
        self.word_emb_size = 200
        self.word_hidden_size = 100
        self.sent_hidden_size = 100
        self.train_word2vec = False

        # self.max_length_sentences = 32
class SentAttNet_bert_fishqa(nn.Module):
    def __init__(self, config):
        super(SentAttNet_bert_fishqa, self).__init__()
        
        # self.hidden_size = config.sent_hidden_size
        self.hidden_size = int(config.hidden_size/2)
        self.gru = nn.GRU(config.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)

        self.att_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, 1, bias=False)
        )
        
        self.att_net_1 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
        )
        self.att_net_2 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
        )
        self.att_net_3 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
        )
        self.att_net_4 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
        )
        self.att_net_5 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
        )

    def forward(self, input, query_emb):
        f_output, h_output = self.gru(input)
        alpha_t = F.softmax(torch.squeeze(self.att_net(f_output), dim=2), dim=1)
        alpha_q1 = F.softmax(torch.sum(self.att_net_1(f_output).mul(query_emb[0]), dim=2), dim=1)
        alpha_q2 = F.softmax(torch.sum(self.att_net_2(f_output).mul(query_emb[1]), dim=2), dim=1)
        alpha_q3 = F.softmax(torch.sum(self.att_net_3(f_output).mul(query_emb[2]), dim=2), dim=1)
        alpha_q4 = F.softmax(torch.sum(self.att_net_4(f_output).mul(query_emb[3]), dim=2), dim=1)
        alpha_q5 = F.softmax(torch.sum(self.att_net_5(f_output).mul(query_emb[4]), dim=2), dim=1)
        alpha_mean = (alpha_t+alpha_q1+alpha_q2+alpha_q3+alpha_q4+alpha_q5)/6
        scores = torch.unsqueeze(alpha_mean, dim=2)
        doc_vector = torch.sum(scores * f_output, dim=1)  #
# dim= 1基于senquence   ,si:batch, hidden
        return doc_vector, h_output# dim= 1基于senquence   ,si:batch, hidden
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False # fine tune or not
        

        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.fc = nn.Linear(config.rnn_hidden, 1)
        self.document_fc = nn.Linear(config.hidden_size, config.num_classes)
        # self.sent_att_net = SentAttNet(config.sent_hidden_size, config.hidden_size, config.num_classes)
        self.sent_att_net = SentAttNet_bert_fishqa(config)
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.sent_hidden_size = config.sent_hidden_size

    #     self._init_hidden_state()

    # def _init_hidden_state(self, last_batch_size=None):
    #     if last_batch_size:
    #         batch_size = last_batch_size
    #     else:
    #         batch_size = self.batch_size
    #     # self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
    #     self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
    #     if torch.cuda.is_available():
    #         # self.word_hidden_state = self.word_hidden_state.cuda()
    #         self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, x, mask, query_ids):
        output_list = []
        contexts = x.permute(1, 0, 2)
        contexts_masks = mask.permute(1, 0, 2)
        for i in range(len(contexts)):
            encoder_out, text_cls = self.bert(contexts[i], attention_mask=contexts_masks[i], output_all_encoded_layers=False)
            # sen_out = self.fc(text_cls)
            output_list.append(text_cls)
        query_emb= []
        for q_i in query_ids.values():
            q_i_us = torch.unsqueeze(q_i, dim=0)
            q_i_out, q_i_emb = self.bert(q_i_us)
            query_emb.append(q_i_emb.squeeze())

        output = torch.stack(output_list)
        # output = output.squeeze(2)

        # output = output.permute(1,0,2) 

        out, self.sent_hidden_state = self.sent_att_net(output.permute(1,0,2), query_emb)
        # self.sent_hidden_state = self.sent_hidden_state.data

        output = self.document_fc(out)

        # out, _ = self.lstm(output)
        # out = self.dropout(out)
        # out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return output
