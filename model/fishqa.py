import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul, Common_Config
from pytorch_pretrained import BertModel, BertTokenizer
import pickle
import pandas as pd


class Config(Common_Config):

    """配置参数"""
    def __init__(self, args):
        super(Config, self).__init__(args)
        self.model_name = 'fishqa'

        # query path 
        self.query_path = 'data/query/query.csv'
        self.processed_query_path = 'result/processed_data/processed_data_nopre/query_id.csv'

        # save path
        self.save_path = 'result/model_saved_dict/'  +self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = 'result/test_result/' + self.model_name + '_log.txt' 
        
        self.is_bert = False  # 是否为bert
        self.bert_path = 'pretrain/bert_pretrain' 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.require_improvement = 10000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                         # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.doc_pad_size = 24                                          # 每篇文章处理成的句子长度(短填长切)
        # self.learning_rate = 0.003                                    # 学习率
        self.hidden_size = 100
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.5
        self.rnn_hidden = 768
        self.num_layers = 2
        self.word_emb_size = 200
        self.word_hidden_size = 100
        self.sent_hidden_size = 100


        # self.max_length_sentences = 32
        
        
        
class WordAttNet_fishqa(nn.Module):
    
    def __init__(self, config, char_word_2id):
        super(WordAttNet_fishqa, self).__init__()

        self.device = config.device
        self.word_emb_size = config.word_emb_size
        self.hidden_size = config.word_hidden_size
        self.char_word_2id = char_word_2id
        self.emb = nn.Embedding(len(self.char_word_2id), self.word_emb_size)
        self.gru = nn.GRU(self.word_emb_size, self.hidden_size, bidirectional=True, batch_first=True)

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
        
    def forward(self, input, query_ids):

        output = self.emb(input) 
        output = output.reshape(input.shape[0]*input.shape[1], input.shape[2], output.shape[3])
        f_output, h_output = self.gru(output.float())  # feature output and hidden state output
        f_output = f_output.reshape(input.shape[0], input.shape[1], input.shape[2], -1) #output:batch,sentences,words,embed
        query_emb = [self.emb(q_i) for q_i in query_ids.values()]
        # sum query emb for the representation of query
        for i in range(len(query_emb)):
            query_emb[i] = torch.sum(query_emb[i], dim=0)/len(query_emb[i])

        # 分句子计算词级别注意力机制
        s_stack = []
        for index in range(input.shape[1]):
            fragment = f_output[:, index, :, :] # fragment：batch,seq,hidden
            alpha_t = F.softmax(torch.squeeze(self.att_net(fragment), dim=2), dim=1)
            alpha_q1 = F.softmax(torch.sum(self.att_net_1(fragment).mul(query_emb[0]), dim=2), dim=1)
            alpha_q2 = F.softmax(torch.sum(self.att_net_2(fragment).mul(query_emb[1]), dim=2), dim=1)
            alpha_q3 = F.softmax(torch.sum(self.att_net_3(fragment).mul(query_emb[2]), dim=2), dim=1)
            alpha_q4 = F.softmax(torch.sum(self.att_net_4(fragment).mul(query_emb[3]), dim=2), dim=1)
            alpha_q5 = F.softmax(torch.sum(self.att_net_5(fragment).mul(query_emb[4]), dim=2), dim=1)
            alpha_mean = (alpha_t+alpha_q1+alpha_q2+alpha_q3+alpha_q4+alpha_q5)/6
            scores = torch.unsqueeze(alpha_mean, dim=2)  # scores: batch,seq,1
            sentences_vector_fragment = torch.sum(scores * fragment, dim=1)  # dim= 1基于senquence   ,si:batch, hidden
            s_stack.append(sentences_vector_fragment)  #需要按照列进行堆叠
        sentences_vector = torch.stack(s_stack, dim=1) #sentences_vector:batch,sentences,hidden
        return sentences_vector, h_output, query_emb

class SentAttNet_fishqa(nn.Module):
    def __init__(self, config):
        super(SentAttNet_fishqa, self).__init__()
        
        self.hidden_size = config.sent_hidden_size
        self.gru = nn.GRU(2 *config.word_hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.att_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, 1)
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
        doc_vector = torch.sum(scores * f_output, dim=1)  
        q1_value = torch.sum(alpha_q1, dim=1).unsqueeze(1)
        q2_value = torch.sum(alpha_q2, dim=1).unsqueeze(1)
        q3_value = torch.sum(alpha_q3, dim=1).unsqueeze(1)
        q4_value = torch.sum(alpha_q4, dim=1).unsqueeze(1)
        q5_value = torch.sum(alpha_q5, dim=1).unsqueeze(1)
        all_q_value = torch.cat([q1_value, q2_value, q3_value, q4_value, q5_value], 1)# dim= 1基于senquence   ,si:batch, hidden
        return doc_vector, h_output, all_q_value

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        with open(config.word2id_path, 'rb') as f:
            self.char_word_2id = pickle.load(f)

        self.word_att_net = WordAttNet_fishqa(config, self.char_word_2id)
        self.sent_att_net = SentAttNet_fishqa(config)
        self.sent_fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.sent_hidden_size * 2, config.num_classes, bias=True),
            nn.Softmax(dim = 1)
        )
        # self.batch_size = config.batch_size
        # self.num_classes = config.num_classes
        # self.sent_hidden_size = config.sent_hidden_size
        # self.word_hidden_size = config.word_hidden_size
        # self.doc_pad_size = config.doc_pad_size

    def forward(self, x, query_ids):
        output, word_hidden_state, query_emb = self.word_att_net(x, query_ids)
        output, sent_hidden_state, all_q_value= self.sent_att_net(output, query_emb)
        p_scores = self.sent_fc(output)
        return p_scores, all_q_value