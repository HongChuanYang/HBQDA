import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.sent_att_model_no_pre import SentAttNet_no_pre
# from models.word_att_model_no_pre import WordAttNet_no_pre
from utils import matrix_mul, element_wise_mul, Common_Config
from pytorch_pretrained import BertModel, BertTokenizer
import pickle
import pandas as pd


class Config(Common_Config):

    """配置参数"""
    def __init__(self, args):
        super(Config, self).__init__(args)

        self.model_name = 'hans'

        # query path 
        self.query_path = 'data/query/query.csv'
        self.processed_query_path = 'result/processed_data/processed_data_nopre/query_id.csv'

        # save path
        self.save_path = 'result/model_saved_dict/'  +self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = 'result/test_result/' + self.model_name + '_log.txt'        # 模型训练结果

        
        # self.device = args.device  # 设备
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
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
class WordAttNet_no_pre(nn.Module):
    def __init__(self, config, char_word_2id):
        super(WordAttNet_no_pre, self).__init__()

        self.device = config.device
        self.word_emb_size = config.word_emb_size
        self.hidden_size = config.word_hidden_size
        self.char_word_2id = char_word_2id
        self.emb = nn.Embedding(len(self.char_word_2id), self.word_emb_size)
        # self.emb = nn.Embedding.from_pretrained(word_embeddings_weight)
        #
        # self.emb = apply_emb_weight(self.emb, word_embeddings_weight, device)

        self.gru = nn.GRU(self.word_emb_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.att_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, 1, bias=False)
        )
        

    def forward(self, input):

        output = self.emb(input)
        output = output.reshape(input.shape[0]*input.shape[1], input.shape[2], output.shape[3])
        f_output, h_output = self.gru(output.float())  # feature output and hidden state output
        f_output = f_output.reshape(input.shape[0], input.shape[1], input.shape[2], -1) #output:batch,sentences,words,embed

        # 分句子计算词级别注意力机制
        s_stack = []
        for index in range(input.shape[1]):
            fragment = f_output[:, index, :, :] # fragment：batch,seq,hidden
            Uw_dot_Uit = self.att_net(fragment) #sentences_vector_fragment: batch,hidden
            Uw_dot_Uit = torch.squeeze(Uw_dot_Uit, dim=2)
            scores = F.softmax(Uw_dot_Uit, dim=1)
            scores = torch.unsqueeze(scores, dim=2)  # scores: batch,seq,1
            sentences_vector_fragment = torch.sum(scores * fragment, dim=1)  # dim= 1基于senquence   ,si:batch, hidden
            s_stack.append(sentences_vector_fragment)  #需要按照列进行堆叠

        sentences_vector = torch.stack(s_stack, dim=1) #sentences_vector:batch,sentences,hidden

        return sentences_vector, h_output
class SentAttNet_no_pre(nn.Module):
    def __init__(self, config):
        super(SentAttNet_no_pre, self).__init__()
        self.hidden_size = config.sent_hidden_size
        self.gru = nn.GRU(2 *config.word_hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.att_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, 1)
        )


    def forward(self, input):

        f_output, h_output = self.gru(input)
        Uw_dot_Uit = self.att_net(f_output)  # sentences_vector_fragment: batch,hidden
        Uw_dot_Uit = torch.squeeze(Uw_dot_Uit, dim=2)
        scores = F.softmax(Uw_dot_Uit, dim=1)
        scores = torch.unsqueeze(scores, dim=2)
        doc_vector = torch.sum(scores * f_output, dim=1)  # dim= 1基于senquence   ,si:batch, hidden
        return doc_vector, h_output

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        with open(config.word2id_path, 'rb') as f:
            self.char_word_2id = pickle.load(f)
        # self.word_embeddings_weight = pickle.load(open(config.word2vec_emb_path, 'rb'))

        self.word_att_net_no_pre = WordAttNet_no_pre(config, self.char_word_2id)
        self.sent_att_net_no_pre = SentAttNet_no_pre(config)
        self.sent_fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.sent_hidden_size * 2, config.num_classes, bias=True),
            nn.Softmax(dim = 1)
        )


    def forward(self, x):
        output, word_hidden_state = self.word_att_net_no_pre(x)
        output, sent_hidden_state = self.sent_att_net_no_pre(output)
        p_scores = self.sent_fc(output)
        return p_scores