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
        self.model_name = 'bert_att'
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
class SentAttNet_bertatt(nn.Module):
    def __init__(self, config):
        super(SentAttNet_bertatt, self).__init__()
        
        self.hidden_size = config.sent_hidden_size
        self.gru = nn.GRU(config.hidden_size, config.sent_hidden_size, bidirectional=True, batch_first=True)
        self.att_net = nn.Sequential(
            nn.Linear(config.sent_hidden_size * 2, config.sent_hidden_size * 2, bias=True),
            nn.Tanh(),
            nn.Linear(config.sent_hidden_size * 2, 1)
        )

    def forward(self, input):
        f_output, h_output = self.gru(input)
        Uw_dot_Uit = self.att_net(f_output)  # sentences_vector_fragment: batch,hidden
        Uw_dot_Uit = torch.squeeze(Uw_dot_Uit, dim=2)
        scores = F.softmax(Uw_dot_Uit, dim=1)
        scores = torch.unsqueeze(scores, dim=2)
        doc_vector = torch.sum(scores * f_output, dim=1)  # dim= 1基于senquence   ,si:batch, hidden
        return doc_vector, h_output# dim= 1基于senquence   ,si:batch, hidden
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True # fine tune or not
        

        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.fc = nn.Linear(config.rnn_hidden, 1)
        self.document_fc = nn.Linear(config.sent_hidden_size*2, config.num_classes)
        # self.sent_att_net = SentAttNet(config.sent_hidden_size, config.hidden_size, config.num_classes)
        self.sent_att_net = SentAttNet_bertatt(config)
        self.batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.sent_hidden_size = config.sent_hidden_size



    def forward(self, x, mask, query_ids):
        output_list = []
        contexts = x.permute(1, 0, 2)
        contexts_masks = mask.permute(1, 0, 2)
        for i in range(len(contexts)):
            encoder_out, text_cls = self.bert(contexts[i], attention_mask=contexts_masks[i], output_all_encoded_layers=False)
            # sen_out = self.fc(text_cls)
            output_list.append(text_cls)
        output = torch.stack(output_list)
        # output = output.squeeze(2)

        # output = output.permute(1,0,2) 

        out, self.sent_hidden_state = self.sent_att_net(output.permute(1,0,2))
        # self.sent_hidden_state = self.sent_hidden_state.data

        output = self.document_fc(out)


        return output
