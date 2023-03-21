# coding: UTF-8
# from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from models.sent_att_model import SentAttNet
from utils import Common_Config




class Config(Common_Config):

    """配置参数"""
    def __init__(self, args):
        super(Config, self).__init__(args)
        self.model_name = 'bert'
          # query path 
        self.query_path = 'data/query/query.csv'
        self.processed_query_path = 'result/processed_data/processed_data_bert/query_id.csv'
        self.processed_train_path = 'result/processed_data/processed_data_bert/train.csv'                                
        self.processed_dev_path = 'result/processed_data/processed_data_bert/dev.csv'                                    
        self.processed_test_path = 'result/processed_data/processed_data_bert/test.csv'
                              # 测试集
        # save path
        self.save_path = 'result/model_saved_dict/'  +self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = 'result/test_result/' + self.model_name + '_log.txt' 
        
        # self.device = args.device  # 设备

        self.is_bert = True  # 是否为bert
        self.bert_path = 'pretrain/bert_pretrain' 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)


        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                        # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.doc_pad_size = 24           
                                       # 每篇文章处理成的句子长度(短填长切)
        # self.bert_path = './bert_pretrain'
        self.bert_hidden_size = 768
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2
        self.sent_hidden_size = 100

        # self.max_length_sentences = 32

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False # fine tune or not
        

        # self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
        #                     bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.sent_fc = nn.Linear(config.doc_pad_size, 1)
        self.document_fc = nn.Linear(config.bert_hidden_size, config.num_classes)
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

    def forward(self, x, mask):
        output_list = []
        contexts = x.permute(1, 0, 2)
        contexts_masks = mask.permute(1, 0, 2)
        for i in range(len(contexts)):
            encoder_out, text_cls = self.bert(contexts[i], attention_mask=contexts_masks[i], output_all_encoded_layers=False)
            # sen_out = self.fc(text_cls)
            # output_list.append(sen_out)
            output_list.append(text_cls)
        output = torch.stack(output_list).permute(1,2,0)

        outputs = self.sent_fc(output)
        outputsq = outputs.squeeze()

        # output = output.permute(1,0,2) 

        # out, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)

        out = self.document_fc(outputsq)
        #如何调整为全连接层

        # out, _ = self.lstm(output)
        # out = self.dropout(out)
        # out = self.fc_rnn(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
