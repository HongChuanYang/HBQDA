"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from utils import matrix_mul, element_wise_mul

def apply_emb_weight(emb, weights, device):
    if type(emb) is not nn.Embedding:
        return nn.init.xavier_uniform_(emb)
    if weights is not None:
        weight = torch.from_numpy(weights).type(torch.FloatTensor).to(device)
        emb = nn.Embedding.from_pretrained(weight)
        emb.weight.requires_grad = True
        emb.padding_idx = 0
    else:
        nn.init.xavier_uniform_(emb.weight.data)
        emb.weight.data[0] = torch.zeros_like(emb.weight.data[0])
    return emb


class WordAttNet_no_pre(nn.Module):
    def __init__(self, device, word_emb_size=200, hidden_size=50, char_word_2id_len = 3000):
        super(WordAttNet_no_pre, self).__init__()

        self.emb = nn.Embedding(char_word_2id_len, word_emb_size)  #21128 为BertTokenizer中vocab中的的ids数量。
        # self.emb = nn.Embedding.from_pretrained(word_embeddings_weight)
        #
        # self.emb = apply_emb_weight(self.emb, word_embeddings_weight, device)

        self.gru = nn.GRU(word_emb_size, hidden_size, bidirectional=True, batch_first=True)
        self.att_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
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


# if __name__ == "__main__":
    # abc = WordAttNet("../data/glove.6B.50d.txt")
