"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul


class WordAttNet(nn.Module):
    def __init__(self, word_emb_size=200,hidden_size=50):
        super(WordAttNet, self).__init__()
        
        self.emb = nn.Embedding(21128, word_emb_size)  #21128 为BertTokenizer中vocab中的的ids数量。
        # if embedding_matrix is not None:
        #     weight = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
        #     self.emb = nn.Embedding.from_pretrained(weight)
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.gru = nn.GRU(word_emb_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.emb(input)
        output = output.reshape(input.shape[0],input.shape[1]*input.shape[2],output.shape[3])
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        # output = matrix_mul(output, self.context_weight).permute(1,0)
        output = matrix_mul(output, self.context_weight).squeeze(2).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output.reshape(input.shape[1],input.shape[2],output.shape[1]), h_output


# if __name__ == "__main__":
    # abc = WordAttNet("../data/glove.6B.50d.txt")
