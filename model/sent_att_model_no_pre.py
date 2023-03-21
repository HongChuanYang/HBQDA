"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul

class SentAttNet_no_pre(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=2):
        super(SentAttNet_no_pre, self).__init__()

        self.gru = nn.GRU(2 *word_hidden_size, sent_hidden_size, bidirectional=True, batch_first=True)
        self.att_net = nn.Sequential(
            nn.Linear(sent_hidden_size * 2, sent_hidden_size, bias=True),
            nn.Tanh(),
            nn.Linear(sent_hidden_size, 1)
        )


    def forward(self, input, hidden_state):

        f_output, h_output = self.gru(input, hidden_state)
        Uw_dot_Uit = self.att_net(f_output)  # sentences_vector_fragment: batch,hidden
        Uw_dot_Uit = torch.squeeze(Uw_dot_Uit, dim=2)
        scores = F.softmax(Uw_dot_Uit, dim=1)
        scores = torch.unsqueeze(scores, dim=2)
        doc_vector = torch.sum(scores * f_output, dim=1)  # dim= 1基于senquence   ,si:batch, hidden
        return doc_vector, h_output


# if __name__ == "__main__":
#     abc = SentAttNet()
