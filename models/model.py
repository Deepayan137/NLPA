import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import  torch.nn.functional as F
import pdb
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
    def forward(self, inputs):
        
        embedding = self.embeddings(inputs)
        avg_embedding = torch.mean(embedding, dim=1)

        out = self.linear(torch.squeeze(avg_embedding))
        log_probs = F.log_softmax(out)
        return log_probs