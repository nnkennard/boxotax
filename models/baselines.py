import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
#import matplotlib.pyplot as plt


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.01

class Bilinear(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(Bilinear, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1, False)
    self.tanh = nn.Tanh()

  def forward(self, x_1, x_2):
    emb_1 = self.embedding(x_1)
    emb_2 = self.embedding(x_2)
    mul = self.bilinear(emb_1, emb_2)
    return self.tanh(mul)

class ComplEx(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(ComplEx, self).__init__()
    self.embedding_re = nn.Embedding(vocab_size, embedding_dim)
    self.embedding_im = nn.Embedding(vocab_size, embedding_dim)
    self.relation_re = nn.Parameter(torch.FloatTensor(embedding_dim,
      embedding_dim))
    self.relation_im  = nn.Parameter(torch.FloatTensor(embedding_dim,
      embedding_dim))

    # Initialization from TypeNet code
    stdv = 1.0 / math.sqrt(embedding_dim)
    self.relation_re.data.uniform_(-stdv, stdv)
    self.relation_im.data.uniform_(-stdv, stdv)

    self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1, False)
    self.tanh = nn.Tanh()

  def forward(self, x_1, x_2):
    emb_1_re = self.embedding_re(x_1).unsqueeze(0)
    emb_2_re = self.embedding_re(x_2).unsqueeze(0)
    emb_1_im = self.embedding_im(x_1).unsqueeze(1)
    emb_2_im = self.embedding_im(x_2).unsqueeze(1)

    imreim = torch.mm(emb_1_im,
    (emb_2_im * self.relation_im).unsqueeze(-1)).squeeze(-1)

    rerere = torch.mm(emb_1_re,
        torch.transpose(emb_2_re * self.relation_re.unsqueeze(1), 0, 1)).squeeze()

    imimre = torch.mm(emb_1_im,
        torch.transpose(emb_2_im * self.relation_re.unsqueeze(1), 0, 1)).squeeze()

    reimim = torch.mm(emb_1_re,
        torch.transpose(emb_2_im * self.relation_im.unsqueeze(1), 0, 1)).squeeze()

    imreim = torch.mm(emb_1_im,
        torch.transpose(emb_2_re * self.relation_im.unsqueeze(1), 0, 1)).squeeze()

    score = self.tanh(rerere + imimre + reimim - imreim)

    return imreim

    unused_nothing = """
        rerere = torch.matmul(input_a_embedding_re, input_b_embedding_re * input_r_embedding_re).squeeze()
        reimim = torch.matmul(input_a_embedding_im, input_b_embedding_im * input_r_embedding_re).squeeze()
        imreim = torch.matmul(input_a_embedding_re, input_b_embedding_im * input_r_embedding_im).squeeze()
        imimre = torch.matmul(input_a_embedding_im, input_b_embedding_re * input_r_embedding_im).squeeze()

        score = self.sigmoid(rerere + reimim + imreim - imimre)#(batch_size,)
        return score
"""
    return self.relu(mul)


def main():
  model = ComplEx(23, 17)
  #model.weight.data.normal_(0.0, 0.02)

  x_train = np.array([
    [0,1],
    [0,2],
    [0,3],
    [0,4],
    [0,5],
    [1,3],
    [1,4],

    [2,1],
    [2,3],
    [3,2],
    [2,4],
    [3,0],
    [4,0],
  ])

  x_0 = torch.from_numpy(np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 2, 3, 4]))
  x_1 = torch.from_numpy(np.array([1, 2, 3, 4, 5, 3, 4, 1, 3, 2, 4, 0, 0]))

  y_train = np.array([1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1], dtype=np.float32)

  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    outputs = model(x_0, x_1)
    loss = criterion(outputs.view(13), targets)
    print(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  predictions = model(x_0, x_1)
  print(predictions)

if __name__ == "__main__":
  main()
