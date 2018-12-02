import numpy as np
import sys
import torch
import torch.nn as nn

from absl import flags
from IPython.core.debugger import set_trace
from torch.utils.data import Dataset, DataLoader

import box_lib

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 10, 'number of epochs (total)')
flags.DEFINE_integer('random_seed', 43, 'value for the random seed')
flags.DEFINE_string('train_path', None,
    'path to train conditional probabilities')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('embedding_size', 20,
    'total size of embedding (including max and min)')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('l2_lambda', 0.0001, 'lambda for l2')
flags.DEFINE_string('device', 'cpu', 'device for torch option')


def set_random_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)


def main():
  set_random_seed(FLAGS.random_seed)

  train_ds = box_lib.BoxDataset(FLAGS.train_path)
  train_dl = DataLoader(train_ds, batch_size=FLAGS.batch_size,
      shuffle=True, num_workers=4)

  model = box_lib.Boxes(train_ds.vocab_size, FLAGS.embedding_size)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

  for epoch in range(FLAGS.num_epochs):
    print(epoch)

    model.train()

    running_loss, correct = 0.0, 0
    for X, y in train_dl:
      X, y = X.to(FLAGS.device), y.to(FLAGS.device)

      #TODO: Use more canonical regularization maybe
      with torch.set_grad_enabled(True):
        y_, norms = model(X)
        loss = criterion(y_, y) + FLAGS.l2_lambda * norms

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Statistics
      print("    batch loss: "+str(loss.item()))
      running_loss += loss.item() * X.shape[0]

    print("  Train Loss: "+str(running_loss / len(train_dl.dataset)))


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
