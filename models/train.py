import numpy as np
import sys
import torch
import torch.nn as nn

from absl import flags
from torch.utils.data import Dataset, DataLoader

import box_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None,
    'config identifier')
flags.DEFINE_integer('num_epochs', 10, 'number of epochs (max)')
flags.DEFINE_integer('random_seed', 43, 'value for the random seed')
flags.DEFINE_string('train_path', None,
    'path to train conditional probabilities')
flags.DEFINE_string('save_path', None, 'directory where models are saved')
flags.DEFINE_integer('save_freq', 5, 'epochs between model saves')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('embedding_size', 20,
    'total size of embedding (including max and min)')
flags.DEFINE_integer('patience', 5, ('number of non-improving dev '
'iterations after which to early-stop'))
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('l2_lambda', 0.001, 'lambda for l2')
flags.DEFINE_string('device', 'cpu', 'device for torch option')
flags.DEFINE_boolean('verbose', False, 'Whether to print batch loss')

# For some reason, pytorch crashes with more workers
NUM_WORKERS = 0

def setup():
  box_lib.set_random_seed(FLAGS.random_seed)


def get_data():
  train_ds = box_lib.BoxDataset(FLAGS.train_path)
  train_dl = DataLoader(train_ds, batch_size=FLAGS.batch_size,
      shuffle=True, num_workers=NUM_WORKERS)
  dev_ds = box_lib.BoxDataset(
      FLAGS.train_path.replace("train", "dev"))
  dev_dl = DataLoader(dev_ds, batch_size=FLAGS.batch_size,
      shuffle=True, num_workers=NUM_WORKERS)
  return train_ds, train_dl, dev_ds, dev_dl


def get_and_maybe_load_model(train_ds):
  model = box_lib.Boxes(train_ds.vocab_size, FLAGS.embedding_size)
  model.to(FLAGS.device)
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
  return model, criterion, optimizer


def save_current_model(model, epoch, is_best=False):
  if is_best:
    config_number = "best"
  else:
    config_number = str(epoch).zfill(4)

  path = "".join([FLAGS.save_path, "/", FLAGS.config, "_", config_number,
      ".params"])

  torch.save(model.state_dict(), path)


def run_train_iters(model, criterion, optimizer,
    train_dl, dev_dl, train_ds, dev_ds):

  best_dev_loss = float('inf')
  best_dev_loss_epoch = -1
  dev_labels = dev_ds.y_train.to(FLAGS.device)
  train_labels = train_ds.y_train.to(FLAGS.device)

  for epoch in range(FLAGS.num_epochs):
    print(epoch)

    running_loss = run_train_iter(model, criterion, optimizer, train_dl)

    dev_loss = criterion(model(dev_ds.X_train)[0], dev_labels).item()
    print("try check loss")
    print(dev_loss, best_dev_loss)
    if dev_loss < best_dev_loss:
      print("".join(["Saving best model. Epoch: ", str(epoch), "\tLoss: ",
        str(dev_loss)]))
      print("Previous best dev loss", best_dev_loss)
      print("Current dev loss", dev_loss)
      best_dev_loss = dev_loss
      best_dev_loss_epoch = epoch
      save_current_model(model, epoch, is_best=True)

    if epoch % 1  == 0:
      box_lib.confusion(dev_ds, model)

    if epoch % FLAGS.save_freq == 0:
      save_current_model(model, epoch)

    if epoch >= best_dev_loss_epoch + FLAGS.patience:
      break

    print("  Train Loss: "+str(running_loss / len(train_dl.dataset)))


def run_train_iter(model, criterion, optimizer, train_dl):
  model.train()
  running_loss = 0.0

  for X, y in train_dl:
    torch.cuda.empty_cache()
    X, y = X.to(FLAGS.device), y.to(FLAGS.device)

    #TODO: Use more canonical regularization maybe
    with torch.set_grad_enabled(True):
      y_, norms = model(X)
      loss = criterion(y_, y)
      running_loss += loss.item() * X.shape[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return running_loss


def numpyize(x):
  return x.detach().cpu().numpy()

def evaluate(model, train_ds):
  # Calculate true probabilities
  forward_probs = model(train_ds.X_train)[0]
  # Calculate reverse probabilities
  reverse_train_set = torch.stack([train_ds.X_train[:,1],
    train_ds.X_train[:,0]], 1)
  reverse_probs, _ = model(reverse_train_set)
  print(forward_probs, reverse_probs)
  forward_np = numpyize(forward_probs)
  reverse_np = numpyize(reverse_probs)
  for i, j in zip(forward_np, reverse_np):
    print(str(i) + "\t" + str(j))


  print box_lib.label_v(reverse_np, forward_np)
  
  print box_lib.label_v(forward_probs.detach().cpu().numpy(),
      reverse_probs.detach().cpu().numpy()).tolist()

def finish_train(train_ds, model):
  for a, b, c in zip(train_ds.X_train, model(train_ds.X_train)[0] ,
      box_lib.check_cond_probs(train_ds, model)):
    print("\t".join([str(a.data.tolist()[0]),
      str(a.data.tolist()[1]), str(b.item()), c]))


def main():

  train_ds, train_dl, dev_ds, dev_dl = get_data()

  model, criterion, optimizer = get_and_maybe_load_model(train_ds)

  run_train_iters(model, criterion, optimizer, train_dl, dev_dl, train_ds,
      dev_ds)

  #finish_train(train_ds, model)
  evaluate(model, train_ds)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
