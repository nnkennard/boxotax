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
flags.DEFINE_string('report_path', None, 'directory where training reports are saved')
flags.DEFINE_string('result_path', None, 'directory where results are saved')
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
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, eps
      =1e-6)
  return model, criterion, optimizer


def get_save_file_name(epoch, is_best):
  if is_best:
    config_number = "best"
  else:
    config_number = str(epoch).zfill(4)

  path = "".join([FLAGS.save_path, "/", FLAGS.config, "_", config_number,
      ".params"])
  return path


def get_report_file_name():
  return FLAGS.report_path + '/' + FLAGS.config + '.train_report.tsv'


def get_result_file_name(dataset):
  return "".join([FLAGS.result_path, '/', FLAGS.config, '.', dataset,
    '_results.tsv'])


def save_current_model(model, epoch, is_best=False):
  path = get_save_file_name(epoch, is_best)
  torch.save(model.state_dict(), path)

def print_report(epoch, dev_loss, train_loss, file_handle):
  report_str = "".join([str(epoch), "\t", str(dev_loss), "\t",
    str(train_loss)])
  file_handle.write(report_str)
  file_handle.write("\n")
  print(report_str)


def run_train_iters(model, criterion, optimizer,
    train_dl, dev_dl, train_ds, dev_ds):

  best_dev_loss = float('inf')
  best_dev_loss_epoch = -1
  dev_labels = dev_ds.y.to(FLAGS.device)
  train_labels = train_ds.y.to(FLAGS.device)

  with open(get_report_file_name(), 'w') as f:

    for epoch in range(FLAGS.num_epochs):

      running_loss = run_train_iter(model, criterion, optimizer, train_dl)
      print(running_loss)

      model.eval()

      dev_loss = criterion(model(dev_ds.X)[0],
          dev_labels).item()/len(dev_dl.dataset)
      if dev_loss < best_dev_loss:
        print("Dev loss", dev_loss, "Best dev loss", best_dev_loss)
        best_dev_loss = dev_loss
        best_dev_loss_epoch = epoch
        save_current_model(model, epoch, is_best=True)

        train_loss = criterion(model(train_ds.X)[0],
          dev_labels).item()/len(train_dl.dataset)
        print_report(epoch, dev_loss, train_loss, sys.stdout)
        print_results_to_file(model, train_ds, sys.stdout)

      #if epoch % 1  == 0:
       # box_lib.confusion(dev_ds, model)

      if epoch % FLAGS.save_freq == 0:
        save_current_model(model, epoch)

      print_report(epoch, dev_loss, running_loss/len(train_dl.dataset), f)

      if epoch >= best_dev_loss_epoch + FLAGS.patience:
        break



def run_train_iter(model, criterion, optimizer, train_dl):
  model.train()
  running_loss = 0.0

  for X, y in train_dl:
    torch.cuda.empty_cache()
    X, y = X.to(FLAGS.device), y.to(FLAGS.device)

    #TODO: Use more canonical regularization maybe
    with torch.set_grad_enabled(True):
      y_, _ = model(X)
      print(y_)
      loss = criterion(y_, y)
      running_loss += loss.item() * X.shape[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return running_loss


def numpyize(x):
  return x.detach().cpu().numpy()


def get_probabilities_from_dataset(model, ds):
  model.eval()
  probs = numpyize(model(ds.X)[0])
  num_examples = len(probs)/2
  forward_probs = probs[:num_examples]
  reverse_probs = probs[num_examples:]
  return numpyize(ds.X[:num_examples]), forward_probs, reverse_probs


def print_results_to_file(model, ds, f):
  true_pairs, pred_forward, pred_reverse = get_probabilities_from_dataset(
      model, ds)
  multi_pred_labels = box_lib.label_multi_v(pred_forward, pred_reverse)
  multi_labels = box_lib.label_multi_v(ds.y_forward, ds.y_reverse)

  for i in zip(true_pairs[:,0], true_pairs[:,1],
     pred_forward, pred_reverse,
     ds.y_forward, ds.y_reverse,
     multi_pred_labels, multi_labels):
   f.write("\t".join([str(j) for j in i]))
   f.write("\n")


def evaluate(model, train_ds, dev_ds):
  model.load_state_dict(torch.load(get_save_file_name(epoch=-1, is_best=True)))
  with open(get_result_file_name('train'), 'w') as f:
    print_results_to_file(model, train_ds, f)
  with open(get_result_file_name('dev'), 'w') as f:
    print_results_to_file(model, dev_ds, f)


def main():

  train_ds, train_dl, dev_ds, dev_dl = get_data()

  model, criterion, optimizer = get_and_maybe_load_model(train_ds)

  run_train_iters(model, criterion, optimizer, train_dl, dev_dl, train_ds,
      dev_ds)

  #finish_train(train_ds, model)
  evaluate(model, train_ds, dev_ds)
  print("Final boxes")
  print(model.boxes)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
