import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.core.debugger import set_trace
import numpy as np

MIN_IND, MAX_IND = range(2)

UNRELATED, HYPO, HYPER = range(3)

def label(x):
  if x < 0.1:
    return UNRELATED
  elif x > 0.95:
    return HYPER
  else:
    return HYPO

label_v = np.vectorize(label)

def confusion(dataset, model):
  y_pred, _ = model(dataset.X_train)
  print sklearn.metrics.confusion_matrix(
      label_v(dataset.y_train),
      label_v(y_pred.detach().cpu().numpy()))

def set_random_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)


# Dataset for pairs
class BoxDataset(Dataset):
  """Dataset of boxes.
  TODO: Better description

  Arguments:
      A CSV file path
  """

  def __init__(self, csv_path):
    data = np.loadtxt(csv_path)
    self.len = len(data)

    # First two columns are the two entity indices, third is cond prob
    self.X_train = torch.from_numpy(data[:,:2].astype(np.long))
    self.y_train = torch.from_numpy(data[:,2].astype(np.float32))

    # TODO: add test
    vocab = set(np.ravel(data[:,:2]).tolist())
    self.vocab_size = int(max(vocab)) + 1

  def __getitem__(self, index):
    return self.X_train[index], self.y_train[index]

  def __len__(self):
    return self.len


# Model = a tensor of boxes
class Boxes(nn.Module):
  def __init__(self, num_boxes, dim):
    super(Boxes, self).__init__()
    box_mins = torch.rand(num_boxes, dim) * 0.0001
    box_maxs = 1 - torch.rand(num_boxes, dim) * 0.0001
    boxes = torch.stack([box_mins, box_maxs], dim=1)
    self.boxes = nn.Parameter(boxes)

  def forward(self, X):
    """Returns box embeddings for ids"""
    x = self.boxes[X]
    torch.div(self.boxes, torch.max(self.boxes))
    norms = torch.norm(x[MAX_IND] - x[MIN_IND])
    return cond_probs(x[:,0,:,:], x[:,1,:,:]), norms


def volumes(boxes):
  """Calculate (soft) volumes of a tensor containing boxes"""
  return torch.nn.functional.relu(
      boxes[:,MAX_IND,:] - boxes[:, MIN_IND,:]).prod(1)


def intersections(boxes1, boxes2):
  """Calculate pairwise intersection boxes of two tensors containing boxes."""
  intersections_min = torch.max(boxes1[:, MIN_IND, :], boxes2[:, MIN_IND, :])
  intersections_max = torch.min(boxes1[:, MAX_IND, :], boxes2[:, MAX_IND, :])
  return torch.stack([intersections_min, intersections_max], 1)

def cond_probs(boxes1, boxes2):
  """Calculate conditional probabilities of tensors of boxes.

  Pairwise, conditions each box in boxes1 on the corresponding box in boxes2.
  """
  return volumes(intersections(boxes1, boxes2)) / volumes(boxes2)

