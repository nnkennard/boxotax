import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import uniform
import numpy as np

MIN_IND, DELTA_IND = range(2)
MAX_IND = DELTA_IND

UNRELATED, HYPER, HYPO, ALIGNED, UNKNOWN = range(5)

UNRELATED_THRESHOLD = 0.05
HYPER_THRESHOLD = 0.90

maybe_hyper = lambda x: x > HYPER_THRESHOLD
maybe_unrelated = lambda x: x < UNRELATED_THRESHOLD
maybe_hypo = lambda x: x < HYPER_THRESHOLD and x > UNRELATED_THRESHOLD

def label_multi(x, y):
  if maybe_hypo(x) and maybe_hyper(y):
    return HYPO
  elif maybe_hyper(x) and maybe_hypo(y):
    return HYPER
  elif maybe_hyper(x) and maybe_hyper(y):
    return ALIGNED
  elif maybe_unrelated(x) and maybe_unrelated(y):
    return UNRELATED
  else:
    return UNKNOWN


def label(x, y):
  if maybe_hyper(x):
    return HYPER
  else:
    return UNRELATED

label_v = np.vectorize(label)
label_multi_v = np.vectorize(label_multi)


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
    X_forward = data[:,:2].astype(np.long)
    X_reverse = np.stack([X_forward[:,1], X_forward[:,0]], 1)
    self.y_forward = data[:,2].astype(np.float32)
    self.y_reverse = data[:,3].astype(np.float32)

    self.X = torch.from_numpy(np.concatenate([X_forward, X_reverse]))
    self.y = torch.from_numpy(np.concatenate([self.y_forward, self.y_reverse]))

    # TODO: add test
    vocab = set(np.ravel(data[:,:2]).tolist())
    self.vocab_size = int(max(vocab)) + 1

  def __getitem__(self, index):
    return self.X[index], self.y[index]

  def __len__(self):
    return self.len


# Model = a tensor of boxes
class Boxes(nn.Module):
  def __init__(self, num_boxes, dim):
    super(Boxes, self).__init__()
    min_distribution = uniform.Uniform(1e-4, 1e-2)
    box_mins = min_distribution.sample((num_boxes, dim))

    delta_distribution = uniform.Uniform(-0.1, -0.01)
    box_deltas = delta_distribution.sample((num_boxes, dim))
    boxes = torch.stack([box_mins, box_deltas], dim=1)
    self.boxes = nn.Parameter(boxes)

  def forward(self, X):
    """Returns box embeddings for ids"""
    x = self.boxes[X]
    torch.div(self.boxes, torch.max(self.boxes))
    norms = None

    cond_probs = get_cond_probs(x[:,0,:,:], x[:,1,:,:])
    predictions = torch.nn.functional.relu(cond_probs)
    return predictions, norms

def volumes(boxes):
  """Calculate (soft) volumes of a tensor containing boxes"""
  return softplus(boxes[:, MAX_IND, :] - boxes[:, MIN_IND, :]).prod(1)


def intersections(boxes1, boxes2):
  """Calculate pairwise intersection boxes of two tensors containing boxes."""

  intersections_min = torch.max(boxes1[:, MIN_IND, :], boxes2[:, MIN_IND, :])
  intersections_max = torch.min(boxes1[:, MAX_IND, :], boxes2[:, MAX_IND, :])
  return torch.stack([intersections_min, intersections_max], 1)

def minmax_format(boxes):
  maxes =  boxes[:, MIN_IND, :] + torch.exp(boxes[:, DELTA_IND, :])
  return torch.stack([boxes[:, MIN_IND, :], maxes], 1)

def get_cond_probs(boxes1, boxes2):
  """Calculate conditional probabilities of tensors of boxes.

  Pairwise, conditions each box in boxes1 on the corresponding box in boxes2.
  """
  b1 = minmax_format(boxes1)
  b2 = minmax_format(boxes2)
  return volumes(intersections(b1, b2)) / volumes(b2)

def softplus(x):
  return torch.log(1.0 + torch.exp(x))
