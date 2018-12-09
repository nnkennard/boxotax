import sklearn.metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

MIN_IND, MAX_IND = range(2)

UNRELATED, HYPO, HYPER = range(3)


def label(x):
  if x < 0.05:
    return UNRELATED
  elif x > 0.95:
    return HYPER
  else:
    return HYPO

label_v = np.vectorize(label)

unused_classification_stuff = """
def check_cond_probs(dataset, model):
  reversed_dataset = torch.stack([dataset.X_train[:,1], dataset.X_train[:,0]],
      dim=1)
  y_pred_actual, _ = model(dataset.X_train)
  y_pred_reverse, _ = model(reversed_dataset)
  print(y_pred_actual)
  print(y_pred_reverse)
  sure_labels = []
  for x, z in zip(y_pred_actual, y_pred_reverse):
    if label(x) == HYPO and label(z) == HYPER:
      sure_labels.append("hypo")
    elif label(x) == HYPER and label(z) == HYPO:
      sure_labels.append("hyper")
    else:
      sure_labels.append("unrelated")
  return sure_labels """

def confusion(dataset, model):
  y_pred, _ = model(dataset.X_train)
  true_labels = label_v(dataset.y_train)
  pred_labels = label_v(y_pred.detach().cpu().numpy())
  print sklearn.metrics.confusion_matrix(true_labels, pred_labels)
  #print sklearn.metrics.f1_score(true_labels, pred_labels, average='micro')

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
    self.y_train = split_bernoulli(
        torch.from_numpy(data[:,2].astype(np.float32)))

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
    cond_probs = get_cond_probs(x[:,0,:,:], x[:,1,:,:])
    predictions = torch.nn.functional.relu(cond_probs)
    return predictions, norms

def volumes(boxes):
  """Calculate (soft) volumes of a tensor containing boxes"""
  return softplus(
      boxes[:,MAX_IND,:] - boxes[:, MIN_IND,:]).prod(1)


def intersections(boxes1, boxes2):
  """Calculate pairwise intersection boxes of two tensors containing boxes."""
  intersections_min = torch.max(boxes1[:, MIN_IND, :], boxes2[:, MIN_IND, :])
  intersections_max = torch.min(boxes1[:, MAX_IND, :], boxes2[:, MAX_IND, :])
  return torch.stack([intersections_min, intersections_max], 1)

def get_cond_probs(boxes1, boxes2):
  """Calculate conditional probabilities of tensors of boxes.

  Pairwise, conditions each box in boxes1 on the corresponding box in boxes2.
  """
  probs = volumes(intersections(boxes1, boxes2)) / volumes(boxes2)
  #print("boxes2")
  #print(volumes(boxes2))
  return split_bernoulli(probs)

def split_bernoulli(probabilities):
  return probabilities
  return torch.stack([probabilities, 1 - probabilities], 1)

def softplus(x):
  return torch.log(1.0 + torch.exp(x))
