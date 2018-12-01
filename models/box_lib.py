import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.core.debugger import set_trace
import numpy as np

# Dataset for pairs
class BoxDataset(Dataset):
  """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

  Arguments:
      A CSV file path
  """

  def __init__(self, csv_path):
    data = np.loadtxt(csv_path)
    self.len = len(data)
    self.X_train = torch.from_numpy(data[:,:2].astype(np.long))
    self.y_train = torch.from_numpy(data[:,2].astype(np.float32))
    print(self.X_train.shape)

  def __getitem__(self, index):
    return self.X_train[index], self.y_train[index]

  def __len__(self):
    return self.len

# Model = a tensor of boxes
class Boxes(nn.Module):
  def __init__(self, num_boxes, dim):
    super(Boxes, self).__init__()
    #box_mins = torch.rand(num_boxes, dim)
    #box_maxs = box_mins + torch.rand(num_boxes, dim) * (1 - box_mins)
    box_mins = torch.rand(num_boxes, dim) * 0.0001
    box_maxs = 1 - torch.rand(num_boxes, dim) * 0.0001
    boxes = torch.stack([box_mins, box_maxs], dim=1)
    self.boxes = nn.Parameter(boxes)

  def forward(self, X):
    """Returns box embeddings for ids"""
    x = self.boxes[X]
    o = cond_probs(x[:,0,:,:], x[:,1,:,:])
    print("Cond probs")
    print(o)
    assert all(np.less_equal(i.detach().numpy(), 1.0) for i in o)
    return o


MIN_IND, MAX_IND = 0, 1

def volumes(boxes):
  r = torch.nn.functional.relu(boxes[:,MAX_IND,:] - boxes[:, MIN_IND,:]).prod(1)
  return r

def intersections(boxes1, boxes2):
  intersections_min = torch.max(boxes1[:, MIN_IND, :], boxes2[:, MIN_IND, :])
  intersections_max = torch.min(boxes1[:, MAX_IND, :], boxes2[:, MAX_IND, :])
  return torch.stack([intersections_min, intersections_max], 1)

def cond_probs(boxes1, boxes2):
  vols =  volumes(intersections(boxes1, boxes2))
  vols2 = volumes(boxes2)
  return vols/volumes(boxes2)

