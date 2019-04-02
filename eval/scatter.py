import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import collections

labels = ["UNRELATED", "HYPER", "HYPO", "ALIGNED", "UNKNOWN"]

def set_lims(ax):
  ax.set_xlim(-0.05, 1.05)
  ax.set_ylim(-0.05, 1.05)

def main():
  probs = collections.defaultdict(list)
  true_labels = collections.defaultdict(list)
  true_probs = collections.defaultdict(list)

  with open(sys.argv[1], 'r') as f:
    for line in f:
      if not line.strip():
        continue
<<<<<<< HEAD
      (x, y, forward_pred, reverse_pred, forward_true, reverse_true, 
              pred_label,true_label
              )= line.strip().split()
      #if int(x) > 1000000 and int(y) < 1000000 and y != '0':
      if True:
        label_name = labels[int(true_label)]
        probs[label_name].append((forward_pred, reverse_pred))
        true_probs[label_name].append((forward_true, reverse_true))
        true_labels[label_name].append(true_label)
=======
      (x, y, forward_pred, reverse_pred, forward_true, reverse_true, pred_label,
          true_label)= line.strip().split()
      label_name = labels[int(true_label)]
      probs[label_name].append((forward_pred, reverse_pred))
      true_probs[label_name].append((forward_true, reverse_true))
      true_labels[label_name].append(true_label)
>>>>>>> 44ae0172d780422679579d84fbfc4dede847be0f

  fig, axes = plt.subplots(2, 5, figsize=(20, 10))
  for x in range(2):
    for y in range(5):
      axes[x, y].set_xlim(-0.05, 1.05)
      axes[x, y].set_ylim(-0.05, 1.05)

  colormap = ['red', 'blue', 'cyan', 'green', 'grey']
  alpha = 0.4

  for i in range(len(labels)):
    label_name = labels[i]
    fake_label_numbers = [colormap[int(i)] for i in true_labels[label_name]]
    k = np.array(true_probs[label_name])
    ax = plt.subplot(2, 5, int(i) + 1) 
    plt.xlabel(label_name)
    if not i:
      plt.ylabel("true probabilities", rotation=90)
    set_lims(ax)
    if len(k):
      plt.scatter(k[:,0], k[:,1],
<<<<<<< HEAD
        c=fake_label_numbers, alpha=alpha)

=======
          c=fake_label_numbers, alpha=alpha)
>>>>>>> 44ae0172d780422679579d84fbfc4dede847be0f

  for i in range(len(labels)):
    label_name = labels[i]
    fake_label_numbers = [colormap[int(i)] for i in true_labels[label_name]]
    k = np.array(probs[label_name])
    ax = plt.subplot(2, 5, int(i) + 6) 
    if not i:
      plt.ylabel("learned probabilities", rotation=90)
    set_lims(ax)
    if len(k):
<<<<<<< HEAD
      #if label_name == "HYPO":
      #  plt.scatter(k[:,1], k[:,0],
      #    c=fake_label_numbers, alpha=alpha)
      #else:
      if True:
        plt.scatter(k[:,0], k[:,1],
=======
      plt.scatter(k[:,0], k[:,1],
>>>>>>> 44ae0172d780422679579d84fbfc4dede847be0f
          c=fake_label_numbers, alpha=alpha)

  plt.show()

if __name__ == "__main__":
  main()
