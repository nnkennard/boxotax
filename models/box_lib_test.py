import box_lib
import torch
import numpy as np

def test_min_max():
  assert box_lib.MIN_IND == 0
  assert box_lib.MAX_IND == 1

def get_sample_boxes():
  return np.array([
      # Box 1
      [
        # Mins
          [1, 1],
        # Maxes
          [5, 5]
        ],
      # Box 2
      [
        # Mins
          [3, 4],
        # Maxes
          [10, 19]
      ],
    ])


def test_volumes():
  boxes = get_sample_boxes()
  volumes = box_lib.volumes(boxes)
  np.testing.assert_array_equal(volumes, np.array([16, 105]))

def test_intersections():
  boxes = get_sample_boxes()
  boxes_1 = torch.from_numpy(np.array([boxes[0]]))
  boxes_2 = torch.from_numpy(np.array([boxes[1]]))
  intersections = box_lib.intersections(boxes_1, boxes_2)
  np.testing.assert_array_equal(intersections, np.array([
    [[3, 4], [5, 5]
      ]
    ]))


