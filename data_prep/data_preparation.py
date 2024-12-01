import json
import sys
import os
import pickle
from sre_constants import error

import numpy as np
from exceptiongroup import catch
from numpy.lib.polynomial import roots


def list_splitter(list_to_split, ratio):
  elements = len(list_to_split)
  middle = int(elements * ratio)
  return [list_to_split[:middle], list_to_split[middle:]]

target = sys.argv[1]
if not os.path.isdir(target): raise IOError("Must specify a directory of JSON keypoint files.")

keypoint_sequences = []
target_predictions = []

keypoint_sequences_val = []
target_predictions_val = []

roots = set()
file_list = os.listdir(target)
for filename in file_list:
  if "-frame" in filename:
    roots.add(filename.split("-frame")[0])
  elif "_keypoints" in filename:
    n = 3
    groups = filename.split('_')
    root = '_'.join(groups[:n])
    roots.add(root)

idx_split = int(len(roots)*0.8)
idx = 0
num_fail = 0
for root in roots:
  idx += 1
  all_keypoints = []
  filenames = [filename for filename in file_list if root in filename]
  for filename in filenames:
    with open(os.path.join(target,filename), "r") as infile:
      data = json.load(infile)
    try:
      keypoints = data["people"][0]["pose_keypoints_2d"]
    except:
      print("failed to get pose from: " + filename)
      num_fail += 1
      continue
    del keypoints[2::3] # Remove every third element, which contains the confidence for each keypoint, and is not relevant to prediction
    # keypoints = keypoints / np.max(np.abs(keypoints),axis=0) # Squeeze everything between 0 and 1
    all_keypoints.append(keypoints)

  for i in range(24, len(all_keypoints) - 1):
    keypoint_sequences.append(all_keypoints[i - 24:i])
    target_predictions.append(all_keypoints[i + 1])

  # if idx < idx_split:
  #   for i in range(24,len(all_keypoints)-1):
  #     keypoint_sequences.append(all_keypoints[i-24:i])
  #     target_predictions.append(all_keypoints[i+1])
  # else:
  #   for i in range(24,len(all_keypoints)-1):
  #     keypoint_sequences_val.append(all_keypoints[i-24:i])
  #     target_predictions_val.append(all_keypoints[i+1])

print("number of root videos:" + str(len(roots)))
print("number of pose files: " + str(len(file_list)))
print("number of failures: " + str(num_fail))
assert(len(keypoint_sequences)==len(target_predictions))
print("number of sequences: " + str(len(keypoint_sequences)))
keypoint_sequences = np.array(keypoint_sequences)
target_predictions = np.array(target_predictions)

keypoint_sequences, keypoint_sequences_val = list_splitter(keypoint_sequences, 0.8)
target_predictions, target_predictions_val = list_splitter(target_predictions, 0.8)

print("train sequence length: " + str(len(keypoint_sequences)))
print("validation sequence length: " + str(len(keypoint_sequences_val)))

# print("test sequence length: " + str(len(keypoint_sequences_val)))
# assert(len(keypoint_sequences_val)==len(target_predictions_val))
# keypoint_sequences_val = np.array(keypoint_sequences)
# target_predictions_val = np.array(target_predictions)

with open("keypoint_sequences_no_normalize.pkl", "wb") as outfile:
  pickle.dump(keypoint_sequences, outfile)
with open("target_predictions_no_normalize.pkl", "wb") as outfile:
  pickle.dump(target_predictions, outfile)
with open("keypoint_sequences_val_no_normalize.pkl", "wb") as outfile:
  pickle.dump(keypoint_sequences_val, outfile)
with open("target_predictions_val_no_normalize.pkl", "wb") as outfile:
  pickle.dump(target_predictions_val, outfile)
