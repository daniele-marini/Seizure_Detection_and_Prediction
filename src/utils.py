import torch
import numpy as np
import random
import mne

def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def process_file(file_path,start,stop,end_recording):

  data=[]
  pre = 900 # 15 minutes
  
  # load the data
  raw = mne.io.read_raw_edf(file_path, preload=True)

  # remove the bad channels
  BAD_CHANNELS = ['EKG EKG', 'SPO2', '1', '2', 'HR', 'MK']
  raw.drop_channels(BAD_CHANNELS)

  # apply filtering
  raw.filter(l_freq=8,h_freq=13)

  # divide channels per seconds
  channels=[]
  for i in range(29):
    channels.append(raw[i][0].reshape(-1,512))

  # no seizure with 0 seconds of overlap
  for i in range(0,start-pre,5):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,0])
  # pre-ictal with 2 seconds of overlap
  for i in range(start-pre,start,3): 
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,1])
  # seizure with 3 seconds of overlap
  for i in range(start,stop,2):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,2])
  # no seizure with 0 seconds of overlap
  for i in range(stop,end_recording,5):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,0])

  return data


def create_test(file_path,start,stop,end):

  data=[]
  pre = 900

  # load the data
  raw = mne.io.read_raw_edf(file_path, preload=True)

  # remove the bad channels
  BAD_CHANNELS = ['EKG EKG', 'SPO2', '1', '2', 'HR', 'MK']
  raw.drop_channels(BAD_CHANNELS)

  # divide channels per seconds
  channels=[]
  for i in range(29):
    channels.append(raw[i][0].reshape(-1,512))

  for i in range(0,start-pre,5):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,0])

  for i in range(start-pre,start,5):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,1])

  for i in range(start,stop,5):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,2])

  for i in range(stop,end,5):
    sample = [channel[i:i + 5] for channel in channels]
    data.append([sample,0])

  return data



