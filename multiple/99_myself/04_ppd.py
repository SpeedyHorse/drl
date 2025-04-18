import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributeDataParallel as DDP
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torch.collectors import SyncDataCollector