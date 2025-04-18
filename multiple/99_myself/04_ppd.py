import pandas as pd
from glob import glob
from flow_package.multi_flow_env import InputType, MultipleFlowEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.collectors import SyncDataCollector
from torchrl.envs import TransformedEnv, Compose, GymEnv, GymWrapper, DoubleToFloat
from torchrl.modules import QValueModule, EGreedyModule
from tensordict.nn import TensorDictModule

import os
from tqdm import tqdm

PORT_DIM = 32

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def _read_data():
    train = pd.DataFrame()
    test = pd.DataFrame()

    train_files = glob("data/train/*.csv")
    test_files = glob("data/test/*.csv")

    for file in train_files:
        train = pd.concat([train, pd.read_csv(file)])

    for file in test_files:
        test = pd.concat([test, pd.read_csv(file)])

    train = train.dropna().dropna(how="all", axis=1).drop_duplicates()
    test = test.dropna().dropna(how="all", axis=1).drop_duplicates()

    return train, test

def create_env(device):
    train, test = _read_data()

    train_input = InputType(
        input_features=train.drop(columns=["Number Label"]),
        input_labels=train["Number Label"],
        reward_list=[1.0, -1.0]
    )
    train_gym_env = MultipleFlowEnv(train_input)
    train_base_env = GymWrapper(train_gym_env)
    train_env = TransformedEnv(
        train_base_env,
        Compose(
            DoubleToFloat()
        )
    )

    test_input = InputType(
        input_features=test.drop(columns=["Number Label"]),
        input_labels=test["Number Label"],
        reward_list=[1.0, -1.0],
        type_env="test"
    )
    test_gym_env = MultipleFlowEnv(test_input)
    test_base_env = GymWrapper(test_gym_env)
    test_env = TransformedEnv(
        test_base_env,
        Compose(
            DoubleToFloat()
        )
    )

    train_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    test_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    return train_env, test_env

class DeepFlowNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.protocol_embedding = nn.Embedding(256, 8)  # -> 8
        self.port_embedding = nn.Embedding(65536, PORT_DIM)  # -> 8
        # other inputs are not embedding: n_inputs - 2

        # all inputs: n_inputs - 2 + 8 + 8 = n_inputs + 14
        n_inputs = input_dim + 6 + PORT_DIM

        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        port_emb = self.port_embedding(x[0].long())
        protocol_emb = self.protocol_embedding(x[1].long())

        # print(port_emb.shape, protocol_emb.shape, x[2].shape)

        renew = torch.cat([port_emb, protocol_emb, x[2]], dim=1)

        renew = F.relu(self.fc1(renew))
        renew = F.relu(self.fc2(renew))
        return self.fc3(renew)

def create_network_modules(env: TransformedEnv, device):
    action_dim = env.action_spec.shape[-1]
    observation_dim = env.observation_spec["observation"].shape

    q_net = DeepFlowNetwork(observation_dim, action_dim).to(device)

    q_module = TensorDictModule(
        q_net,
        in_keys=["observation"],
        out_keys=["action_values"]
    )

    q_value_module = QValueModule(
        q_module,
        action_value_key="action_values",
        out_keys=["action"]
    )

    policy_module = EGreedyModule(
        q_value_module,
        action_key="action_spec",
        eps_init=1.0,
        eps_end=0.01,
        annealing_num_steps=1000000
    )

    target_q_net = DeepFlowNetwork(observation_dim, action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    target_q_module = TensorDictModule(
        target_q_net,
        in_keys=["observation"],
        out_keys=["target_action_values"]
    )

    return q_net, policy_module, target_q_net, target_q_module

def setup_collector_and_buffer(env, policy_module, device, frames_per_batch=1000, total_frames=100000, buffer_size=100000):
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage = LazyTensorStorage(buffer_size),
        sampler=RandomSampler(),
        batch_size=32
    )

    return collector, replay_buffer

def compute_network_loss(batch, q_net, target_q_net, gamma=0.99):
    states = batch["observation"]
    actions = batch["action"]
    rewards = batch["reward"]
    next_states = batch["next_observation"]
    dones = batch["done"]

    q_values = q_net(states).gather(1, actions.unsqueeze(1).to(torch.float64)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_q_net(next_states).max(1)[0]
        target_values = rewards + gamma * next_q_values * (~dones)

    loss = F.mse_loss(q_values, target_values)

    return loss

def train(rank, world_size, buffer_size=100000, gamma=0.99, target_update=10, total_frames=1000000):
    setup(rank, world_size)
    device = torch.device(rank)

    train_env, test_env = create_env(device)

    q_net, policy_module, target_q_net, target_q_module = create_network_modules(train_env, device)

    collector, replay_buffer = setup_collector_and_buffer(
        train_env,
        policy_module,
        device,
        frames_per_batch=1000,
        total_frames=total_frames,
        buffer_size=buffer_size
    )

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)

    pbar = tqdm(total=total_frames) if rank == 0 else None
    step_count = 0

    for i, tensordict_data in enumerate(collector):
        replay_buffer.extend(tensordict_data.reshape(-1).cpu())
        step_count += tensordict_data.numel()

        if len(replay_buffer) >= 1000:
            for _ in range(tensordict_data.numel() // 4):
                batch = replay_buffer.sample().to(device)

                loss = compute_network_loss(batch, q_net, target_q_net, gamma=gamma)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

        if i % target_update == 0:
            target_q_net.load_state_dict(q_net.module.state_dict())

        if rank == 0:
            pbar.update(tensordict_data.numel())
            pbar.set_description(f"Step: {step_count}")

    if rank == 0:
        pbar.close()

    cleanup()

    return q_net, target_q_net

def main():
    world_size = torch.cuda.device_count()
    print(f"world_size: {world_size}")

    import torch.multiprocessing as mp
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()