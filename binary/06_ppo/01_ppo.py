from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from flow_package.binary_flow_env import BinaryFlowEnv, InputType


def calculate_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if tp + fp != 0 else -1
    recall = tp / (tp + fn) if tp + fn != 0 else -1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn != 0 else 0.0

    if precision < 0:
        precision = 0.0
    if recall < 0:
        recall = 0.0
    return accuracy, precision, recall, f1, fpr


TRAIN_PATH = os.path.abspath("../../raw_after_sample/cicids2017/binary_train.csv")
TEST_PATH = os.path.abspath("../../raw_after_sample/cicids2017/binary_test.csv")

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
train_input = InputType(
    input_features=train_data.drop(columns=["Number Label"]),
    input_labels=train_data["Number Label"],
    reward_list=[1.0, -1.0]
)
train_env = BinaryFlowEnv(train_input)

test_input = InputType(
    input_features=test_data.drop(columns=["Number Label"]),
    input_labels=test_data["Number Label"],
    reward_list=[1.0, -1.0]
)
test_env = BinaryFlowEnv(test_input)

# A2Cエージェントの作成
print("make model")
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=0,
    learning_rate=1e-5,
    n_steps=2048,
    batch_size=32,
    n_epochs=20
)

for i in range(10):
    print(f"train start {i}")
    # トレーニング
    model.learn(total_timesteps=100000)
    print("train end")

    # model.save("ppo_no4")

    # model = PPO.load("ppo_no4")

    # トレーニング済みモデルでテスト
    print("test start")
    confusion_array = np.zeros((2, 2), dtype=np.int32)
    obs = test_env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        index = info["confusion_position"]
        confusion_array[index[0], index[1]] += 1
        if terminated or truncated:
            obs = test_env.reset()  # Changed to test_env.reset()

    # print(confusion_array)

    tp = confusion_array[1, 1]
    tn = confusion_array[0, 0]
    fp = confusion_array[1, 0]
    fn = confusion_array[0, 1]

    accuracy, precision, recall, f1, fpr = calculate_metrics(tp, tn, fp, fn)
    print(accuracy, precision, recall, f1, fpr)
else:
    with open("test_result.csv", "w") as f:
        f.write("accuracy,precision,recall,f1,fpr\n")
        f.write(f"{accuracy},{precision},{recall},{f1},{fpr}\n")
    plt.figure()
    plt.bar(
        ["accuracy", "precision", "recall", "f1", "fpr"],
        [accuracy, precision, recall, f1, fpr]
    )
    plt.pause(0.1)