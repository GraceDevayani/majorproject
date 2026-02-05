import numpy as np
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================
ppo_rewards   = np.load("ppo_rewards.npy")
mappo_rewards = np.load("mappo_rewards.npy")
qmix_rewards  = np.load("qmix_rewards.npy")

ppo_trees   = np.load("ppo_trees.npy")
mappo_trees = np.load("mappo_trees.npy")
qmix_trees  = np.load("qmix_trees.npy")

episodes = range(1, len(ppo_rewards) + 1)

# ======================
# NUMERIC TABLE
# ======================
print("\n========= FINAL COMPARISON TABLE =========\n")
print(f"{'Algorithm':<10}{'Avg Reward':<15}{'Avg Trees':<15}")
print("-" * 40)
print(f"{'PPO':<10}{ppo_rewards.mean():<15.2f}{ppo_trees.mean():<15.2f}")
print(f"{'MAPPO':<10}{mappo_rewards.mean():<15.2f}{mappo_trees.mean():<15.2f}")
print(f"{'QMIX':<10}{qmix_rewards.mean():<15.2f}{qmix_trees.mean():<15.2f}")

# ======================
# TWO GRAPHS IN ONE FIGURE
# ======================
plt.figure(figsize=(14,5))

# ---- Graph 1: Reward ----
plt.subplot(1,2,1)
plt.plot(episodes, ppo_rewards, label="PPO")
plt.plot(episodes, mappo_rewards, label="MAPPO")
plt.plot(episodes, qmix_rewards, label="QMIX")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Total Reward Comparison")
plt.legend()
plt.grid()

# ---- Graph 2: Trees ----
plt.subplot(1,2,2)
plt.plot(episodes, ppo_trees, label="PPO")
plt.plot(episodes, mappo_trees, label="MAPPO")
plt.plot(episodes, qmix_trees, label="QMIX")
plt.xlabel("Episodes")
plt.ylabel("Trees Planted")
plt.title("Trees Planted Comparison")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
