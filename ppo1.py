import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from envi import DroneEnv   # your environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG
# ======================
EPISODES = 20
MAX_STEPS = 100
GAMMA = 0.99
LR = 3e-4

# ======================
# ACTOR–CRITIC NETWORK
# ======================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def value(self, state):
        return self.critic(state)

# ======================
# PPO TRAINING
# ======================
def train_ppo(env):
    state_dim = len(env.reset())
    action_dim = 9  # 9 drone actions

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    rewards_history = []
    trees_history = []

    print("\n--- PPO TRAINING STARTED ---\n")

    for ep in range(EPISODES):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)

        log_probs = []
        values = []
        rewards = []

        total_reward = 0

        for _ in range(MAX_STEPS):
            action, logp = model.act(state)
            value = model.value(state)

            next_state, reward = env.step([action.item()])

            log_probs.append(logp)
            values.append(value)
            rewards.append(reward)

            total_reward += reward
            state = torch.tensor(next_state, dtype=torch.float32).to(device)

        # ===== RETURNS =====
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze()

        advantage = returns - values

        # ===== PPO LOSS =====
        actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards_history.append(total_reward)
        trees_history.append(len(env.trees))

        print(f"Episode {ep+1:02d} | Reward: {total_reward:4d} | Trees: {len(env.trees)}")

    # ===== SAVE RESULTS =====
    np.save("ppo_rewards.npy", rewards_history)
    np.save("ppo_trees.npy", trees_history)
    torch.save(model.state_dict(), "ppo_model.pt")

    print("\nPPO training completed and results saved.")
    return rewards_history, trees_history

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    # 🔹 Ask grid size & drones ONLY ONCE
    env = DroneEnv()

    rewards, trees = train_ppo(env)

    # ===== NUMERIC TABLE =====
    print("\nEpisode | Reward | Trees")
    print("------------------------")
    for i in range(len(rewards)):
        print(f"{i+1:^7} | {rewards[i]:^6} | {trees[i]:^5}")

    # ===== GRAPHS =====
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(rewards, marker="o")
    plt.title("PPO Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(trees, marker="o", color="green")
    plt.title("Trees Planted")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.grid()

    plt.tight_layout()
    plt.show()
