import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from envi import DroneEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG
# ======================
EPISODES = 20
MAX_STEPS = 100
GAMMA = 0.99
LR = 3e-4
ACTION_SIZE = 9

# ======================
# ACTOR (DECENTRALIZED)
# ======================
class Actor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

# ======================
# CENTRALIZED CRITIC
# ======================
class CentralCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s):
        return self.net(s)

# ======================
# TRAIN MAPPO
# ======================
def train_mappo():
    env = DroneEnv()
    state = env.reset()

    n_agents = len(env.drones)
    state_dim = len(state)
    obs_dim = state_dim // n_agents

    actors = [Actor(obs_dim).to(device) for _ in range(n_agents)]
    critic = CentralCritic(state_dim).to(device)

    actor_opts = [optim.Adam(a.parameters(), lr=LR) for a in actors]
    critic_opt = optim.Adam(critic.parameters(), lr=LR)

    rewards_hist = []
    trees_hist = []

    print("\n=== MAPPO TRAINING STARTED ===\n")

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        log_probs = []
        values = []
        rewards = []

        for step in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).to(device)
            obs = state_t.view(n_agents, -1)

            actions = []
            step_logp = []

            for i in range(n_agents):
                dist = Categorical(actors[i](obs[i]))
                a = dist.sample()
                actions.append(a.item())
                step_logp.append(dist.log_prob(a))

            value = critic(state_t)

            next_state, reward = env.step(actions)

            log_probs.append(torch.stack(step_logp).sum())
            values.append(value.squeeze())
            rewards.append(reward)

            state = next_state
            total_reward += reward

        # ===== COMPUTE RETURNS & ADVANTAGES =====
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.stack(values)
        advantages = returns - values.detach()

        # ===== UPDATE CRITIC =====
        critic_loss = (values - returns).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # ===== UPDATE ACTORS =====
        for opt in actor_opts:
            opt.zero_grad()

        actor_loss = -(torch.stack(log_probs) * advantages).mean()
        actor_loss.backward()

        for opt in actor_opts:
            opt.step()

        rewards_hist.append(total_reward)
        trees_hist.append(len(env.trees))

        print(f"Episode {ep+1:02d} | Reward: {total_reward:4d} | Trees: {len(env.trees)}")

    # ======================
    # SAVE RESULTS
    # ======================
    torch.save([a.state_dict() for a in actors], "mappo_actors.pt")
    torch.save(critic.state_dict(), "mappo_critic.pt")

    np.save("mappo_rewards.npy", rewards_hist)
    np.save("mappo_trees.npy", trees_hist)

    print("\nMAPPO training completed.\n")

    # ======================
    # PLOT GRAPHS (LIKE PPO)
    # ======================
    episodes = range(1, EPISODES + 1)

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(episodes, rewards_hist, label="MAPPO Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("MAPPO Reward vs Episodes")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(episodes, trees_hist, label="MAPPO Trees", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Trees Planted")
    plt.title("MAPPO Trees vs Episodes")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# ======================
# RUN
# ======================
if __name__ == "__main__":
    train_mappo()
