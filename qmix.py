import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
from envi import DroneEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG
# ======================
EPISODES = 20
MAX_STEPS = 100
GAMMA = 0.99
LR = 5e-4
BUFFER_SIZE = 5000
BATCH_SIZE = 64
TARGET_UPDATE = 5
ACTION_SIZE = 9

# ======================
# AGENT Q NETWORK
# ======================
class AgentQNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# MIXING NETWORK (QMIX)
# ======================
class MixingNetwork(nn.Module):
    def __init__(self, n_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_agents, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, agent_qs):
        return self.net(agent_qs)

# ======================
# REPLAY BUFFER
# ======================
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def push(self, state, actions, reward, next_state):
        self.buffer.append((state, actions, reward, next_state))

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

# ======================
# TRAIN QMIX
# ======================
def train_qmix():
    env = DroneEnv()
    state = env.reset()

    n_agents = len(env.drones)
    state_dim = len(state)
    obs_dim = state_dim // n_agents

    agents = [AgentQNet(obs_dim).to(device) for _ in range(n_agents)]
    target_agents = [AgentQNet(obs_dim).to(device) for _ in range(n_agents)]

    mixer = MixingNetwork(n_agents).to(device)
    target_mixer = MixingNetwork(n_agents).to(device)

    for i in range(n_agents):
        target_agents[i].load_state_dict(agents[i].state_dict())
    target_mixer.load_state_dict(mixer.state_dict())

    params = []
    for a in agents:
        params += list(a.parameters())
    params += list(mixer.parameters())

    optimizer = optim.Adam(params, lr=LR)
    buffer = ReplayBuffer()

    rewards_hist = []
    trees_hist = []

    print("\n=== QMIX TRAINING STARTED ===\n")

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            state_t = torch.tensor(state, dtype=torch.float32).to(device)
            obs = state_t.view(n_agents, -1)

            actions = []
            agent_qs = []

            for i in range(n_agents):
                q = agents[i](obs[i])
                a = torch.argmax(q).item()
                actions.append(a)
                agent_qs.append(q[a])

            next_state, reward = env.step(actions)
            buffer.push(state, actions, reward, next_state)

            state = next_state
            total_reward += reward

            if len(buffer) < BATCH_SIZE:
                continue

            states, acts, rews, next_states = buffer.sample()

            batch_agent_qs = []
            batch_target_qs = []

            states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)

            states_t = states_t.view(BATCH_SIZE, n_agents, -1)
            next_states_t = next_states_t.view(BATCH_SIZE, n_agents, -1)

            for i in range(n_agents):
                q = agents[i](states_t[:, i, :])
                a = torch.tensor([a[i] for a in acts]).to(device)
                batch_agent_qs.append(q.gather(1, a.unsqueeze(1)).squeeze())

                tq = target_agents[i](next_states_t[:, i, :])
                batch_target_qs.append(tq.max(1)[0])

            batch_agent_qs = torch.stack(batch_agent_qs, dim=1)
            batch_target_qs = torch.stack(batch_target_qs, dim=1)

            q_tot = mixer(batch_agent_qs).squeeze()
            with torch.no_grad():
                q_tot_target = target_mixer(batch_target_qs).squeeze()
                y = torch.tensor(rews, dtype=torch.float32).to(device) + GAMMA * q_tot_target

            loss = (q_tot - y).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rewards_hist.append(total_reward)
        trees_hist.append(len(env.trees))

        print(f"Episode {ep+1:02d} | Reward: {total_reward:4d} | Trees: {len(env.trees)}")

        if ep % TARGET_UPDATE == 0:
            for i in range(n_agents):
                target_agents[i].load_state_dict(agents[i].state_dict())
            target_mixer.load_state_dict(mixer.state_dict())

    np.save("qmix_rewards.npy", rewards_hist)
    np.save("qmix_trees.npy", trees_hist)

    print("\nQMIX training completed.\n")

    # ======================
    # PLOT GRAPHS (LIKE PPO & MAPPO)
    # ======================
    episodes = range(1, EPISODES + 1)

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(episodes, rewards_hist, label="QMIX Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("QMIX Reward vs Episodes")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(episodes, trees_hist, label="QMIX Trees", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Trees Planted")
    plt.title("QMIX Trees vs Episodes")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# ======================
# RUN
# ======================
if __name__ == "__main__":
    train_qmix()
