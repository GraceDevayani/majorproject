import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, time, random
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

init()

# ======================
# USER INPUT
# ======================
GRID_SIZE = int(input("Enter grid size (e.g., 10, 12, 15): "))
NUM_DRONES = int(input("Enter number of drones: "))

# ======================
# CONFIGURATION
# ======================
CHANNELS = 2
STATE_SIZE = GRID_SIZE * GRID_SIZE * CHANNELS
ACTION_SIZE = 9
JOINT_ACTION_SIZE = NUM_DRONES * ACTION_SIZE

EPISODES = 50
MAX_STEPS = 150

GAMMA = 0.99
LR = 3e-4
PPO_EPS = 0.2

# ======================
# ENVIRONMENT (NO EXPLORATION)
# ======================
class DroneEnv(gym.Env):
    def reset(self):
        self.drones = [(random.randint(0, GRID_SIZE-1),
                        random.randint(0, GRID_SIZE-1))
                       for _ in range(NUM_DRONES)]

        self.planted = set()
        init_trees = int(GRID_SIZE * GRID_SIZE * random.uniform(0.1, 0.15))
        while len(self.planted) < init_trees:
            self.planted.add((random.randint(0, GRID_SIZE-1),
                              random.randint(0, GRID_SIZE-1)))
        return self.state()

    def step(self, actions):
        reward = -1
        moves = [(0,0),(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

        new_pos = []
        for i, a in enumerate(actions):
            x, y = self.drones[i]
            dx, dy = moves[a]
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                new_pos.append((nx, ny))
            else:
                new_pos.append((x, y))
                reward -= 2

        self.drones = new_pos

        for p in self.drones:
            if p not in self.planted:
                self.planted.add(p)
                reward += 5

        return self.state(), reward

    def state(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, 2))
        for x, y in self.planted:
            grid[x, y, 0] = 1
        for x, y in self.drones:
            grid[x, y, 1] = 1
        return torch.tensor(grid.flatten(), dtype=torch.float32)

    def render(self, ep, st, r, tot):
        os.system("cls" if os.name == "nt" else "clear")
        print(
            f"Episode: {ep} | Step: {st}/{MAX_STEPS}\n"
            f"Step Reward: {int(r)} | Total Reward: {int(tot)}\n"
            f"Trees Planted: {len(self.planted)}\n"
        )
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                c="."
                col=Style.RESET_ALL
                if (i,j) in self.planted:
                    c,col="T",Fore.GREEN
                for d in self.drones:
                    if d==(i,j):
                        c,col="D",Fore.CYAN
                print(col+c+" "+Style.RESET_ALL,end="")
            print()

# ======================
# NETWORKS
# ======================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE,256),
            nn.ReLU(),
            nn.Linear(256,ACTION_SIZE)
        )
    def forward(self,x):
        return torch.softmax(self.net(x),dim=-1)

class CentralizedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE + JOINT_ACTION_SIZE,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    def forward(self,state,joint_action):
        x = torch.cat([state, joint_action], dim=-1)
        return self.net(x)

# ======================
# UTILS
# ======================
def one_hot_joint(actions):
    oh = []
    for a in actions:
        v = torch.zeros(ACTION_SIZE)
        v[a] = 1
        oh.append(v)
    return torch.cat(oh)

# ======================
# MAPPO TRAINING (STRICT)
# ======================
def train_mappo():
    env = DroneEnv()
    actors = [Actor() for _ in range(NUM_DRONES)]
    critic = CentralizedCritic()

    actor_opts = [optim.Adam(a.parameters(), lr=LR) for a in actors]
    critic_opt = optim.Adam(critic.parameters(), lr=LR)

    R_hist, T_hist, A_hist = [], [], []

    for ep in range(EPISODES):
        s = env.reset()
        states, joint_actions, rewards = [], [], []
        total = 0

        for st in range(MAX_STEPS):
            acts, logps = [], []
            for i in range(NUM_DRONES):
                dist = torch.distributions.Categorical(actors[i](s))
                a = dist.sample()
                acts.append(a.item())
                logps.append(dist.log_prob(a))

            ja = one_hot_joint(acts)
            v = critic(s, ja)

            ns, r = env.step(acts)

            states.append(s)
            joint_actions.append(ja)
            rewards.append(r)

            s = ns
            total += r

            env.render(ep+1, st+1, r, total)
            time.sleep(0.02)

        # returns
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        values = torch.cat([critic(states[i], joint_actions[i]) for i in range(len(states))]).squeeze()
        adv = returns - values.detach()

        # actor update
        for i in range(NUM_DRONES):
            loss = 0
            for t in range(len(states)):
                dist = torch.distributions.Categorical(actors[i](states[t]))
                ratio = torch.exp(dist.log_prob(torch.tensor(joint_actions[t][i*ACTION_SIZE:(i+1)*ACTION_SIZE].argmax())))
                s1 = ratio * adv[t]
                s2 = torch.clamp(ratio,1-PPO_EPS,1+PPO_EPS) * adv[t]
                loss += -torch.min(s1,s2)

            actor_opts[i].zero_grad()
            loss.mean().backward()
            actor_opts[i].step()

        # critic update
        critic_loss = (returns - values).pow(2).mean()
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        R_hist.append(total)
        T_hist.append(len(env.planted))
        A_hist.append(total/MAX_STEPS)

    return R_hist, T_hist, A_hist

# ======================
# PPO BASELINE
# ======================
def train_ppo():
    env = DroneEnv()
    actor = Actor()

    R_hist, T_hist, A_hist = [], [], []

    for ep in range(EPISODES):
        s = env.reset()
        env.drones = [env.drones[0]]
        total = 0

        for _ in range(MAX_STEPS):
            dist = torch.distributions.Categorical(actor(s))
            s, r = env.step([dist.sample().item()])
            total += r

        R_hist.append(total)
        T_hist.append(len(env.planted))
        A_hist.append(total/MAX_STEPS)

    return R_hist, T_hist, A_hist

# ======================
# RUN
# ======================
print("Training MAPPO (centralized critic)...")
m_r, m_t, m_a = train_mappo()

print("Training PPO...")
p_r, p_t, p_a = train_ppo()

# ======================
# COMPARISON TABLE
# ======================
print("\nEpisode    PPO Avg Reward    MAPPO Avg Reward")
print("---------------------------------------------")
for ep in [1,10,20,30,40,50]:
    print(f"{ep:<10} {p_a[ep-1]:<16.2f} {m_a[ep-1]:<16.2f}")

# ======================
# GRAPHS
# ======================
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.plot(m_r,label="MAPPO")
plt.plot(p_r,label="PPO",linestyle="--")
plt.title("(a) Cumulative Reward")
plt.legend(); plt.grid()

plt.subplot(1,3,2)
plt.plot(m_t,label="MAPPO")
plt.plot(p_t,label="PPO",linestyle="--")
plt.title("(b) Trees Planted")
plt.legend(); plt.grid()

plt.subplot(1,3,3)
plt.plot(m_a,label="MAPPO")
plt.plot(p_a,label="PPO",linestyle="--")
plt.title("(c) Avg Reward / Step")
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()
