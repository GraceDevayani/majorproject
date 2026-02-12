import numpy as np
import random
import os
from colorama import Fore, Style, init

init(autoreset=True)

class DroneEnv:
    def __init__(self):
        # ===== USER INPUT =====
        self.grid_size = int(input("Enter grid size (e.g., 8, 10, 12): "))
        self.num_drones = int(input("Enter number of drones: "))

        self.reset()

    # ======================
    # RESET ENVIRONMENT
    # ======================
    def reset(self):
        self.drones = [
            (random.randint(0, self.grid_size - 1),
             random.randint(0, self.grid_size - 1))
            for _ in range(self.num_drones)
        ]

        self.trees = set()
        init_trees = int(self.grid_size * self.grid_size * 0.1)

        while len(self.trees) < init_trees:
            self.trees.add((
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ))

        return self.get_state()

    # ======================
    # STEP FUNCTION
    # ======================
    def step(self, actions):
        reward = -1  # movement penalty

        moves = [
            (0, 0),    # 0 stay
            (-1, 0),   # 1 up
            (1, 0),    # 2 down
            (0, -1),   # 3 left
            (0, 1),    # 4 right
            (-1, -1),  # 5 up-left
            (-1, 1),   # 6 up-right
            (1, -1),   # 7 down-left
            (1, 1)     # 8 down-right
        ]

        new_positions = []

        for i, action in enumerate(actions):
            x, y = self.drones[i]
            dx, dy = moves[action]
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                new_positions.append((nx, ny))
            else:
                new_positions.append((x, y))
                reward -= 2  # boundary penalty

        self.drones = new_positions

        # Plant trees
        for d in self.drones:
            if d not in self.trees:
                self.trees.add(d)
                reward += 5

        return self.get_state(), reward

    # ======================
    # GLOBAL STATE
    # ======================
    def get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size, 2))

        for x, y in self.trees:
            grid[x, y, 0] = 1

        for x, y in self.drones:
            grid[x, y, 1] = 1

        return grid.flatten()

    # ======================
    # VISUAL RENDER (VIDEO STYLE)
    # ======================
    def render(self, episode=None, step=None, reward=None, total_reward=None):
        os.system("cls" if os.name == "nt" else "clear")

        # ===== HEADER INFO =====
        if episode is not None:
            print(f"Episode       : {episode}")
        if step is not None:
            print(f"Step          : {step}")
        if reward is not None:
            print(f"Step Reward   : {reward}")
        if total_reward is not None:
            print(f"Total Reward  : {total_reward}")

        print(f"Trees Planted : {len(self.trees)}\n")

        # ===== GRID =====
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                char = "."
                color = Style.RESET_ALL

                if (i, j) in self.trees:
                    char, color = "T", Fore.GREEN

                if (i, j) in self.drones:
                    char, color = "D", Fore.CYAN

                print(color + char + " " + Style.RESET_ALL, end="")
            print()
