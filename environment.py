import random
from typing import Tuple, Dict, Any

class TrafficSignalEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        if task == "easy":
            self.min_cars, self.max_cars = 3, 8
            self.arrival_max = 1
            self.max_steps = 10
        elif task == "medium":
            self.min_cars, self.max_cars = 5, 15
            self.arrival_max = 2
            self.max_steps = 20
        else:  # hard
            self.min_cars, self.max_cars = 10, 20
            self.arrival_max = 3
            self.max_steps = 30

        self.green_reduction = 3
        self.lane1 = 0
        self.lane2 = 0
        self.steps = 0
        self.reset()

    def reset(self) -> Tuple:
        self.lane1 = random.randint(self.min_cars, self.max_cars)
        self.lane2 = random.randint(self.min_cars, self.max_cars)
        self.steps = 0
        return self.state()

    def state(self) -> Tuple:
        return (self.lane1, self.lane2)

    def step(self, action: int) -> Tuple:
        arrival = random.randint(0, self.arrival_max)
        if action == 0:
            self.lane1 = max(0, self.lane1 - self.green_reduction)
            self.lane2 += arrival
        else:
            self.lane2 = max(0, self.lane2 - self.green_reduction)
            self.lane1 += arrival

        self.steps += 1
        total = self.lane1 + self.lane2
        max_possible = (self.max_cars * 2)
        reward = 1.0 - (total / max_possible)
        reward = max(0.0, min(1.0, reward))
        done = total == 0 or self.steps >= self.max_steps
        return self.state(), reward, done

    def grade(self) -> float:
        total = self.lane1 + self.lane2
        max_possible = self.max_cars * 2
        return max(0.0, min(1.0, 1.0 - (total / max_possible)))