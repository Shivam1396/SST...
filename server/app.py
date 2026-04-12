import sys
import os

# Add parent directory to path so environment.py can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from environment import TrafficSignalEnv
import uvicorn

app = FastAPI()
envs = {}

@app.get("/")
def root():
    return {"status": "ok", "env": "traffic-signal-env"}

@app.post("/reset")
def reset(task: str = "easy"):
    env = TrafficSignalEnv(task=task)
    envs[task] = env
    state = env.reset()
    return {"state": list(state)}

@app.post("/step")
def step(task: str = "easy", action: int = 0):
    env = envs.get(task)
    if not env:
        env = TrafficSignalEnv(task=task)
        envs[task] = env
        env.reset()
    state, reward, done = env.step(action)
    return {"state": list(state), "reward": reward, "done": done}

@app.get("/state")
def state(task: str = "easy"):
    env = envs.get(task)
    if not env:
        return {"state": None}
    return {"state": list(env.state())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)