# Traffic Signal Control — OpenEnv

A real-world reinforcement learning environment where an AI agent controls traffic signals at a two-lane intersection to minimize congestion.

Built for the **Meta × Scaler OpenEnv Hackathon**.

---

## The Problem

Urban traffic congestion wastes millions of hours daily. This environment simulates a two-lane intersection where an agent must decide which lane gets the green light at each step — learning to clear cars faster than random or fixed-cycle signals.

---

## Environment Overview

At each step, the agent observes how many cars are waiting in each lane and chooses which lane gets the green light. Cars in the green lane are reduced, while the red lane receives new arrivals. The goal is to minimize total cars across both lanes.

### Action Space
| Action | Meaning |
|--------|---------|
| `0` | Give green light to Lane 1 |
| `1` | Give green light to Lane 2 |

### State Space
A tuple `(lane1_cars, lane2_cars)` representing the number of waiting cars in each lane.

### Reward
`reward = 1.0 - (total_cars / max_possible_cars)` — clipped to [0.0, 1.0].  
Higher reward = fewer cars waiting = better signal control.

---

## Tasks

| Task | Cars Range | Arrivals/Step | Max Steps | Difficulty |
|------|-----------|---------------|-----------|------------|
| `easy` | 3–8 | 0–1 | 10 | Low traffic |
| `medium` | 5–15 | 0–2 | 20 | Moderate traffic |
| `hard` | 10–20 | 0–3 | 30 | High traffic |

---

## API

```python
from environment import TrafficSignalEnv

env = TrafficSignalEnv(task="easy")  # "easy", "medium", or "hard"

state = env.reset()        # → (lane1_cars, lane2_cars)
state = env.state()        # → current state
state, reward, done = env.step(action)  # action: 0 or 1
score = env.grade()        # → float in [0.0, 1.0]
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Start the API server
python app.py

# Run baseline inference (requires HF_TOKEN)
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset?task=easy` | POST | Reset environment |
| `/step?task=easy&action=0` | POST | Take a step |
| `/state?task=easy` | GET | Get current state |

---

## Project Structure

```
traffic-signal-env/
├── environment.py     # Core OpenEnv environment
├── inference.py       # Baseline LLM agent script
├── app.py             # FastAPI server for HF Space
├── openenv.yaml       # Environment metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Inference Log Format

Logs follow the required `[START]` / `[STEP]` / `[END]` structured format:

```json
{"type": "START", "task": "easy", "initial_state": [6, 4]}
{"type": "STEP", "step": 1, "action": 0, "state": [3, 5], "reward": 0.8125, "done": false}
{"type": "END", "task": "easy", "score": 0.85, "total_reward": 7.4}
```

---

## Scoring

Each task is graded independently. The final score is `grade()` output — a float between 0.0 and 1.0 based on how many cars remain when the episode ends.

---

## Infrastructure

- **Runtime:** < 20 minutes
- **Memory:** Compatible with 2 vCPU / 8GB RAM
- **Deployment:** HuggingFace Spaces (Docker)
