# 🧠 Realtime AI Betting System (Modular Edition)

A modular, fully-configurable FastAPI application for real-time betting prediction, using online machine learning, voting agents, frequency estimators, and optional reinforcement learning.

---

## 📦 Project Structure

```
.
├── main.py               # FastAPI app with all endpoints
├── logic_predict.py      # Core prediction logic
├── config.py             # Constants, flags, payout configs
├── state.py              # Global RAM state and log handling
├── model_manager.py      # Load/save models and state
├── requirements.txt
├── agents/               # AI modules
│   ├── frequency.py      # FrequencyEstimator
│   ├── voting.py         # VotingAgent
│   ├── rl.py             # Reinforcement learning agent
│   ├── pattern.py        # Pattern memory and boost logic
│   └── drift.py          # Drift detection (ADWIN, PH, DDM)
├── model/                # Model checkpoints (auto created)
├── data/                 # Training logs or inputs (optional)
├── logs/                 # Auto-generated logs
└── static/               # Optional frontend dashboard
```

---

## 🚀 Getting Started

### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
uvicorn main:app --reload
```

### 3. API Endpoints
- `POST /predict_and_train` – main API to update result + predict next
- `POST /update_bet_amounts` – update stake levels
- `POST /reset` – reset model + stats
- `GET /status/json` – return JSON summary
- `GET /events` – Server-sent event stream for dashboard

---

## ⚙️ Feature Flags (`config.py`)
Enable/disable AI components:
```python
FEATURE_FLAGS = {
  "USE_MAIN_MODEL": True,
  "USE_SECONDARY_MODEL": False,
  "USE_PATTERN_MEMORY": False,
  "USE_DRIFT_DETECTION": False,
  "USE_VOTING_AGENT": True,
  "USE_FREQUENCY_ESTIMATOR": True,
  "USE_RL_AGENT": False,
  "USE_BOOST_TRAINING": False,
  "USE_BAIT_STUCK_PENALTY": True
}
```

---

## 🧪 Example JSON Input
```json
POST /predict_and_train
{
  "result": "Tien",
  "main_count": 3,
  "rare_count": 2
}
```

---

## 📌 Notes
- Models auto-save every 60s.
- Training uses online `partial_fit()`.
- All logic from original monolith is preserved exactly.

---

Made with ❤️ to help you win smarter.
