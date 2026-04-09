from collections import deque, Counter, defaultdict
from config import WINDOW_SIZE, DRIFT_WINDOW, BET_AMOUNTS
from datetime import datetime
import os

history = deque(maxlen=WINDOW_SIZE)
prediction_history = deque(maxlen=20)
training_history = deque(maxlen=500)
pattern_memory = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
recent_results = deque(maxlen=DRIFT_WINDOW)

prediction_next = []
logs = deque(maxlen=2000)
profit_history = deque(maxlen=100)
current_bets = BET_AMOUNTS.copy()

prediction_counter = Counter()
stuck_counter = Counter()
bait_counter = Counter()

HISTORY_RESULTS = deque(maxlen=30)
HISTORY_OUTCOMES = deque(maxlen=30)

consecutive_misses = 0
drift_detected = False
boost_training = False

stats = {
    "total_called": 0,
    "total_correct": 0,
    "total_profit": 0,
    "class_correct": Counter(),
    "class_called": Counter(),
    "drift_alert": False,
}

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    logs.appendleft(line)
    with open(os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

def update_history(result: str):
    HISTORY_RESULTS.appendleft(result)

def update_outcome(is_correct):
    HISTORY_OUTCOMES.appendleft(is_correct)

def reset_runtime_state():
    global consecutive_misses, drift_detected

    history.clear()
    prediction_history.clear()
    training_history.clear()
    recent_results.clear()

    prediction_next.clear()
    logs.clear()
    profit_history.clear()

    prediction_counter.clear()
    stuck_counter.clear()
    bait_counter.clear()

    HISTORY_RESULTS.clear()
    HISTORY_OUTCOMES.clear()

    current_bets.clear()
    current_bets.update(BET_AMOUNTS)

    consecutive_misses = 0
    drift_detected = False

    stats.clear()
    stats.update({
        "total_called": 0,
        "total_correct": 0,
        "total_profit": 0,
        "class_correct": Counter(),
        "class_called": Counter(),
        "drift_alert": False,
    })