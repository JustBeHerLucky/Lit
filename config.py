FEATURE_FLAGS = {
    "USE_MAIN_MODEL": True,
    "USE_SECONDARY_MODEL": False,
    "USE_PATTERN_MEMORY": True,
    "USE_DRIFT_DETECTION": True,
    "USE_VOTING_AGENT": True,
    "USE_FREQUENCY_ESTIMATOR": True,
    "USE_RL_AGENT": False,
    "USE_BOOST_TRAINING": False,
    "USE_BAIT_STUCK_PENALTY": True
}

MODEL_PATH = "model/betting_model.pkl"
PATTERN_MEMORY_PATH = "model/pattern_memory.pkl"
RL_AGENT_PATH = "model/rl_agent.pkl"
DATA_DIR = "data"
LOG_DIR = "logs"

WINDOW_SIZE = 8
PATTERN_LENGTH = 3
FREQUENCY_WINDOW = 80
DRIFT_WINDOW = 80

CLASSES = ["Kem", "Phao", "Bong", "Dam", "Gau", "Tien", "Anh", "Niem"]
MAIN_CLASSES = {"Kem", "Phao", "Bong", "Dam"}

PAYOUTS = {
    "Kem": 5, "Phao": 5, "Bong": 5, "Dam": 5,
    "Gau": 10, "Tien": 15, "Anh": 25, "Niem": 45
}

BET_AMOUNTS = {cls: 1000 for cls in CLASSES}

PATTERN_DECAY = 0.90
STUCK_THRESHOLD = 3
BAIT_THRESHOLD = 4
MAX_MARTINGALE = 0