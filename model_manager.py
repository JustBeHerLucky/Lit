import os
import joblib
import numpy as np
from collections import defaultdict
from config import MODEL_PATH, PATTERN_MEMORY_PATH, RL_AGENT_PATH, CLASSES
from agents.pattern import pattern_outer_default
from agents.rl import default_q

model = None
model_secondary = None
rl_agent = None
pattern_memory = None

# 25 = 8 encoded history + 8 frequency + 1 streak + 8 recency
EXPECTED_FEATURES = 25


def _new_model():
    from sklearn.linear_model import SGDClassifier
    m = SGDClassifier(loss="log_loss", warm_start=True)
    m.partial_fit([[0] * EXPECTED_FEATURES], [0], classes=np.arange(len(CLASSES)))
    return m


def _load_or_rebuild_model(path: str):
    if os.path.exists(path):
        try:
            m = joblib.load(path)
            if getattr(m, "n_features_in_", None) != EXPECTED_FEATURES:
                print(
                    f"⚠️ Feature mismatch in {path}: "
                    f"{getattr(m, 'n_features_in_', 'unknown')} != {EXPECTED_FEATURES}. Rebuilding model."
                )
                return _new_model()
            return m
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}. Rebuilding model.")
            return _new_model()
    return _new_model()


def save_models():
    if model is not None:
        joblib.dump(model, MODEL_PATH)
    if model_secondary is not None:
        joblib.dump(model_secondary, MODEL_PATH + ".secondary")
    if pattern_memory is not None:
        joblib.dump(dict(pattern_memory), PATTERN_MEMORY_PATH)
    if rl_agent is not None:
        joblib.dump(dict(rl_agent.q_table), RL_AGENT_PATH)


def load_models(rl_instance):
    global model, model_secondary, pattern_memory, rl_agent
    rl_agent = rl_instance

    model = _load_or_rebuild_model(MODEL_PATH)
    model_secondary = _load_or_rebuild_model(MODEL_PATH + ".secondary")

    if os.path.exists(PATTERN_MEMORY_PATH):
        loaded = joblib.load(PATTERN_MEMORY_PATH)
        pattern_memory = defaultdict(pattern_outer_default, loaded)
    else:
        pattern_memory = defaultdict(pattern_outer_default)

    if os.path.exists(RL_AGENT_PATH):
        rl_agent.q_table = defaultdict(default_q, joblib.load(RL_AGENT_PATH))

    return model, model_secondary, pattern_memory, rl_agent