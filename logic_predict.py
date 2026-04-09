import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

import state
import model_manager

from config import (
    FEATURE_FLAGS,
    CLASSES,
    MAIN_CLASSES,
    WINDOW_SIZE,
    FREQUENCY_WINDOW,
    PATTERN_LENGTH,
)
from agents.pattern import adjust_probs, pattern_memory
from agents.voting import VotingAgent
from agents.frequency import FrequencyEstimator
from agents.rl import RLAgent
from agents.drift import detect_drift


voting_agent = VotingAgent()
frequency_estimator = FrequencyEstimator()
rl_agent = RLAgent()

encoder = LabelEncoder()
encoder.fit(CLASSES)


class PredictInput(BaseModel):
    result: str
    main_count: int = 2
    rare_count: int = 1


def softmax(x, temp=0.7):
    x = np.asarray(x, dtype=float)
    e_x = np.exp((x - np.max(x)) / temp)
    denom = e_x.sum()
    return e_x / denom if denom > 0 else np.ones(len(CLASSES)) / len(CLASSES)


def _extract_features(hist):
    feats = []

    feats += list(encoder.transform(hist))

    tail = hist[-FREQUENCY_WINDOW:]
    for cls in CLASSES:
        feats.append(tail.count(cls) / max(len(tail), 1))

    last = hist[-1]
    streak = 1
    for i in range(len(hist) - 2, -1, -1):
        if hist[i] == last:
            streak += 1
        else:
            break
    feats.append(streak / len(hist))

    for cls in CLASSES:
        try:
            rev_idx = hist[::-1].index(cls)
            feats.append(rev_idx / len(hist))
        except ValueError:
            feats.append(1.0)

    return feats


def _update_pattern_memory(result: str):
    if len(state.history) >= PATTERN_LENGTH:
        pattern = tuple(list(state.history)[-PATTERN_LENGTH:])
        score, _ = pattern_memory[pattern][result]
        pattern_memory[pattern][result] = [score + 1.0, len(state.training_history)]


def _evaluate_previous_prediction(result: str):
    if not state.prediction_history:
        return

    prev = state.prediction_history.popleft()
    predicted = prev["predictions"]

    state.stats["total_called"] += 1
    for pred in predicted:
        state.stats["class_called"][pred] += 1

    is_hit = result in predicted
    state.recent_results.append(is_hit)
    state.update_outcome(is_hit)

    if is_hit:
        state.stats["total_correct"] += 1
        state.stats["class_correct"][result] += 1
        state.consecutive_misses = 0
    else:
        state.consecutive_misses += 1
        state.log(f"[MISS] Consecutive misses: {state.consecutive_misses}")


def _train_models(result: str):
    if len(state.history) < WINDOW_SIZE:
        return None

    hist_list = list(state.history)
    X_sample = _extract_features(hist_list)
    y = encoder.transform([result])[0]

    state.training_history.append((X_sample, y))
    X_hist, y_hist = zip(*list(state.training_history)[-300:])

    weights = np.linspace(0.6, 1.4, len(X_hist))

    if FEATURE_FLAGS["USE_MAIN_MODEL"]:
        model_manager.model.partial_fit(
            X_hist,
            y_hist,
            classes=np.arange(len(CLASSES)),
            sample_weight=weights,
        )

    if FEATURE_FLAGS["USE_SECONDARY_MODEL"]:
        model_manager.model_secondary.partial_fit(
            X_hist,
            y_hist,
            classes=np.arange(len(CLASSES)),
            sample_weight=weights,
        )

    return X_sample


def _get_model_probs(X_sample):
    uniform = np.ones(len(CLASSES)) / len(CLASSES)
    if X_sample is None:
        return uniform

    if FEATURE_FLAGS["USE_MAIN_MODEL"]:
        scores_main = model_manager.model.decision_function([X_sample])[0]
        probs_main = softmax(scores_main, 0.7)

        if FEATURE_FLAGS["USE_SECONDARY_MODEL"]:
            scores_secondary = model_manager.model_secondary.decision_function([X_sample])[0]
            probs_secondary = softmax(scores_secondary, 0.7)
            return (probs_main + probs_secondary) / 2

        return probs_main

    return uniform


def _get_freq_probs():
    if not FEATURE_FLAGS["USE_FREQUENCY_ESTIMATOR"]:
        return np.ones(len(CLASSES)) / len(CLASSES)

    probs = frequency_estimator.get_probabilities()
    return np.array([probs[c] for c in CLASSES], dtype=float)


def _get_voting_probs():
    if not FEATURE_FLAGS["USE_VOTING_AGENT"]:
        return np.ones(len(CLASSES)) / len(CLASSES)

    hist = list(state.history)
    if len(hist) >= 3:
        voting_input = [[CLASSES.index(x) for x in hist[-3:]]]
    else:
        voting_input = [[0, 0, 0]]

    probs = voting_agent.predict_proba(voting_input)
    return np.asarray(probs, dtype=float)


def _build_combined_probs(X_sample):
    model_probs = _get_model_probs(X_sample)
    freq_probs = _get_freq_probs()
    voting_probs = _get_voting_probs()

    combined = (
        0.60 * model_probs +
        0.20 * voting_probs +
        0.20 * freq_probs
    )

    if FEATURE_FLAGS["USE_PATTERN_MEMORY"] or FEATURE_FLAGS["USE_BAIT_STUCK_PENALTY"]:
        combined = adjust_probs(
            combined,
            pattern_memory,
            state.stuck_counter,
            state.bait_counter,
            len(state.training_history),
        )

    combined = np.asarray(combined, dtype=float)
    total = combined.sum()
    if total <= 0:
        return np.ones(len(CLASSES)) / len(CLASSES)
    return combined / total


def _update_runtime_modules(result: str):
    frequency_estimator.update(result)
    voting_agent.update(result)

    state.bait_counter[result] = 0
    for cls in CLASSES:
        if cls != result:
            state.bait_counter[cls] += 1


def predict_and_train(data: PredictInput):
    result = data.result
    if result not in CLASSES:
        return {"error": f"Invalid result. Must be one of: {CLASSES}"}

    _evaluate_previous_prediction(result)

    X_sample = _train_models(result)
    _update_pattern_memory(result)
    combined_probs = _build_combined_probs(X_sample)

    avg_conf = float(np.max(combined_probs))
    rl_action = (
        rl_agent.step(avg_conf, state.stats["drift_alert"], state.consecutive_misses)
        if FEATURE_FLAGS["USE_RL_AGENT"]
        else "Normal"
    )

    sorted_idx = np.argsort(-combined_probs) if rl_action != "Invert" else np.argsort(combined_probs)
    predict_order = [CLASSES[idx] for idx in sorted_idx]

    drift_now = FEATURE_FLAGS["USE_DRIFT_DETECTION"] and detect_drift(
        result in (state.prediction_history[0]["predictions"] if state.prediction_history else []),
        state.consecutive_misses,
    )

    if drift_now:
        state.log("🌪️ [DRIFT] Detected → clear training history.")
        state.training_history.clear()
        state.stats["drift_alert"] = True
    else:
        state.stats["drift_alert"] = False

    main = [p for p in predict_order if p in MAIN_CLASSES][:data.main_count]
    rare = [p for p in predict_order if p not in MAIN_CLASSES][:data.rare_count]
    predictions = main + rare

    state.prediction_next.clear()
    state.prediction_next.extend(predictions)

    for p in predictions:
        state.prediction_counter[p] += 1

    state.prediction_history.append({
        "predictions": predictions,
        "actual": result,
    })

    state.log(
        f"New prediction → Main: {main} | Rare: {rare} | "
        f"Top confidence: {round(float(np.max(combined_probs)), 4)} | Strategy: {rl_action}"
    )

    state.history.append(result)
    state.update_history(result)
    _update_runtime_modules(result)

    top_scores = {
        CLASSES[i]: round(float(combined_probs[i]), 4)
        for i in np.argsort(-combined_probs)[:5]
    }

    return {
        "main_predictions": main,
        "rare_predictions": rare,
        "top_scores": top_scores,
        "drift_alert": state.stats["drift_alert"],
    }