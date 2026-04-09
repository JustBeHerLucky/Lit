# ===============================================
# ✅ REALTIME AI API - ULTIMATE CLEAN VERSION ✅
# ===============================================

import os
import time
import joblib
import threading
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from collections import deque, defaultdict, Counter
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from river import drift as river_drift
import random
import asyncio
from fastapi.staticfiles import StaticFiles
import json


model = None
model_secondary = None
model_classes_initialized = False  # Flag cho biết model chính đã init chưa
secondary_classes_initialized = False  # Flag cho biết model phụ đã init chưa
# ==== FEATURE FLAGS ====
FEATURE_FLAGS = {
    "USE_MAIN_MODEL": True,              # ✅ Dùng mô hình chính (SGDClassifier) để huấn luyện và dự đoán
    "USE_SECONDARY_MODEL": False,         # ✅ Dùng mô hình phụ để ensemble cùng mô hình chính
    "USE_PATTERN_MEMORY": False,          # ✅ Tăng xác suất dựa vào pattern chuỗi gần đây
    "USE_DRIFT_DETECTION": False,         # ✅ Phát hiện drift dữ liệu với ADWIN, PH, DDM
    "USE_VOTING_AGENT": True,            # ✅ Sử dụng VotingAgent để vote xác suất dựa theo chuỗi lịch sử
    "USE_FREQUENCY_ESTIMATOR": True,    # ✅ Dùng FrequencyEstimator tính xác suất theo tần suất gần nhất
    "USE_RL_AGENT": False,                # ✅ Dùng Reinforcement Learning để điều chỉnh chiến lược
    "USE_BOOST_TRAINING": False,         # ✅ Boost lại training nếu thua liên tiếp (xóa pattern và lịch sử)
    "USE_BAIT_STUCK_PENALTY": True      # ✅ Giảm xác suất các ô bị "mồi nhử" hoặc "bị kẹt"
}

# ==== CONFIGURATION ====
# 🔧 Đường dẫn lưu mô hình và dữ liệu
MODEL_PATH = "model/betting_model.pkl"                    # Đường dẫn lưu mô hình chính
PATTERN_MEMORY_PATH = "model/pattern_memory.pkl"          # Lưu pattern memory (ghi nhớ chuỗi)
RL_AGENT_PATH = "model/rl_agent.pkl"                      # Lưu trạng thái Q-table của RL Agent
DATA_DIR = "data"                                         # Thư mục dữ liệu
LOG_DIR = "logs"                                          # Thư mục log

# 📏 Cấu hình độ dài và khung thời gian
WINDOW_SIZE = 6                                            # Số lượng kết quả gần nhất dùng để dự đoán
PATTERN_LENGTH = 2                                         # Độ dài pattern cần ghi nhớ
FREQUENCY_WINDOW = 50                                      # Cửa sổ đếm tần suất gần đây
DRIFT_WINDOW = 50                                          # Cửa sổ phát hiện drift

# 🧠 Các lớp và nhóm lớp trong trò chơi
CLASSES = ["Kem", "Phao", "Bong", "Dam", "Gau", "Tien", "Anh", "Niem"]
MAIN_CLASSES = {"Kem", "Phao", "Bong", "Dam"}              # Các ô chính (thường đặt tiền cao hơn)

# 💰 Tỉ lệ trả thưởng (hệ số nhân)
PAYOUTS = {
    "Kem": 5, "Phao": 5, "Bong": 5, "Dam": 5,
    "Gau": 10, "Tien": 15, "Anh": 25, "Niem": 45
}

# 💸 Số tiền đặt mặc định mỗi ô
BET_AMOUNTS = {cls: 1000 for cls in CLASSES}

# ⚖️ Các hệ số điều chỉnh xác suất
PATTERN_DECAY = 0.90                                       # Tỉ lệ giảm giá trị pattern theo thời gian
STUCK_THRESHOLD = 3                                        # Số lần lặp lại liên tiếp coi là stuck
BAIT_THRESHOLD = 4                                         # Số lần xuất hiện sai coi là bait
MAX_MARTINGALE = 0                                         # Gấp thếp tối đa bao nhiêu lần khi thua

# ==== GLOBAL STATE ====
# 🌐 Trạng thái dùng chung toàn hệ thống (RAM cache)
app = FastAPI()                                                                 # Ứng dụng FastAPI chính
clients = []                                                                    # Danh sách các client đang kết nối SSE (event-stream)
drift_detected = False                                                          # Biến đánh dấu có drift dữ liệu hay không
history = deque(maxlen=WINDOW_SIZE)                                             # Lịch sử kết quả gần nhất (đầu vào cho AI)
prediction_history = deque(maxlen=3)                                            # Dự đoán gần nhất (để kiểm tra đúng/sai)
training_history = deque(maxlen=200)                                            # Dữ liệu đã học gần nhất (dùng cho huấn luyện mô hình)
pattern_memory = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))             # Ghi nhớ pattern (chuỗi) → tăng xác suất
recent_results = deque(maxlen=DRIFT_WINDOW)                                     # Kết quả gần đây để phát hiện drift
prediction_next = []                                                            # Kết quả dự đoán tiếp theo
logs = deque(maxlen=2000)                                                       # Lưu log nội bộ
profit_history = deque(maxlen=100)                                              # Lịch sử lợi nhuận
current_bets = BET_AMOUNTS.copy()                                               # Cược hiện tại đang áp dụng
prediction_counter = Counter()                                                  # Đếm số lần mỗi ô được dự đoán
stuck_counter = Counter()                                                       # Đếm số lần liên tiếp xuất hiện (để phạt stuck)
bait_counter = Counter()                                                        # Đếm số lần bị đoán sai (bait)
HISTORY_RESULTS = deque(maxlen=30)                                              # Kết quả các vòng gần nhất (hiển thị frontend)
consecutive_losses = 0                                                          # Số vòng thua liên tiếp
boost_training = False                                                          # Flag nội bộ kích hoạt boosting nếu cần

app.mount("/static", StaticFiles(directory="static"), name="static")

stats = {"total_called": 0, "total_correct": 0, "total_profit": 0, "class_correct": Counter(), "class_called": Counter(), "drift_alert": False}

os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

encoder = LabelEncoder()
encoder.fit(CLASSES)

def pattern_inner_default():
    return [0.0, 0]

def pattern_outer_default():
    return defaultdict(pattern_inner_default)

# ==== UTILITIES ====
def log(msg):
    logs.appendleft(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    with open(os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

def softmax(x, temp=0.7):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum()

def notify_clients(data):
    for queue in clients:
        queue.put_nowait(data)

def update_history(result):
    HISTORY_RESULTS.appendleft(result)


# ==== AI MODULES ====
class SimpleDDM:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 1
        self.error_sum = 0
        self.min_error = float('inf')
        self.min_std = float('inf')
        self.drift_detected = False

    def update(self, is_correct):
        error = 1 - is_correct  # DDM đo lỗi
        self.error_sum += error
        p = self.error_sum / self.n
        s = (p * (1 - p) / self.n) ** 0.5

        if p + s < self.min_error + self.min_std:
            self.min_error = p
            self.min_std = s

        if p + s > self.min_error + 2 * self.min_std:
            self.drift_detected = True
        else:
            self.drift_detected = False

        self.n += 1
        return self.drift_detected

class FrequencyEstimator:
    def __init__(self):
        self.memory = deque(maxlen=200)
        self.counter = Counter()
    def update(self, result):
        self.memory.append(result)
        self.counter[result] += 1
    def get_probabilities(self):
        total = len(self.memory)
        return {cls: self.counter[cls] / total if total else 1/len(CLASSES) for cls in CLASSES}

class VotingAgent:
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss')
        self.scaler = StandardScaler()
        self.memory = deque(maxlen=200)
        self.classes_seen = False
    def update(self, result):
        self.memory.append(CLASSES.index(result))
        if len(self.memory) >= 5:
            X = self.scaler.fit_transform([list(self.memory)[i:i+3] for i in range(len(self.memory)-3)])
            y = list(self.memory)[3:]
            if not self.classes_seen:
                self.model.partial_fit(X, y, classes=np.arange(len(CLASSES)))
                self.classes_seen = True
            else:
                self.model.partial_fit(X, y)
    def predict_proba(self, X):
        if not self.classes_seen:
            return np.ones(len(CLASSES)) / len(CLASSES)
        return self.model.predict_proba(self.scaler.transform(X))[0]

RL_ACTIONS = ["Strong", "Normal", "Safe", "Invert"]
def default_q(): return np.zeros(len(RL_ACTIONS))

class RLAgent:
    def __init__(self):
        self.q_table = defaultdict(default_q)
        self.last_state, self.last_action = None, None
    def step(self, conf, drift, streak, reward=None):
        loss_level = min(streak, 3)
        low_conf = int(conf < 0.6)
        drift_flag = int(drift)
        state = (low_conf, drift_flag, loss_level)
        if reward is not None:
            self.q_table[self.last_state][self.last_action] += 0.1 * (reward + 0.9 * np.max(self.q_table[state]) - self.q_table[self.last_state][self.last_action])
        action = random.randint(0, len(RL_ACTIONS)-1) if random.random() < 0.1 else np.argmax(self.q_table[state])
        self.last_state, self.last_action = state, action
        return RL_ACTIONS[action]

ph_drift = river_drift.PageHinkley()
adwin_drift = river_drift.ADWIN()
simple_ddm = SimpleDDM()

# ==== MODEL MANAGEMENT ====
def save_models():
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model_secondary, MODEL_PATH + ".secondary")
    joblib.dump(dict(pattern_memory), PATTERN_MEMORY_PATH)
    joblib.dump(dict(rl_agent.q_table), RL_AGENT_PATH)
    log("[SAVE] 📦 Models and states saved.")

def load_models():
    global model, model_secondary, pattern_memory
    global model_classes_initialized, secondary_classes_initialized
    INIT_FEATURES = WINDOW_SIZE + len(CLASSES)

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = SGDClassifier(loss="log_loss", warm_start=True)
        model.partial_fit([[0]*INIT_FEATURES], [0], classes=np.arange(len(CLASSES)))  # ✅ first call with classes

    if os.path.exists(MODEL_PATH + ".secondary"):
        model_secondary = joblib.load(MODEL_PATH + ".secondary")
    else:
        model_secondary = SGDClassifier(loss="log_loss", warm_start=True)
        model_secondary.partial_fit([[0]*INIT_FEATURES], [0], classes=np.arange(len(CLASSES)))  # ✅ fix here

    if os.path.exists(PATTERN_MEMORY_PATH):
        pattern_memory = defaultdict(pattern_outer_default, joblib.load(PATTERN_MEMORY_PATH))

    if os.path.exists(RL_AGENT_PATH):
        rl_agent.q_table = defaultdict(default_q, joblib.load(RL_AGENT_PATH))
    model_classes_initialized = True  # Flag cho biết model chính đã init chưa
    secondary_classes_initialized = True  # Flag cho biết model phụ đã init chưa


# ==== PREDICTION HELPERS ====
# (Phần tiếp theo ở đây - PREDICTION HELPERS)

def adjust_probs(combined_probs, pattern_memory, stuck_counter, bait_counter, current_round):
    if FEATURE_FLAGS["USE_PATTERN_MEMORY"] and len(history) >= PATTERN_LENGTH:
        pattern = tuple(history)[-PATTERN_LENGTH:]
        total_pattern = sum([score for score, _ in pattern_memory[pattern].values()])
        if total_pattern > 0:
            for cls, (score, last_seen) in pattern_memory[pattern].items():
                age = current_round - last_seen
                decayed = score * (PATTERN_DECAY ** age)
                winrate = stats["class_correct"][cls] / stats["class_called"][cls] if stats["class_called"][cls] else 0.5
                boost = 0.3 + (winrate - 0.5) * 0.5
                combined_probs[CLASSES.index(cls)] += (decayed / total_pattern) * boost

    if FEATURE_FLAGS["USE_BAIT_STUCK_PENALTY"]:
        for idx, cls in enumerate(CLASSES):
            if stuck_counter[cls] >= STUCK_THRESHOLD:
                combined_probs[idx] *= 0.5
            if bait_counter[cls] >= BAIT_THRESHOLD:
                combined_probs[idx] += 0.05

    combined_probs /= combined_probs.sum()
    return combined_probs

# ==== API INPUT SCHEMA ====
class PredictInput(BaseModel):
    result: str
    main_count: int
    rare_count: int

# ==== API ROUTES ====
@app.post("/predict_and_train")
def predict_and_train(data: PredictInput):
    global current_bets, drift_detected, consecutive_losses
    global model_classes_initialized, secondary_classes_initialized
    global model, model_secondary
    result = data.result
    if prediction_history:
        predicted = prediction_history.popleft()
        stats['total_called'] += 1
        for pred in predicted:
            stats['class_called'][pred] += 1

        payout = sum(current_bets[o] * PAYOUTS[o] for o in predicted if o == result)
        total_bet = sum(current_bets[o] for o in predicted)
        profit = payout - total_bet
        stats['total_profit'] += profit
        profit_history.append(profit)
        recent_results.append(result in predicted)

        if result in predicted:
            stats['total_correct'] += 1
            stats['class_correct'][result] += 1
            current_bets.update(BET_AMOUNTS)
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            log(f"[LOSS] Consecutive losses: {consecutive_losses}")
            if consecutive_losses == 3 and FEATURE_FLAGS["USE_BOOST_TRAINING"]:
                log("⚠️ [ADAPTIVE] 3 consecutive losses → Reset pattern memory & boosting weights.")
                pattern_memory.clear()
                training_history.clear()
            if MAX_MARTINGALE > 0 and consecutive_losses <= MAX_MARTINGALE:
                for o in CLASSES:
                    current_bets[o] *= 2

        if FEATURE_FLAGS["USE_DRIFT_DETECTION"]:
            drift_detected = (
                adwin_drift.update(int(result in predicted)) or
                (ph_drift.update(int(result in predicted)) is not None) or
                simple_ddm.update(int(result in predicted)) or
                (consecutive_losses >= 3)
            )
        else:
            drift_detected = False

    if len(history) == WINDOW_SIZE:
        counts = [list(history)[-FREQUENCY_WINDOW:].count(cls) / FREQUENCY_WINDOW for cls in CLASSES]
        X_sample = list(encoder.transform(list(history))) + counts
        y = encoder.transform([result])[0]
        training_history.append((X_sample, y))
        X, y_hist = zip(*list(training_history)[-100:])
        INIT_FEATURES = WINDOW_SIZE + len(CLASSES)
        X = [list(xi)[:INIT_FEATURES] for xi in X]
        decay_factor = 0.95
        weights = [decay_factor ** (len(X) - i - 1) for i in range(len(X))]

        if FEATURE_FLAGS["USE_MAIN_MODEL"]:
            if not model_classes_initialized:
                model.partial_fit(X, y_hist, classes=np.arange(len(CLASSES)), sample_weight=weights)
                globals()['model_classes_initialized'] = True
            else:
                model.partial_fit(X, y_hist, sample_weight=weights)

        if FEATURE_FLAGS["USE_SECONDARY_MODEL"]:
            if not secondary_classes_initialized:
                model_secondary.partial_fit(X, y_hist, classes=np.arange(len(CLASSES)), sample_weight=weights)
                globals()['secondary_classes_initialized'] = True
            else:
                model_secondary.partial_fit(X, y_hist, sample_weight=weights)

        model_probs = np.zeros(len(CLASSES))
        if FEATURE_FLAGS["USE_MAIN_MODEL"] and FEATURE_FLAGS["USE_SECONDARY_MODEL"]:
            scores = (model.decision_function([X_sample])[0] + model_secondary.decision_function([X_sample])[0]) / 2
            model_probs = softmax(scores, 0.7)

        freq_probs = np.ones(len(CLASSES)) / len(CLASSES)
        if FEATURE_FLAGS["USE_FREQUENCY_ESTIMATOR"]:
            freq_probs = np.array([frequency_estimator.get_probabilities()[c] for c in CLASSES])

        voting_probs = np.ones(len(CLASSES)) / len(CLASSES)
        if FEATURE_FLAGS["USE_VOTING_AGENT"]:
            voting_input = [[CLASSES.index(x) for x in list(history)[-3:]]] if len(history) >= 3 else [[0, 0, 0]]
            voting_probs = voting_agent.predict_proba(voting_input)

        combined_probs = 0.2 * freq_probs + 0.3 * voting_probs + 0.5 * model_probs

        if FEATURE_FLAGS["USE_PATTERN_MEMORY"] or FEATURE_FLAGS["USE_BAIT_STUCK_PENALTY"]:
            combined_probs = adjust_probs(combined_probs, pattern_memory, stuck_counter, bait_counter, len(training_history))

        avg_conf = np.mean(combined_probs)
        rl_action = rl_agent.step(avg_conf, stats['drift_alert'], consecutive_losses) if FEATURE_FLAGS["USE_RL_AGENT"] else "Normal"

        sorted_idx = np.argsort(-combined_probs) if rl_action != "Invert" else np.argsort(combined_probs)
        predict_order = [CLASSES[idx] for idx in sorted_idx]

        if drift_detected:
            log("🌪️ [DRIFT] Detected → Reset pattern + training history.")
            pattern_memory.clear()
            training_history.clear()
            stats['drift_alert'] = True
            if rl_action == "Invert":
                log("🧠 [RLAgent] Invert + Drift → RESET FULL MODEL + RLAgent")
                globals()['model_classes_initialized'] = False
                globals()['secondary_classes_initialized'] = False
                model = SGDClassifier(loss="log_loss", warm_start=True)
                model.partial_fit([[0]*(WINDOW_SIZE + len(CLASSES))], [0], classes=np.arange(len(CLASSES)))
                model_secondary = SGDClassifier(loss="log_loss", warm_start=True)
                model_secondary.partial_fit([[0]*(WINDOW_SIZE + len(CLASSES))], [0], classes=np.arange(len(CLASSES)))
                rl_agent.q_table = defaultdict(default_q)
        else:
            stats['drift_alert'] = False

        main = [p for p in predict_order if p in MAIN_CLASSES][:data.main_count]
        rare = [p for p in predict_order if p not in MAIN_CLASSES][:data.rare_count]

        prediction_next.clear()
        prediction_next.extend(main + rare)
        for p in main + rare:
            prediction_counter[p] += 1

        prediction_history.append(list(main) + list(rare))
        log(f"New prediction → Main: {main} | Rare: {rare} | Strategy: {rl_action}")
        response = {"main_predictions": main, "rare_predictions": rare}
    else:
        response = {"message": "Not enough data."}

    bait_counter[result] = 0
    for cls in CLASSES:
        if cls != result:
            bait_counter[cls] += 1

    history.append(result)
    update_history(result)
    notify_clients("update")
    return response

def get_status():
    total_called = stats["total_called"]
    total_correct = stats["total_correct"]
    winrate = round(total_correct / total_called, 4) if total_called else 0.0

    return JSONResponse(
        content={
            "total_called": total_called,
            "total_correct": total_correct,
            "total_profit": stats["total_profit"],
            "winrate": winrate,
            "drift_alert": stats["drift_alert"],
            "class_correct": dict(stats["class_correct"]),
            "class_called": dict(stats["class_called"]),
            "profit_history": list(profit_history),
            "logs": list(logs),
            "history_results": list(HISTORY_RESULTS),
            "last_prediction": list(prediction_next)
        },
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


class UpdateBetAmountsInput(BaseModel):
    bet_amounts: dict

@app.post("/update_bet_amounts")
def update_bet_amounts(bets: UpdateBetAmountsInput):
    BET_AMOUNTS.update(bets.bet_amounts)
    current_bets.update(BET_AMOUNTS)

    # Reset toàn bộ dữ liệu mô phỏng về ban đầu
    stats.update({
        "total_called": 0,
        "total_correct": 0,
        "total_profit": 0,
        "class_correct": Counter(),
        "class_called": Counter(),
        "drift_alert": False
    })

    log("⚡ BET_AMOUNTS updated.")

    return {"message": "Bet amounts updated."}

@app.post("/reset")
def reset():
    
    profit_history.clear()
    logs.clear()
    HISTORY_RESULTS.clear()
    prediction_history.clear()
    prediction_next.clear()
    recent_results.clear()
    training_history.clear()
    pattern_memory.clear()

    log("⚡ system reset.")

    return {"message": "system reset."}


@app.get("/events")
def events():
    async def event_generator():
        queue = asyncio.Queue()
        clients.append(queue)
        try:
            while True:
                total_called = stats["total_called"]
                total_correct = stats["total_correct"]
                winrate = round(total_correct / total_called, 4) if total_called else 0.0

                data = {
                    "total_called": total_called,
                    "total_correct": total_correct,
                    "total_profit": stats["total_profit"],
                    "winrate": winrate,
                    "drift_alert": stats["drift_alert"],
                    "class_correct": dict(stats["class_correct"]),
                    "class_called": dict(stats["class_called"]),
                    "profit_history": list(profit_history),
                    "logs": list(logs),
                    "history_results": list(HISTORY_RESULTS),
                    "last_prediction": list(prediction_next)
                }
                await queue.put(json.dumps(data))
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(1)
        finally:
            clients.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/status/json")
def status_json():
    return get_status()


# ==== START AUTO SAVE THREAD ====
@app.on_event("startup")
async def start_background_tasks():
    async def autosave_loop():
        while True:
            save_models()
            await asyncio.sleep(60)

    asyncio.create_task(autosave_loop())

rl_agent = RLAgent()
load_models()
frequency_estimator = FrequencyEstimator()
voting_agent = VotingAgent()

log("✅ System Ready.")
