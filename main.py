import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import model_manager
import state
import logic_predict
from agents.drift import reset_drift_detectors
from logic_predict import PredictInput, predict_and_train, reset_runtime_modules

app = FastAPI(title="Realtime Sequence Prediction API")

os.makedirs("model", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

clients = []


class ConfigInput(BaseModel):
    main_count: int | None = None
    rare_count: int | None = None


@app.on_event("startup")
async def startup_event():
    model_manager.load_models(logic_predict.rl_agent)
    state.log("✅ System Ready.")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        model_manager.save_models()
    except Exception as e:
        state.log(f"[SAVE-ERROR] {e}")


def build_status_payload():
    total_called = state.stats["total_called"]
    total_correct = state.stats["total_correct"]
    winrate = round((total_correct / total_called) * 100, 2) if total_called else 0.0

    return {
        "total_called": total_called,
        "total_correct": total_correct,
        "total_profit": state.stats["total_profit"],
        "winrate": winrate,
        "drift_alert": state.stats["drift_alert"],
        "class_correct": dict(state.stats["class_correct"]),
        "class_called": dict(state.stats["class_called"]),
        "profit_history": list(state.profit_history),
        "logs": list(state.logs),
        "history_results": list(state.HISTORY_RESULTS),
        "history_outcomes": list(state.HISTORY_OUTCOMES),
        "last_prediction": list(state.prediction_next),
    }



async def broadcast_update():
    if not clients:
        return

    payload = json.dumps(build_status_payload())
    dead = []

    for q in clients:
        try:
            await q.put(payload)
        except Exception:
            dead.append(q)

    for q in dead:
        if q in clients:
            clients.remove(q)


@app.post("/predict_and_train")
async def predict_and_train_route(data: PredictInput):
    result = predict_and_train(data)
    await broadcast_update()
    return result


@app.post("/reset")
async def reset():
    state.reset_runtime_state()
    reset_runtime_modules()
    reset_drift_detectors()
    state.log("⚡ System reset.")
    await broadcast_update()
    return {"message": "system reset"}


@app.get("/status/json")
def status_json():
    return JSONResponse(
        content=build_status_payload(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/events")
async def events():
    async def event_generator():
        queue = asyncio.Queue()
        clients.append(queue)

        try:
            # gửi snapshot đầu tiên ngay lúc connect
            await queue.put(json.dumps(build_status_payload()))

            while True:
                data = await queue.get()
                yield f"data: {data}\n\n"
        finally:
            if queue in clients:
                clients.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")




@app.get("/")
def root():
    return {
        "message": "Realtime Sequence Prediction API is running.",
        "dashboard": "/static/dashboard.html" if os.path.isdir("static") else None,
    }
