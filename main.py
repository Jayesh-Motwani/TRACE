from fastapi import FastAPI
import compute
from pydantic import BaseModel
from typing import Any, Dict
app = FastAPI()


class RowIn(BaseModel):
    row: Dict[str, Any]


@app.post("/analyze-batch")
def analyze_batch(req: RowIn):
    out = compute.predict(req.rows)
    return out


@app.post("/predict")
def predict_one(req: RowIn):
    out = compute.predict([req.row])
    result0 = out["results"][0]
    return {
        "is_anomaly": result0["is_anomaly"],
        "score": result0["score"],
        "threshold": result0["threshold"],
        "attack_type": result0["attack_type"],
        "attack_confidence": result0["attack_confidence"],
        "top_contributors": result0["top_contributors"],
        "input_issues": out.get("input_issues", {})
    }


@app.post("/attack-type")
def predict_attack_type(req: RowIn):
    out = compute.predict([req.row])
    r = out["results"][0]
    return {"attack_type": r["attack_type"], "attack_confidence": r["attack_confidence"]}


@app.post("/anomaly-flag")
def get_anomaly_flag(req: RowIn):
    out = compute.predict([req.row])
    r = out["results"][0]
    return {
        "is_anomaly": r["is_anomaly"],
        "score": r["score"],
        "threshold": r["threshold"],
    }


@app.post("/analyze")
def analyze(req: RowIn):
    out = compute.predict([req.row])
    r = out["results"][0]

    llm_payload = {
        "is_anomaly": r["is_anomaly"],
        "score": r["score"],
        "threshold": r["threshold"],
        "attack_type": r["attack_type"],
        "attack_confidence": r["attack_confidence"],
        "top_contributors": r["top_contributors"],
        "input_issues": out.get("input_issues", {}),
    }

    return {
        "model_output": r,
        "llm_payload": llm_payload
    }
