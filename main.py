from fastapi import FastAPI
import compute
from pydantic import BaseModel
from typing import Any, Dict

data = [{...}]
app = FastAPI()


class RowIn(BaseModel):
    row: Dict[str, Any]


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


