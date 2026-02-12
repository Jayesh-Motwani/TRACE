POST /analyze-batch → takes all data and gives full anomaly + attack results for each row (used for initial table scan)

POST /predict → takes one req and returns all important fields

POST /attack-type → takes one req and returns only the predicted attack label and confidence

POST /anomaly-flag → takes one req and returns only anomaly flag, score, and threshold

POST /analyze → takes one req and returns raw output plus a curated payload ready to be sent to the LLM

POST /summarize → takes model output + LLM payload and returns a human-readable alert explanation in JSON
