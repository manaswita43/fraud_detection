from locust import HttpUser, task, between
import random
import json

# prepare a dummy transaction payload according to features available
# you should inspect CSV columns to build a realistic payload; here's a generic template:
FEATURES = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Time"]

def random_transaction():
    payload = {f: float(random.gauss(0,1)) for f in FEATURES}
    payload["Amount"] = abs(random.gauss(50, 100))
    payload["Time"] = random.randint(0, 100000)
    return {"features": payload}

class FraudUser(HttpUser):
    wait_time = between(1, 2)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=random_transaction())
