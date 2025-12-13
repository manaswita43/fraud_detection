from locust import HttpUser, task, between
import random

class FraudUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get("/")

    @task(3)
    def predict(self):
        payload = {
            "features": {
                "Time": random.randint(1, 100000),
                **{f"V{i}": random.random() for i in range(1, 29)},
                "Amount": random.uniform(1, 500)
            }
        }

        self.client.post("/predict", json=payload)
