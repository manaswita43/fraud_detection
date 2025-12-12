# MLOPs Oppe 2

## Project structure
```
fraud-detection/
├─ data/
│  ├─ v0/
│  └─ v1/
├─ app/
│  ├─ main.py
│  ├─ model.pkl
│  ├─ requirements.txt
│  └─ Dockerfile
├─ k8s/
│  ├─ deployment.yaml
│  ├─ service.yaml
│  └─ hpa.yaml
├─ locust/
│  └─ locustfile.py
├─ scripts/
│  ├─ split_data.py
│  ├─ poison_data.py
│  ├─ add_location.py
│  └─ train_and_log.py
├─ .github/
│  └─ workflows/
│     └─ cd.yml
└─ README.md
```

## Commands
### Set project and region
```bash
gcloud config set project mlops-473405
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-c
```

### Enable required APIs
```bash
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  ml.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudtrace.googleapis.com
```

### Create GCS bucket
```bash
gsutil mb -l us-central1 gs://mlops-oppe2
```

---

## 1. Data preparation (split into v0 and v1)
**Refer scripts/split_data.py**

```bash
# place transactions.csv in gcp bucket as - mlops-oppe2/data/transactions.csv:
python3 scripts/split_data.py
ls -R data
```

---

## 2. Train a baseline model (and save as model.pkl)
**Refer scripts/train_and_log.py**
Store requirements.txt under app folder

```bash
pip install -r app/requirements.txt
# run mlflow
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root gs://mlops-oppe2/mlflow-artifacts \
  --host 0.0.0.0 --port 8100 --allowed-hosts '*' --cors-allowed-origins '*'

python3 scripts/train_and_log.py

ls app/model.pkl
```

---

## 3. FastAPI app with OpenTelemetry custom span & /predict endpoint
**Refer app/main.py**

---

## 4. Dockerfile (containerize model + API)
**Refer app/Dockerfile**

---

## 5. Build and push Docker image to Artifact Registry (gcloud commands)
```bash
# create artifact registry repo (us-central1)
gcloud artifacts repositories create aml-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker repo for fraud detector"

# configure docker auth (Cloud Build can also push)
gcloud auth configure-docker us-central1-docker.pkg.dev

# build and tag
cd app
docker build -t us-central1-docker.pkg.dev/mlops-473405/aml-repo/fraud-detector:v1 .

# push (if local docker has permission)
docker push us-central1-docker.pkg.dev/mlops-473405/aml-repo/fraud-detector:v1
```

---

## 6. GitHub Actions CI/CD + CML snippet
**Refer .github/workflows/cd.yml**

Notes:
- Add GCP_SA_KEY secret with your service account JSON and PROJECT_ID.
- The action builds and pushes image tagged with commit SHA and latest.

---

## 7. Deploy to GKE (cluster + manifests)
```bash
# create GKE cluster (autopilot or standard; we'll use standard with 3 nodes)
gcloud container clusters create fraud-gke-cluster \
  --zone us-central1-c \
  --num-nodes 3 \
  --machine-type=e2-standard-4 \
  --project mlops-473405

# get credentials
gcloud container clusters get-credentials fraud-gke-cluster --zone us-central1-c
```
**Refer k8s/deployment.yaml, k8s/service.yaml**


Each time a GKE cluster is created, do these 3 steps:
```bash
sudo apt-get install 

google-cloud-cli-gke-gcloud-auth-plugin

kubectl create serviceaccount telemetry-access --namespace default

kubectl annotate serviceaccount telemetry-access \
  --namespace default \
  iam.gke.io/gcp-service-account=telemetry-access@mlops-473405.iam.gserviceaccount.com
```

To confirm the telemetry-access serviceaccount exists,
```bash
kubectl get serviceaccount telemetry-access -n default

kubectl describe serviceaccount telemetry-access -n default - check for annotation exists

gcloud iam service-accounts get-iam-policy telemetry-access@mlops-473405.iam.gserviceaccount.com - check IAM policy includes it
```



Apply manifests:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc fraud-detector-lb -w  # wait for external IP
```

---

## 8. HorizontalPodAutoscaler (HPA)
**Refer k8s/hpa.yaml**

Apply:
```bash
kubectl apply -f k8s/hpa.yaml
kubectl get hpa -w
```
Alternative (imperative):
```bash
kubectl autoscale deployment fraud-detector --cpu-percent=50 --min=2 --max=10
```

---

## 9. Load testing with Locust
**Refer locust/locustfile.py**

Run Locust from a client machine (or Cloud Shell port-forward):
```bash
# run locust pointing to the LoadBalancer IP:
LOCUST_HOST="http://<EXTERNAL_IP>"
locust -f locustfile.py --host=${LOCUST_HOST} --users 50 --spawn-rate 5 --run-time 5m
```

Watch HPA scale:
```bash
kubectl get hpa -w
kubectl get pods -o wide
```

---

## 10. Observability: OpenTelemetry custom span
view logs:
```bash
kubectl logs -l app=fraud-detector -c fraud-detector --tail=200
# or stream
kubectl logs -l app=fraud-detector -c fraud-detector -f
```

Open logging in console to view logs

---

## 11. Data poisoning simulation script
**Refer scripts/poison_data.py**

Create three poisoned files:
```bash
python3 scripts/poison_data.py --percent 2
python3 scripts/poison_data.py --percent 8
python3 scripts/poison_data.py --percent 20
ls data/v0 | grep poisoned
```

---

## 12. Version poisoned datasets with DVC and push to GCS remote
```bash
dvc init
dvc remote add -d gcsremote gs://mlops-oppe2/mlops-473405-dvc
# If not already set up, give dvc access to GCS via gcloud credentials:
gcloud auth application-default login
```
Track files
```bash
dvc add data/v0/poisoned_2_percent.csv
dvc add data/v0/poisoned_8_percent.csv
dvc add data/v0/poisoned_20_percent.csv
dvc add data/v0/transactions_2022.csv
dvc add data/v1/transactions_2023.csv

git add data/*.dvc .gitignore
git commit -m "dvc track poisoned datasets and v0/v1"
dvc push  # pushes to gs://mlops-oppe2/mlops-473405-dvc
```

---

## 13. Train experiments with MLflow for poisoned datasets
```bash
python3 scripts/train_and_log.py --data data/v0/poisoned_2_percent.csv --run-name poisoned_2

python3 scripts/train_and_log.py --data data/v0/poisoned_8_percent.csv --run-name poisoned_8

python3 scripts/train_and_log.py --data data/v0/poisoned_20_percent.csv --run-name poisoned_20
```

---

## 14. Explainability: Add synthetic location attribute, train final model, SHAP
**Refer scripts/add_location.py**

```bash
python3 scripts/add_location.py
```
**Refer scripts/train_shap_fair.py**

```bash
python3 scripts/train_shap_fair.py
# check mlflow UI for artifacts (shap_summary.png) and metrics
```
(Please check for textual report in shap)
(Check fairness report as well)

---

## 15. Detect concept drift: evaluate v1 using model trained on v0
**Refer scripts/drift_check.py**

(check drift file, should use evidently)

Run:
```bash
python3 scripts/drift_check.py --model app/model_with_location.pkl
```

---

## Action Plans
- Please check for textual report in shap
- Check fairness report as well
- Check drift file, should use evidently
- Write test cases
- Write CI file
