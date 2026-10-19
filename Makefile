# ---------------------------------------------------------------------------
# Makefile for the Heart Disease Prediction MLOps assignment.
#
# The idea behind this file is to let the whole project run through a few short,
# memorable commands instead of long chains of docker / kubectl / python
# commands. It wires together every stage of the assignment - preparing the
# data, training the model, serving it as an API, monitoring it, and deploying
# it to Kubernetes.
#
# It is written to work in BOTH environments used for this assignment:
#   * a local macOS laptop, where Docker runs through Colima, and
#   * the college Prayogshala Linux lab, where Docker runs natively.
# The right container-engine setup is detected automatically from the operating
# system, so the exact same commands work in either place.
#
# Most-used commands (run `make help` to see them all):
#   make all       - first time: build the Python environment, then train and
#                    start the fully monitored API in one go
#   make demo      - everyday run: train + start API + Prometheus + Grafana
#   make k8s-demo  - deploy the API to a local Kubernetes cluster (Minikube)
#   make mlflow    - open the MLflow experiment-tracking dashboard
#   make down      - stop everything
# ---------------------------------------------------------------------------

PY  := .venv/bin/python
PIP := .venv/bin/pip
UNAME := $(shell uname -s)

# Prefer Docker Compose v2 ("docker compose"); fall back to the older
# standalone "docker-compose" (v1) if that is what the lab provides.
COMPOSE_BIN := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")
COMPOSE := $(COMPOSE_BIN) -f monitoring/docker-compose.yml

.PHONY: help setup data train test lint pipeline serve \
        docker-build docker-run docker-stop \
        k8s-deploy k8s-status k8s-delete k8s-demo k8s-down \
        monitor monitor-down mlflow clean \
        engine-up demo down all

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:  ## Download and clean the dataset
	$(PY) src/download_data.py
	$(PY) src/preprocess.py

train:  ## Train models, log to MLflow, save best model
	$(PY) src/train.py

test:  ## Run unit tests
	$(PY) -m pytest

lint:  ## Lint the code
	$(PY) -m ruff check src tests api

pipeline: data train test  ## Full ML pipeline: data -> train -> test (one command)

serve:  ## Run the API locally (dev mode) on http://127.0.0.1:8000
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000

docker-build:  ## Build the Docker image (train first so the model exists)
	$(PY) src/train.py
	docker build -t heart-disease-api:latest .

docker-run:  ## Run the container on http://127.0.0.1:8000
	docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

docker-stop:  ## Stop and remove the container
	docker rm -f heart-api

k8s-deploy:  ## Load image into Minikube and deploy
	minikube image load heart-disease-api:latest
	kubectl apply -f k8s/

k8s-status:  ## Show pods and service
	kubectl get pods -l app=heart-disease-api -o wide
	kubectl get svc heart-disease-api

k8s-delete:  ## Remove the Kubernetes deployment
	kubectl delete -f k8s/

monitor:  ## Start the FULL stack (API + Prometheus + Grafana) with one command
	$(COMPOSE) up -d --build
	@echo ""
	@echo "API docs    -> http://localhost:8000/docs"
	@echo "Prometheus  -> http://localhost:9090/targets"
	@echo "Grafana     -> http://localhost:3000 (admin/admin)"

monitor-down:  ## Stop the monitoring stack
	$(COMPOSE) down

mlflow:  ## Open the MLflow tracking UI
	.venv/bin/mlflow ui --backend-store-uri "$(PWD)/mlruns" --port 5001

# ---------------------------------------------------------------------------
# One-command automation (auto-detects macOS Colima vs Linux native Docker)
# ---------------------------------------------------------------------------

engine-up:  ## Prepare the container engine (Colima on macOS; native Docker on Linux)
ifeq ($(UNAME),Darwin)
	@echo ">> macOS detected: starting Colima 'mlops' profile..."
	colima start --profile mlops
	docker context use colima-mlops
	@echo ">> Docker context = colima-mlops"
else
	@echo ">> Linux detected: using native Docker"
	@docker info >/dev/null 2>&1 || { echo "ERROR: Docker not reachable. Start it (e.g. 'sudo systemctl start docker') or check permissions."; exit 1; }
endif

demo: engine-up  ## ONE COMMAND: engine up -> train -> run full stack -> traffic
	$(PY) src/train.py
	$(COMPOSE) up -d --build
	@echo ">> Waiting for services to start..."
	@sleep 10
	@echo ">> Generating sample traffic..."
	@for i in $$(seq 1 20); do \
	  curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
	    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}' >/dev/null; \
	  curl -s http://127.0.0.1:8000/health >/dev/null; \
	done
	@echo ""
	@echo "==================== Everything is running ===================="
	@echo "  API docs    -> http://localhost:8000/docs"
	@echo "  Prometheus  -> http://localhost:9090/targets"
	@echo "  Grafana     -> http://localhost:3000  (admin/admin)"
	@echo "  MLflow      -> run 'make mlflow' then http://localhost:5001"
	@echo "==============================================================="

down:  ## Stop the full stack (and on macOS, the Colima engine)
	-$(COMPOSE) down
	-minikube stop
ifeq ($(UNAME),Darwin)
	-docker context use desktop-linux
	-colima stop --profile mlops
endif
	@echo ">> Stopped."

all: setup demo  ## FIRST-TIME full run: create venv + install, then demo

# ---------------------------------------------------------------------------
# One-command Kubernetes deployment (Stage 7) - works on macOS and Linux
# ---------------------------------------------------------------------------

k8s-demo: engine-up  ## ONE COMMAND: build image, start Minikube, deploy to k8s
	$(PY) src/train.py
	docker build -t heart-disease-api:latest .
	minikube start --driver=docker
	minikube image load heart-disease-api:latest
	kubectl apply -f k8s/
	@echo ">> Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=heart-disease-api --timeout=150s
	@echo ""
	kubectl get pods -l app=heart-disease-api -o wide
	kubectl get svc heart-disease-api
	@echo ""
	@echo "==================== Deployed to Kubernetes ===================="
	@echo "  To access the API, run this in a SEPARATE terminal:"
	@echo "      minikube service heart-disease-api --url"
	@echo "  then curl the printed URL, e.g. curl http://127.0.0.1:PORT/health"
	@echo "==============================================================="

k8s-down:  ## Remove the k8s deployment and stop Minikube
	-kubectl delete -f k8s/
	-minikube stop
	@echo ">> Kubernetes deployment removed and Minikube stopped."


