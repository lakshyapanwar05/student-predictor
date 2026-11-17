# Student Performance Predictor (Podman Capstone)

Predicts whether a student will pass based on study hours, attendance, and previous marks.
A logistic regression model (compared against a decision tree) is trained in one container, and a
FastAPI service with an animated web UI serves predictions and logs them to PostgreSQL in another.

## Architecture
- **trainer**: one-shot container that trains the models and writes `model.joblib` and `metrics.json` to the `model-store` volume
- **api**: FastAPI + web UI on port 8000, reads the model from `model-store` (read-only)
- **db**: PostgreSQL 16, data stored in the `pgdata` volume
- **predictor-net**: custom Podman network so `api` reaches `db` by name

## Images
- https://hub.docker.com/r/lakshyapanwar05/student-api
- https://hub.docker.com/r/lakshyapanwar05/student-trainer

## Prerequisites
Podman (rootless), Python 3 (only to generate the dataset).

## Steps

### 1. Generate the dataset
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas
python data/make_data.py
```

### 2. Build the images
```bash
podman build -t student-trainer:1.0 trainer/
podman build -t student-api:1.1 api/
```

### 3. Create the network and volumes
```bash
podman network create predictor-net
podman volume create model-store
podman volume create pgdata
```

### 4. Train the model
```bash
podman run --rm --name trainer -v "$(pwd)/data:/data:ro" -v model-store:/models student-trainer:1.0
```

### 5. Start the database and the API
```bash
podman run -d --name db --network predictor-net -v pgdata:/var/lib/postgresql/data \
  -e POSTGRES_USER=predictor -e POSTGRES_PASSWORD=predictor -e POSTGRES_DB=predictions \
  docker.io/library/postgres:16
podman run -d --name api --network predictor-net -p 8000:8000 \
  -v model-store:/models:ro -e DB_HOST=db student-api:1.1
```
Open http://localhost:8000

### 6. Test
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"study_hours": 6, "attendance": 80, "previous_marks": 65}'
curl http://localhost:8000/history
```

### 7. Management and troubleshooting
```bash
podman ps -a
podman logs api
podman inspect api
podman exec api python -c "import socket; print(socket.gethostbyname('db'))"
podman network inspect predictor-net
podman volume inspect model-store
podman stop db && podman start db
```
If the database is down, the API still predicts and returns `"logged": false`.

### 8. Push and pull
```bash
podman login docker.io
podman tag student-api:1.1 docker.io/lakshyapanwar05/student-api:1.1
podman push docker.io/lakshyapanwar05/student-api:1.1
podman pull docker.io/lakshyapanwar05/student-api:1.1
```

## Results
Logistic regression: 0.82 accuracy. Decision tree: 0.81 (100 held-out rows, synthetic dataset).
