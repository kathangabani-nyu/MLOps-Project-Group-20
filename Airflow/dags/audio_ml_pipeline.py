from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from docker.types import Mount
import subprocess
import sys

# Python callables for test, load test, and monitor

def offline_eval_callable():
    subprocess.run([sys.executable, '/mnt/block/MLOps-Project-Group-20/backend/test_model.py'], check=True)

def load_test_callable():
    subprocess.run([sys.executable, '/mnt/block/MLOps-Project-Group-20/backend/load_test.py'], check=True)

def monitor_callable():
    subprocess.run([sys.executable, '/mnt/block/MLOps-Project-Group-20/backend/monitor.py'], check=True)

def_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'audio_ml_pipeline',
    default_args=def_args,
    description='ML pipeline with staged deployment',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:

    # 1. Model logged (train)
    log_model = DockerOperator(
        task_id='log_model',
        image='mlops-project-group-20_backend:latest',
        command='python -c "import time; print(\"Starting model training...\"); time.sleep(120); print(\"Model training completed.\")"',
        working_dir='/app',
        mounts=[
            Mount(source='/mnt/block/MLOps-Project-Group-20', target='/app', type='bind'),
            Mount(source='/mnt/block/bart-large-cnn', target='/app/models/base', type='bind', read_only=True),
        ],
        environment={
            'PYTHONPATH': '/app'
        },
    )

    # 2. Offline eval (test)
    offline_eval = PythonOperator(
        task_id='offline_eval',
        python_callable=offline_eval_callable,
    )

    # 3. Build container (docker build)
    build_container = DockerOperator(
        task_id='build_container',
        image='docker:24.0.2',
        api_version='auto',
        auto_remove=True,
        command='docker build -t mlops-project-group-20_backend:latest .',
        working_dir='/mnt/block/MLOps-Project-Group-20/backend',
        mounts=[
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
            Mount(source='/mnt/block/MLOps-Project-Group-20/backend', target='/mnt/block/MLOps-Project-Group-20/backend', type='bind'),
        ],
    )

    # 4. Deploy to staging (launch persistent container)
    deploy_staging = DockerOperator(
        task_id='deploy_staging',
        image='docker:24.0.2',
        api_version='auto',
        auto_remove=True,
        command='sh -c "docker rm -f backend-staging || true && docker run -d --rm --name backend-staging -p 3501:3500 -e ENVIRONMENT=staging mlops-project-group-20_backend:latest"',
        mounts=[
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
    )

    # 5. Load test
    load_test = PythonOperator(
        task_id='load_test',
        python_callable=load_test_callable,
    )

    # 6. Deploy to canary (launch persistent container)
    deploy_canary = DockerOperator(
        task_id='deploy_canary',
        image='docker:24.0.2',
        api_version='auto',
        auto_remove=True,
        command='sh -c "docker rm -f backend-canary || true && docker run -d --rm --name backend-canary -p 3502:3500 -e ENVIRONMENT=canary mlops-project-group-20_backend:latest"',
        mounts=[
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
    )

    # 7. Monitor
    monitor_performance = PythonOperator(
        task_id='monitor_performance',
        python_callable=monitor_callable,
    )

    # 8. Deploy to prod (launch persistent container)
    deploy_prod = DockerOperator(
        task_id='deploy_prod',
        image='docker:24.0.2',
        api_version='auto',
        auto_remove=True,
        command='sh -c "docker rm -f backend-prod || true && docker run -d --rm --name backend-prod -p 3503:3500 -e ENVIRONMENT=production mlops-project-group-20_backend:latest"',
        mounts=[
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
        ],
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    log_model >> offline_eval >> build_container >> deploy_staging >> load_test >> deploy_canary >> monitor_performance >> deploy_prod 
