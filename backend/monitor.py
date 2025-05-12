#!/usr/bin/env python3
import os
import sys
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

SERVICES = {
    "Airflow": os.getenv("AIRFLOW_URL", "http://localhost:8080/health"),
    "FastAPI": os.getenv("BACKEND_URL", "http://localhost:3500/health"),
    "Prometheus": os.getenv("PROM_URL", "http://localhost:9090/metrics"),
    "Grafana": os.getenv("GRAF_URL", "http://localhost:3000/api/health")
}

def check(name, url):
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            logging.info("%s is healthy (%s)", name, url)
        else:
            logging.error("%s returned %d", name, r.status_code)
            sys.exit(1)
    except Exception as e:
        logging.error("%s check failed: %s", name, e)
        sys.exit(1)

def main():
    for svc, url in SERVICES.items():
        check(svc, url)

if __name__ == "__main__":
    main()
