#!/usr/bin/env python3
import os
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def main():
    env = os.getenv("ENVIRONMENT", "staging").lower()
    image = "mlops-project-group-20-backend:latest"
    port_map = {
        "staging": "3501:3500",
        "canary":  "3502:3500",
        "production": "3503:3500"
    }

    if env not in port_map:
        logging.error(f"Unknown ENVIRONMENT '{env}', must be one of {list(port_map)}")
        exit(1)

    container = f"backend-{env}"
    ports = port_map[env]

    logging.info(f"Deploying image '{image}' as '{container}' on port {ports}")
    subprocess.run(["docker", "rm", "-f", container], check=False)
    subprocess.run([
        "docker", "run", "-d", "--rm",
        "--name", container,
        "-p", ports,
        "-e", f"ENVIRONMENT={env}",
        image
    ], check=True)
    logging.info("Deployment succeeded!")

if __name__ == "__main__":
    main()
