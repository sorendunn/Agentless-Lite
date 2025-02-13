import logging
import os


def setup_logging(instance_id, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"instance_{instance_id}")

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        log_file = os.path.join(log_dir, f"{instance_id}.log")
        file_handler = logging.FileHandler(log_file, mode="a")

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
