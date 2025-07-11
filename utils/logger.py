from loguru import logger
import sys
import os

# Remove default logger
logger.remove()

# Add colored, timestamped stdout logger
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>", level="INFO")


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger.add(f"{log_dir}/pipeline.log", rotation="1 MB", retention="7 days", level="DEBUG")



