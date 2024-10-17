import logging
import os
from datetime import datetime

# Create a log file name with a timestamp
Log_File = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Set up the log path and ensure the directory exists
Log_Path = os.path.join(os.getcwd(), "logs")
os.makedirs(Log_Path, exist_ok=True)       

# Define the full log file path
Log_File_Path = os.path.join(Log_Path, Log_File)   

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=Log_File_Path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)
