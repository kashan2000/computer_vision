"This module is created to Records the logs"
import os
import logging
from datetime import datetime

LogFile = f'{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log'
LogPath = os.path.join(os.getcwd(), 'logs', LogFile)
os.makedirs(LogPath, exist_ok=True)
log_path = os.path.join(LogPath,LogFile)
logging.basicConfig(filename=log_path,
format='[%(asctime)s]%(lineno)d%(levelname)s-%(message)s', level=logging.INFO)
