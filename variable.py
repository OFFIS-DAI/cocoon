import os.path
from pathlib import Path

path = Path(__file__).absolute()
PATH = os.path.abspath(os.path.join(path, os.pardir))

SIMULATION_DATA_FOLDERS = ['simulation_data']
STATE_DATA_FOLDER = 'state_data'
