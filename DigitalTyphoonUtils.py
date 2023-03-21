import json
import csv
from datetime import datetime
import pandas as pd
import numpy as np


def parse_image_filename(filename: str, separator='-') -> (str, datetime, str):
    date, sequence_num, satellite, _ = filename.split(separator)
    date_year = int(date[:4])
    date_month = int(date[4:6])
    date_day = int(date[6:8])
    date_hour = int(date[8:10])
    sequence_datetime = datetime(year=date_year, month=date_month,
                                 day=date_day, hour=date_hour)
    return sequence_num, sequence_datetime, satellite

def get_seq_str_from_track_filename(filename: str) -> str:
    sequence_num = filename.removesuffix(".csv")
    return sequence_num

def is_image_file(filename: str) -> bool:
    return filename.endswith(".h5")