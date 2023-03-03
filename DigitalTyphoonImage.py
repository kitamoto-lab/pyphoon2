import h5py
import numpy as np
from datetime import datetime


class DigitalTyphoonImage:
    def __init__(self, image_path: str, date: datetime, sequence: str, satellite: str):
        self.image_path = image_path
        self.date = ''
        self.sequence = ''
        self.satellite = ''

        self.set_date(date)
        self.set_sequence(sequence)
        self.set_satellite(satellite)

    def get_date(self) -> datetime:
        return self.date

    def get_sequence(self) -> str:
        return self.sequence

    def get_satellite(self) -> str:
        return self.satellite

    def set_date(self, date: datetime) -> datetime:
        self.date = date
        return self.date

    def set_sequence(self, sequence: str) -> str:
        self.sequence = sequence
        return self.sequence

    def set_satellite(self, satellite: str) -> str:
        self.satellite = satellite
        return self.satellite

    def get_image_array(self, spectrum='infrared'):
        with h5py.File(self.image_path, 'r') as h5f:
            image = np.array(h5f.get(spectrum))
        return image
