import numpy as np
from typing import List

from DigitalTyphoonImage import DigitalTyphoonImage


class DigitalTyphoonSequence:

    def __init__(self, sequence: str, year: int, satellite: str, image_track_data_path=''):
        self.sequence = sequence
        self.year = year
        self.satellite = satellite
        self.image_track_data_path = image_track_data_path
        self.track_data = np.array([])

        # Ordered list containing sequence images
        self.images: List[DigitalTyphoonImage] = list()

        # Path to tsv containing track data
        self.set_track_data_path(image_track_data_path)

        # Track data
        self.track_data = []

    def get_sequence(self):
        return self.sequence

    def append_image_to_sequence(self, image: DigitalTyphoonImage) -> DigitalTyphoonImage:
        self.images.append(image)
        return self.images[-1]

    def set_track_data_path(self, track_data_path: str) -> str:
        self.image_track_data_path = track_data_path
        return self.image_track_data_path

    def get_track_data_path(self):
        return self.image_track_data_path

    def get_start_year(self):
        return self.year

    def get_number_of_images_in_sequence(self):
        return len(self.images)

    def has_images(self):
        return len(self.images) != 0

    def append_track_frame(self, year: int, month: int, day: int, hour: int,
                           grade: int, lat: float, long: float, pressure: float,
                           max_wind: float, max_gust: float, storm_wind_direction: float,
                           storm_radius_major: float, storm_radius_minor: float,
                           gale_wind_direction: float, gale_radius_major: float,
                           gale_radius_minor: float, indicator_landfall: float,
                           moving_speed: float, moving_direction: float,
                           interpolated_flag: int):
        if len(self.track_data) == 0:
            self.year = year

        self.track_data.append([year, month, day, hour,
                                grade, lat, long, pressure,
                                max_wind, max_gust, storm_wind_direction,
                                storm_radius_major, storm_radius_minor,
                                gale_wind_direction, gale_radius_major,
                                gale_radius_minor, indicator_landfall,
                                moving_speed, moving_direction,
                                interpolated_flag])

    # def get_h5_image_as_numpy(self, i):
