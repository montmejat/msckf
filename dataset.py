import csv
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml
from PIL import Image


@dataclass
class ImuData:
    timestamp_ns: int
    gyro: np.ndarray
    accel: np.ndarray


@dataclass
class CameraData:
    timestamp_ns: int
    image: np.ndarray


class TumDataset:
    def __init__(self, folder_path: str, imu: str = "imu0", camera: str = "cam0", max_frames: int = None):
        with open(f"{folder_path}/mav0/{imu}/data.csv") as f:
            reader = csv.reader(f)
            next(reader)
            self.imu = [
                ImuData(int(row[0]), np.array(row[1:4], dtype=float), np.array(row[4:], dtype=float))
                for row in reader
            ]

        with open(f"{folder_path}/mav0/{camera}/data.csv") as f:
            reader = csv.reader(f)
            next(reader)
            self.cam = [
                {
                    "timestamp": int(row[0]),
                    "filepath": f"{folder_path}/mav0/{camera}/data/{row[1]}",
                }
                for row in reader
            ]

        self.camera = camera
        self.max_frames = max_frames
        self.imu_idx = 0
        self.cam_idx = 0
        self.timestamp = 0

        with open(f"{folder_path}/dso/imu_config.yaml") as f:
            imu_config = yaml.safe_load(f)
            self.gyro_noise_density = np.array([imu_config["gyroscope_noise_density"]] * 3)
            self.gyro_random_walk = np.array([imu_config["gyroscope_random_walk"]] * 3)
            self.accel_noise_density = np.array([imu_config["accelerometer_noise_density"]] * 3)
            self.accel_random_walk = np.array([imu_config["accelerometer_random_walk"]] * 3)
            self.imu_sampling_frequency = imu_config["update_rate"]

        with open(f"{folder_path}/dso/camchain.yaml") as f:
            imu_config = yaml.safe_load(f)
            self.T_cam_imu = np.array(imu_config[self.camera]["T_cam_imu"])
            self.R_cam_imu = self.T_cam_imu[:3, :3]
            self.t_cam_imu = self.T_cam_imu[:3, 3]

    def __next__(self) -> Tuple[ImuData, CameraData]:
        if self.max_frames is not None and self.imu_idx >= self.max_frames:
            raise StopIteration

        def next_camera_data():
            cam_data = self.cam[self.cam_idx]
            self.timestamp = cam_data["timestamp"]
            self.cam_idx += 1
            return CameraData(self.timestamp, np.array(Image.open(cam_data["filepath"])))

        def next_imu_data():
            imu_data = self.imu[self.imu_idx]
            self.timestamp = imu_data.timestamp_ns
            self.imu_idx += 1
            return imu_data

        if self.imu_idx < len(self.imu) and self.cam_idx < len(self.cam):
            if self.imu[self.imu_idx].timestamp_ns < self.cam[self.cam_idx]["timestamp"]:
                return (next_imu_data(), None)
            elif self.imu[self.imu_idx].timestamp_ns > self.cam[self.cam_idx]["timestamp"]:
                return (None, next_camera_data())
            else:
                return (next_imu_data(), next_camera_data())
        elif self.imu_idx < len(self.imu):
            return (next_imu_data(), None)
        elif self.cam_idx < len(self.cam):
            return (None, next_camera_data())

        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.imu) + len(self.cam)
