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
    filepath: str
    image: np.ndarray


@dataclass
class GroundTruthData:
    timestamp_ns: int
    translation: np.ndarray
    quaternion: np.ndarray


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
                CameraData(int(row[0]), f"{folder_path}/mav0/{camera}/data/{row[1]}", None)
                for row in reader
            ]

        with open(f"{folder_path}/dso/gt_imu.csv") as f:
            reader = csv.reader(f)
            next(reader)
            self.ground_truth = [
                GroundTruthData(
                    int(row[0]), np.array(row[1:4], dtype=float), np.array(row[4:], dtype=float)
                )
                for row in reader
            ]

        self.camera = camera
        self.max_frames = max_frames
        self.imu_idx = 0
        self.cam_idx = 0
        self.gt_idx = 0
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

    def __next__(self) -> Tuple[ImuData, CameraData, GroundTruthData]:
        if self.max_frames is not None and self.imu_idx >= self.max_frames:
            raise StopIteration

        choices = []
        if self.imu_idx < len(self.imu):
            choices.append(self.imu[self.imu_idx])
        if self.cam_idx < len(self.cam):
            choices.append(self.cam[self.cam_idx])
        if self.gt_idx < len(self.ground_truth):
            choices.append(self.ground_truth[self.gt_idx])

        if len(choices) == 0:
            raise StopIteration

        next_data = min(choices, key=lambda x: x.timestamp_ns)
        self.timestamp = next_data.timestamp_ns

        if isinstance(next_data, ImuData):
            self.imu_idx += 1
            return next_data, None, None
        elif isinstance(next_data, CameraData):
            self.cam_idx += 1
            next_data.image = np.array(Image.open(next_data.filepath))
            return None, next_data, None
        elif isinstance(next_data, GroundTruthData):
            self.gt_idx += 1
            return None, None, next_data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.imu) + len(self.cam)
