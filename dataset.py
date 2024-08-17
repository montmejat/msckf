import csv
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ImuData:
    timestamp: int
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float


@dataclass
class CameraData:
    timestamp: int
    image: np.ndarray


class TumDataset:
    def __init__(self, folder_path: str, imu: str = "imu0", camera: str = "cam0"):
        with open(f"{folder_path}/mav0/{imu}/data.csv") as f:
            reader = csv.reader(f)
            next(reader)
            self.imu = [ImuData(int(row[0]), *map(float, row[1:])) for row in reader]

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
        self.imu_idx = 0
        self.cam_idx = 0
        self.timestamp = 0

    def __next__(self):
        def next_camera_data():
            cam_data = self.cam[self.cam_idx]
            self.timestamp = cam_data["timestamp"]
            self.cam_idx += 1
            return CameraData(
                self.timestamp, np.array(Image.open(cam_data["filepath"]))
            )

        def next_imu_data():
            imu_data = self.imu[self.imu_idx]
            self.timestamp = imu_data.timestamp
            self.imu_idx += 1
            return imu_data

        if self.imu_idx < len(self.imu) and self.cam_idx < len(self.cam):
            if self.imu[self.imu_idx].timestamp < self.cam[self.cam_idx]["timestamp"]:
                return next_imu_data()
            else:
                return next_camera_data()
        elif self.imu_idx < len(self.imu):
            return next_imu_data()
        elif self.cam_idx < len(self.cam):
            return next_camera_data()
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.imu) + len(self.cam)
