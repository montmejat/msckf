import argparse

import rerun as rr

from dataset import CameraData, ImuData, TumDataset
from msckf import MSCKF


def log_imu(imu_data: ImuData):
    if imu_data is not None:
        rr.log("imu/wx", rr.Scalar(imu_data.wx))
        rr.log("imu/wy", rr.Scalar(imu_data.wy))
        rr.log("imu/wz", rr.Scalar(imu_data.wz))
        rr.log("imu/ax", rr.Scalar(imu_data.ax))
        rr.log("imu/ay", rr.Scalar(imu_data.ay))
        rr.log("imu/az", rr.Scalar(imu_data.az))


def log_camera(camera_data: CameraData):
    if camera_data is not None:
        rr.log("camera", rr.Image(camera_data.image))


def setup_msckf(dataset: TumDataset) -> MSCKF:
    gyro_var = dataset.gyroscope_noise_density**2 * dataset.imu_sampling_frequency

    return MSCKF(
        # Note: (rad/s)/√Hz = rad/s)*√s
        # ((rad/s)/√Hz)**2 * Hz = rad**2 / s**2
        [gyro_var] * 3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    dataset = TumDataset(args.dataset)
    msckf = setup_msckf(dataset)

    rr.init("MSCKF", spawn=True)

    i = 0
    for imu_data, camera_data in dataset:
        rr.set_time_nanos("sensors", dataset.timestamp)
        log_imu(imu_data)
        log_camera(camera_data)

        i += 1
        if i == 50:
            break
