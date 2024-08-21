import argparse

import rerun as rr

from dataset import CameraData, ImuData, TumDataset
from msckf import MSCKF


def log_imu(imu_data: ImuData):
    if imu_data is not None:
        rr.log("imu/wx", rr.Scalar(imu_data.gyro[0]))
        rr.log("imu/wy", rr.Scalar(imu_data.gyro[1]))
        rr.log("imu/wz", rr.Scalar(imu_data.gyro[2]))
        rr.log("imu/ax", rr.Scalar(imu_data.accel[0]))
        rr.log("imu/ay", rr.Scalar(imu_data.accel[1]))
        rr.log("imu/az", rr.Scalar(imu_data.accel[2]))


def log_camera(camera_data: CameraData):
    if camera_data is not None:
        rr.log("camera", rr.Image(camera_data.image))


def setup_msckf(dataset: TumDataset) -> MSCKF:
    return MSCKF(
        gyro_noise=dataset.gyro_noise_density,
        gyro_bias_noise=dataset.gyro_random_walk,
        accel_noise=dataset.accel_noise_density,
        accel_bias_noise=dataset.accel_random_walk,
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

        if imu_data is not None:
            msckf.propagate(1 / dataset.imu_sampling_frequency, imu_data.gyro, imu_data.accel)

        i += 1
        if i == 50:
            break
