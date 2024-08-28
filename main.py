import argparse

import numpy as np
import rerun as rr

from dataset import CameraData, GroundTruthData, ImuData, TumDataset
from msckf import MSCKF

positions = []
gt_positions = []


def log_imu(imu_data: ImuData):
    if imu_data is not None:
        rr.log("imu/gyro/x", rr.Scalar(imu_data.gyro[0]))
        rr.log("imu/gyro/y", rr.Scalar(imu_data.gyro[1]))
        rr.log("imu/gyro/z", rr.Scalar(imu_data.gyro[2]))
        rr.log("imu/accel/x", rr.Scalar(imu_data.accel[0]))
        rr.log("imu/accel/y", rr.Scalar(imu_data.accel[1]))
        rr.log("imu/accel/z", rr.Scalar(imu_data.accel[2]))


def log_camera(camera_data: CameraData):
    if camera_data is not None:
        rr.log("camera", rr.Image(camera_data.image))


def log_frame(position: np.ndarray, rotation_matrix: np.ndarray, radii: float = 0.01, size: float = 1.0):
    x = rotation_matrix @ np.array([size, 0.0, 0.0])
    y = rotation_matrix @ np.array([0.0, size, 0.0])
    z = rotation_matrix @ np.array([0.0, 0.0, size])

    origins = [position, position, position]
    vectors = [x, y, z]
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    positions.append(position)

    rr.log("world/frame", rr.Arrows3D(vectors=vectors, origins=origins, colors=colors, radii=radii))
    rr.log("world/positions", rr.Points3D(positions=positions), timeless=True)


def log_pinhole(position: np.ndarray, rotation_matrix: np.ndarray, image: np.ndarray):
    height, width = image.shape[:2]
    rr.log(
        "world/camera",
        rr.Transform3D(
            translation=position.flatten(),
            mat3x3=rotation_matrix,
            axis_length=1.0,
        ),
    )
    rr.log(
        "world/camera/image",
        rr.Pinhole(
            focal_length=255,
            width=width,
            height=height,
            image_plane_distance=0.5,
        ),
    )
    rr.log("world/camera/image", rr.Image(image))


def log_state(msckf: MSCKF):
    velocity = msckf.state.velocity.reshape(3)
    rr.log("state/velocity/x", rr.Scalar(velocity[0]))
    rr.log("state/velocity/y", rr.Scalar(velocity[1]))
    rr.log("state/velocity/z", rr.Scalar(velocity[2]))
    rr.log("state/velocity/norm", rr.Scalar(np.linalg.norm(velocity)))


def log_ground_truth(ground_truth: GroundTruthData):
    gt_positions.append(ground_truth.translation)
    rr.log("world/ground_truth", rr.Points3D(positions=gt_positions))


def setup_msckf(dataset: TumDataset) -> MSCKF:
    return MSCKF(
        gyro_noise=dataset.gyro_noise_density,
        gyro_bias_noise=dataset.gyro_random_walk,
        accel_noise=dataset.accel_noise_density,
        accel_bias_noise=dataset.accel_random_walk,
        gyro_bias=[0.0283122, 0.00723077, 0.0165292],
        accel_bias=[-1.30318, -0.391441, 0.380509],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-f", "--frames", type=int)
    args = parser.parse_args()

    dataset = TumDataset(args.dataset, max_frames=args.frames)
    msckf = setup_msckf(dataset)

    rr.init("MSCKF", spawn=True)

    for imu_data, camera_data, ground_truth in dataset:
        rr.set_time_nanos("sensors", dataset.timestamp)
        log_imu(imu_data)
        log_camera(camera_data)

        if imu_data is not None:
            msckf.propagate(1 / dataset.imu_sampling_frequency, imu_data.gyro, imu_data.accel)
            log_frame(msckf.state.position.reshape(3), msckf.state.rotation_matrix)
            log_state(msckf)

        if camera_data is not None:
            rotation = msckf.state.rotation_matrix @ dataset.R_cam_imu
            position = msckf.state.position.reshape(3) + dataset.t_cam_imu
            log_pinhole(position.reshape(3), rotation, camera_data.image)

        if ground_truth is not None:
            log_ground_truth(ground_truth)
