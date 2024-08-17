import argparse

import rerun as rr

from dataset import CameraData, ImuData, TumDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()

    dataset = TumDataset(args.dataset)

    rr.init("MSCKF", spawn=True)

    for data in dataset:
        rr.set_time_nanos("sensors", data.timestamp)

        if isinstance(data, CameraData):
            rr.log("camera", rr.Image(data.image))
        elif isinstance(data, ImuData):
            rr.log("imu/wx", rr.Scalar(data.wx))
            rr.log("imu/wy", rr.Scalar(data.wy))
            rr.log("imu/wz", rr.Scalar(data.wz))
            rr.log("imu/ax", rr.Scalar(data.ax))
            rr.log("imu/ay", rr.Scalar(data.ay))
            rr.log("imu/az", rr.Scalar(data.az))
