import numpy as np
from numpy.typing import ArrayLike

from utils import cov_matrix


class MSCKF:
    def __init__(
        self,
        gyro_noise: ArrayLike,
        gyro_bias_noise: ArrayLike,
        accel_noise: ArrayLike,
        accel_bias_noise: ArrayLike,
        init_gyro_std: ArrayLike = [1.0, 1.0, 1.0],
        init_gyro_bias_std: ArrayLike = [1.0, 1.0, 1.0],
        init_accel_std: ArrayLike = [1.0, 1.0, 1.0],
        init_vel_std: ArrayLike = [1.0, 1.0, 1.0],
        init_pos_std: ArrayLike = [1.0, 1.0, 1.0],
    ):
        """
        - gyro_noise (3x1): The noise density of the gyroscope in (rad/s) * (1/√Hz).
        - gyro_bias_noise (3x1): The noise of the gyroscope bias (rad/s/s).
        - acceleration_noise (3x1): The noise of the accelerometer (m/s²/√Hz).
        - acceleration_bias_noise (3x1): The noise of the velocity (m/s²/s).
        """

        self.P = cov_matrix(
            np.concatenate([init_gyro_std, init_gyro_bias_std, init_accel_std, init_vel_std, init_pos_std])
        )
        self.Q_imu = cov_matrix(
            np.concatenate([gyro_noise, gyro_bias_noise, accel_noise, accel_bias_noise])
        )

        self.Phi = np.eye(15)
