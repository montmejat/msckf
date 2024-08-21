import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from utils import cov_matrix, norm, omega, rk4

GRAVITY = np.array([0.0, 0.0, 9.81])


class State:
    def __init__(self):
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.gyro_bias = np.zeros(3)
        self.velocity = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.position = np.zeros(3)


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
        - gyro_bias_noise (3x1): The noise of the gyroscope bias in (rad/s²) * (1/√Hz).
        - acceleration_noise (3x1): The noise density of the accelerometer in (m/s²) * (1/√Hz).
        - acceleration_bias_noise (3x1): The noise of the velocity in (m/s³) * (1/√Hz).
        """

        self.P = cov_matrix(
            np.concatenate([init_gyro_std, init_gyro_bias_std, init_accel_std, init_vel_std, init_pos_std])
        )
        self.Q_imu = cov_matrix(
            np.concatenate([gyro_noise, gyro_bias_noise, accel_noise, accel_bias_noise])
        )

        self.Phi = np.eye(15)
        self.state = State()

    def propagate(self, dt: float, gyro: ArrayLike, accel: ArrayLike):
        gyro = gyro - self.state.gyro_bias
        accel = accel - self.state.accel_bias

        def dq(quat):
            return 0.5 * omega(gyro) @ quat

        quat = norm(rk4(dq, dt, self.state.quaternion))

        vel = self.state.velocity + (R.from_quat(quat).as_matrix().T @ accel - GRAVITY) * dt
        pos = self.state.position + dt * vel

        self.state.quat = quat
        self.state.velocity = vel
        self.state.position = pos
