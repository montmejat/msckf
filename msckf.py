import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

from utils import cov_matrix, norm, omega, rk4

GRAVITY = np.array([[0.0], [0.0], [9.81]])


class State:
    def __init__(
        self,
        quaternion: ArrayLike = [0.0, 0.0, 0.0, 1.0],
        position: ArrayLike = [0.0, 0.0, 0.0],
        gyro_bias: ArrayLike = [0.0, 0.0, 0.0],
        accel_bias: ArrayLike = [0.0, 0.0, 0.0],
    ):
        self.quat = np.array(quaternion).reshape(4, 1)
        self.gyro_bias = np.array(gyro_bias).reshape(3, 1)
        self.velocity = np.zeros((3, 1))
        self.accel_bias = np.array(accel_bias).reshape(3, 1)
        self.position = np.array(position).reshape(3, 1)

    @property
    def rotation_matrix(self):
        return R.from_quat(self.quat.reshape(4)).as_matrix()


class MSCKF:
    def __init__(
        self,
        gyro_noise: ArrayLike,
        gyro_bias_noise: ArrayLike,
        accel_noise: ArrayLike,
        accel_bias_noise: ArrayLike,
        gyro_bias: ArrayLike = [0.0, 0.0, 0.0],
        accel_bias: ArrayLike = [0.0, 0.0, 0.0],
        init_gyro_std: ArrayLike = [1.0, 1.0, 1.0],
        init_gyro_bias_std: ArrayLike = [1.0, 1.0, 1.0],
        init_accel_std: ArrayLike = [1.0, 1.0, 1.0],
        init_vel_std: ArrayLike = [1.0, 1.0, 1.0],
        init_pos_std: ArrayLike = [1.0, 1.0, 1.0],
        init_position: ArrayLike = [0.0, 0.0, 0.0],
        init_quaternion: ArrayLike = [0.0, 0.0, 0.0, 1.0],
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
        self.state = State(init_quaternion, init_position, gyro_bias, accel_bias)

    def propagate(self, dt: float, gyro: np.ndarray, accel: np.ndarray):
        gyro = gyro.reshape(3, 1) - self.state.gyro_bias
        accel = accel.reshape(3, 1) - self.state.accel_bias

        def dq(quat):
            return 0.5 * omega(gyro) @ quat

        self.state.quat = norm(rk4(dq, dt, self.state.quat))

        rot_matrix = R.from_quat(self.state.quat.reshape(4)).as_matrix()
        accel = rot_matrix @ accel - GRAVITY

        self.state.velocity += accel * dt
        self.state.position += self.state.velocity * dt
