"""Fixed sensor calibration loaded from a GLIM config_sensors.json.

GLIM names transforms ``T_A_B``: the matrix that takes a point expressed in
frame B and expresses it in frame A (p_A = T_A_B @ p_B). This was checked
against the bag's own /tf_static (os_sensor -> os_imu matches T_lidar_imu
exactly), and the "camera" frame it uses (rgb_camera_link) is the optical
convention (x-right, y-down, z-forward) GLIM's own visual factors need to use
K/D directly -- the same convention OpenCV's projectPoints expects.
"""
import json

import numpy as np
from scipy.spatial.transform import Rotation


def _pose7_to_matrix(pose7):
    t = np.asarray(pose7[:3], dtype=np.float64)
    q = np.asarray(pose7[3:], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


class SensorConfig:
    def __init__(self, config_sensors_path):
        with open(config_sensors_path) as f:
            sensors = json.load(f)["sensors"]

        self.T_lidar_camera = _pose7_to_matrix(sensors["T_lidar_camera"])
        self.T_camera_lidar = np.linalg.inv(self.T_lidar_camera)

        self.image_width, self.image_height = sensors["image_size"]
        fx, fy, cx, cy = sensors["intrinsics"]
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.D = np.asarray(sensors["distortion_coeffs"], dtype=np.float64)

        if sensors["distortion_model"] != "rational_polynomial":
            raise ValueError(f"Unsupported distortion model: {sensors['distortion_model']}")
