
from scipy.spatial.transform import Rotation
import numpy as np

def convert_extrinsic_to_pos_and_quat(extrinsic):
    pos = extrinsic[:3, 3]
    rot = Rotation.from_matrix(extrinsic[:3, :3])
    quat_xyzw = rot.as_quat()  # scipy returns [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # convert to [w,x,y,z]
    return pos, quat_wxyz

# since mujoco x axis is flipped, we need to flip the x axis of the command
def qpos_command_wrapper(command: np.ndarray):
    new_command = np.zeros_like(command)
    new_command[0] = -command[0]
    new_command[1] = command[1]
    new_command[2] = command[2]
    return new_command