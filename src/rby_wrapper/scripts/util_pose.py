import numpy as np
from dataclasses import dataclass

@dataclass
class Pose:
    torso: np.ndarray
    right_arm: np.ndarray
    left_arm: np.ndarray

INIT_POSE = {
    "base": np.deg2rad([0.0, 0.0, 0.0]),
    # "right_arm": np.deg2rad([25.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
    "right_arm": np.array([ 0.38086009, -0.07534731, -0.08269068, -2.01318651, -0.22272251, 1.18611996, -0.47718307]),
    "right_arm_intermediate": np.deg2rad([25.0, -15.0, 0.0, -120.0, 0.0, 70.0, -85.0]),
    "left_arm": np.deg2rad([25.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
    "torso_top": np.deg2rad([0.0, 22.0, -44.0, 22.0, 0.0, 0.0]),
    "torso_mid": np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]),
    "torso_bottom": np.deg2rad([0.0, 68.0, -136.0, 68.0, 0.0, 0.0]),
}

MANIPULATION_POSE = {
    "shelf_top": Pose(
        torso=INIT_POSE["torso_top"],
        right_arm=INIT_POSE["right_arm"],
        left_arm=INIT_POSE["left_arm"],
    ),
    "shelf_mid": Pose(
        torso=INIT_POSE["torso_mid"],
        right_arm=INIT_POSE["right_arm"],
        left_arm=INIT_POSE["left_arm"],
    ),
    "shelf_bottom": Pose(
        torso=INIT_POSE["torso_bottom"],
        right_arm=INIT_POSE["right_arm"],
        left_arm=INIT_POSE["left_arm"],
    ),
    "shelf_bottom_intermediate": Pose(
        torso=INIT_POSE["torso_bottom"],
        right_arm=INIT_POSE["right_arm_intermediate"],
        left_arm=INIT_POSE["left_arm"],
    ),
    "microwave": Pose(
        torso=INIT_POSE["torso_mid"],
        right_arm=INIT_POSE["right_arm"],
        left_arm=INIT_POSE["left_arm"],
    ),
}

MANIPULATION_POSE_DICT = {
    0: "shelf_top",
    1: "shelf_mid",
    2: "shelf_bottom",
    3: "microwave",
}

def SE2_to_xytheta(SE2_matrix):
    """
    Convert an SE2 matrix to x, y, theta representation.
    
    Parameters:
    -----------
    SE2_matrix : np.ndarray
        A 3x3 homogeneous transformation matrix of the form:
        [R t]
        [0 1]
        where R is a 2x2 rotation matrix and t is a 2x1 translation vector.
    
    Returns:
    --------
    np.ndarray
        A 3-element array containing [x, y, theta] where:
        - x, y are the translation components
        - theta is the rotation angle in radians
    """
    # Extract x, y from the translation part
    x = SE2_matrix[0, 2]
    y = SE2_matrix[1, 2]
    
    # Extract theta from the rotation part (arctan2(R21, R11))
    theta = np.arctan2(SE2_matrix[1, 0], SE2_matrix[0, 0])
    
    return np.array([x, y, theta])

def SE2_to_SE3(SE2_matrix):
    """
    Convert an SE2 matrix to SE3 matrix for 3D transformations.
    
    Parameters:
    -----------
    SE2_matrix : np.ndarray
        A 3x3 homogeneous transformation matrix in SE(2).
        
    Returns:
    --------
    np.ndarray
        A 4x4 homogeneous transformation matrix in SE(3) with:
        - Same x, y translation as input
        - z = 0
        - Rotation around z-axis from the input's rotation
    """
    # Create a 4x4 identity matrix
    SE3_matrix = np.eye(4)
    
    # Copy the rotation part (upper-left 2x2) to the new matrix
    SE3_matrix[0:2, 0:2] = SE2_matrix[0:2, 0:2]
    
    # Set the rotation elements for z-dimension
    # The rotation is only around z-axis, so R31=R32=R13=R23=0, R33=1
    SE3_matrix[2, 2] = 1.0
    
    # Copy the translation for x and y
    SE3_matrix[0, 3] = SE2_matrix[0, 2]
    SE3_matrix[1, 3] = SE2_matrix[1, 2]
    # z translation is 0
    SE3_matrix[2, 3] = 0.0
    
    return SE3_matrix

def xytheta_to_SE3(x, y, theta):
    """
    Convert x, y, theta representation to SE3 matrix for TF transformations.
    
    Parameters:
    -----------
    x : float
        X-coordinate in meters
    y : float
        Y-coordinate in meters
    theta : float
        Rotation angle in radians
        
    Returns:
    --------
    np.ndarray
        A 4x4 homogeneous transformation matrix in SE(3)
    """
    # Create rotation matrix around z-axis
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Create a 4x4 transformation matrix
    SE3_matrix = np.array([
        [c, -s, 0, x],
        [s,  c, 0, y],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    
    return SE3_matrix