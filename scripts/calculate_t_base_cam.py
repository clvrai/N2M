import numpy as np

def calculate_target_se3(se2_pose):
    """Convert SE2 pose to SE3 transformation matrix"""
    x, y, theta = se2_pose
    se3_pose = np.array([
        [np.cos(theta), -np.sin(theta), 0, x],
        [np.sin(theta), np.cos(theta), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return se3_pose

def calculate_t_base_cam(base_se2_pose, camera_extrinsic):
    """
    Calculate T_base_cam transformation matrix
    
    Args:
        base_se2_pose: [x, y, theta] of the base
        camera_extrinsic: 4x4 camera extrinsic matrix
        
    Returns:
        T_base_cam: 4x4 transformation matrix from base to camera
    """
    # Calculate base SE3 transformation matrix
    base_se3 = calculate_target_se3(base_se2_pose)
    
    # Calculate T_base_cam using the equation:
    # camera_extrinsic = base_se3 @ T_base_cam
    # Therefore: T_base_cam = base_se3^(-1) @ camera_extrinsic
    T_base_cam = np.linalg.inv(base_se3) @ camera_extrinsic
    
    return T_base_cam

def main():
    # Example usage with values from your data
    # Base SE2 pose [x, y, theta]
    base_se2_pose = [1.05258238, 0.393582493, -0.050777142265839278]
    
    # Camera extrinsic matrix (example from your data)
    camera_extrinsic = np.array([
        [0.87645483, 0.050665088, 0.478811026, 1.13526034],
        [-0.0444627628, 0.998715699, -0.0242901985, 0.419428736],
        [-0.479426771, 1.44728368e-10, 0.877581894, 1.41216779],
        [0, 0, 0, 1]
    ])
    
    # Calculate T_base_cam
    T_base_cam = calculate_t_base_cam(base_se2_pose, camera_extrinsic)
    
    print("T_base_cam matrix:")
    print(T_base_cam)
    
    # Verify the calculation
    base_se3 = calculate_target_se3(base_se2_pose)
    reconstructed_camera_extrinsic = base_se3 @ T_base_cam
    
    print("\nVerification:")
    print("Original camera extrinsic:")
    print(camera_extrinsic)
    print("\nReconstructed camera extrinsic:")
    print(reconstructed_camera_extrinsic)
    print("\nMaximum difference:", np.max(np.abs(camera_extrinsic - reconstructed_camera_extrinsic)))

if __name__ == "__main__":
    main()




