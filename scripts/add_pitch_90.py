import numpy as np
from typing import List, Tuple

def matrix_to_euler(matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert 4x4 transformation matrix to Euler angles (roll, pitch, yaw)
    
    Args:
        matrix: 4x4 transformation matrix
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Extract rotation matrix (3x3) from transformation matrix
    R = matrix[:3, :3]
    
    # Calculate Euler angles
    # Pitch (y-axis rotation)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    # Yaw (z-axis rotation)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    # Roll (x-axis rotation)
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return roll, pitch, yaw

def euler_to_matrix(roll: float, pitch: float, yaw: float, translation: List[float] = None) -> np.ndarray:
    """
    Convert Euler angles to 4x4 transformation matrix
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        translation: Optional translation vector [x, y, z]
        
    Returns:
        4x4 transformation matrix
    """
    # Create rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations
    R = Rz @ Ry @ Rx
    
    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    
    # Add translation if provided
    if translation is not None:
        matrix[:3, 3] = translation
    
    return matrix

def add_pitch_90(matrix: np.ndarray) -> np.ndarray:
    """
    Add 90 degrees to the pitch angle of a 4x4 transformation matrix
    
    Args:
        matrix: Input 4x4 transformation matrix
        
    Returns:
        New 4x4 transformation matrix with 90 degrees added to pitch
    """
    # Extract current translation
    translation = matrix[:3, 3]
    
    # Convert to Euler angles
    roll, pitch, yaw = matrix_to_euler(matrix)
    
    # Add 90 degrees (π/2 radians) to pitch
    pitch += np.pi / 2
    
    # Convert back to matrix with original translation
    return euler_to_matrix(roll, pitch, yaw, translation)

def main():
    # Example usage
    # Original 4x4 transformation matrix
    original_matrix = np.array([
        [0.998711109, -0.0507553257, 0, 1.05258238],
        [0.0507553257, 0.998711109, 0, 0.393582493],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Add 90 degrees to pitch
    new_matrix = add_pitch_90(original_matrix)
    
    print("Original matrix:")
    print(original_matrix)
    print("\nNew matrix:")
    print(new_matrix)
    
    # Verify the change in pitch
    _, original_pitch, _ = matrix_to_euler(original_matrix)
    _, new_pitch, _ = matrix_to_euler(new_matrix)
    
    print(f"\nOriginal pitch: {np.degrees(original_pitch):.2f} degrees")
    print(f"New pitch: {np.degrees(new_pitch):.2f} degrees")

if __name__ == "__main__":
    main()
