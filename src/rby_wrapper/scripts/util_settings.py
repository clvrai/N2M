import numpy as np

class Settings:
    # ===== Data Collection =====
    master_arm_loop_period = 1 / 100
    data_collection_rate = 200
    master_arm_q_limit_barrier = 0.5
    master_arm_min_q = np.deg2rad(
        [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
    )
    master_arm_max_q = np.deg2rad(
        [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
    )
    master_arm_torque_limit = np.array([4.0] * 14)
    master_arm_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)
    
    # ===== IMPEDANCE CONTROL =====
    impedance_stiffness = 30
    impedance_damping_ratio = 1.0
    impedance_torque_limit = 10.0
    
    # ===== RBY1 =====
    rby1_address = "192.168.30.1:50051"
    rby1_model = "A"
    rby1_power = ".*"
    rby1_servo = "torso_.*|right_arm_.*|left_arm_.*"
    rby1_control_mode = "position"
    
    rby1_control_frequency = 50
    rby1_state_update_frequency = rby1_control_frequency