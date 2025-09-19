# Timestamp
builtin_interfaces/Time timestamp

# command_type ("base", "right_arm", "torso")
string command_type

# Left arm state
float64[] left_arm_qpos
float64[] left_arm_torque

# Right arm state
float64[] right_arm_qpos
float64[] right_arm_torque

# Torso state
float64[] torso_qpos
float64[] torso_torque

# Mobility state
float64 base_vel
float64 base_omega

# Gripper state
float64 left_gripper_qpos_normalized
float64 right_gripper_qpos_normalized
