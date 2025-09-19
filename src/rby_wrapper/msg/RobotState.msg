# Timestamp
builtin_interfaces/Time timestamp

# Left arm state
float64[] left_arm_qpos
float64[] left_arm_vel
float64[] left_arm_current
float64[] left_arm_torque
float64[] left_arm_target_qpos
float64[] left_arm_target_qvel

# Right arm state
float64[] right_arm_qpos
float64[] right_arm_vel
float64[] right_arm_current
float64[] right_arm_torque
float64[] right_arm_target_qpos
float64[] right_arm_target_qvel

# Torso state
float64[] torso_qpos
float64[] torso_vel
float64[] torso_current
float64[] torso_torque
float64[] torso_target_qpos
float64[] torso_target_qvel

# Mobility state
float64[] mobility_qpos
float64[] mobility_vel
float64[] mobility_current
float64[] mobility_torque
float64[] mobility_target_qpos
float64[] mobility_target_qvel

# Gripper state
float64 left_gripper_qpos
float64 left_gripper_target_qpos
float64 left_gripper_min_qpos
float64 left_gripper_max_qpos
float64 left_gripper_qpos_normalized
float64 left_gripper_target_qpos_normalized

float64 right_gripper_qpos
float64 right_gripper_target_qpos
float64 right_gripper_min_qpos
float64 right_gripper_max_qpos
float64 right_gripper_qpos_normalized
float64 right_gripper_target_qpos_normalized
