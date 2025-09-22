import numpy as np

def arm_fake_controller(env, mode):
    maximum_step = 0.05
    threshold = 0.01
    
    # Define preset joint positions for different modes
    DETECT_POS = [0.00748372, -1.28215526, -0.03359372, -1.71793475, -0.01200928, 1.37912621, 0.75347592]
    MANIPULATION_POS = np.array(env.robots[0].init_qpos)
    
    # Set target based on mode
    if mode == "DETECT":
        target = DETECT_POS
        print("[arm_fake_controller] arm move for detect!")
    elif mode == "MANIPULATION":
        target = MANIPULATION_POS
        print("[arm_fake_controller] arm move for manipulation!")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'DETECT' or 'MANIPULATION'")
    
    # Move arm to target position
    while True:
        curr_arm_joint_pos = env.sim.data.qpos[env.robots[0]._ref_arm_joint_pos_indexes]
        arm_joint_pos_error = target - curr_arm_joint_pos
        next_arm_joint_pos = curr_arm_joint_pos + np.clip(arm_joint_pos_error, -maximum_step, maximum_step)
        env.sim.data.qpos[env.robots[0]._ref_arm_joint_pos_indexes] = next_arm_joint_pos
        env.sim.forward()
        env.step(np.zeros(12))
        if np.all(np.abs(arm_joint_pos_error) < threshold):
            break