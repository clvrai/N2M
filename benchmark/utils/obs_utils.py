import numpy as np
from termcolor import colored
NAVI_MODE = "omni"

######################### memo for previous version #########################   
# navi_policy.set_goal_with_reset(target_se2)
# while True:
#     start = time.time()
#     se2 = obs_to_SE2(ob_dict)
#     ac[7:10] = navi_policy.step(se2)
#     ob_dict, r, done, info = env.step(ac)
#     if navi_policy.done:
#         break
#     end = time.time()
#     if end - start > navi_policy.dt:
#         print(colored("warning: step time is too long: {}ms".format((end - start)*1000), "yellow"))
#     else:
#         time.sleep(navi_policy.dt - (end - start))
                    
def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def obs_to_SE2(dic, algorithm_name=None):
    """
    Convert observation dictionary to SE2 pose
    """
    # if algorithm_name == "act":
    #     pos = dic['robot0_base_pos']
    #     quat = dic['robot0_base_quat']
    # else:
    name = "robot0_base_pos"
    # name = "mobilebase0_"
    pos = dic['robot0_base_pos'][-1] # [x, y, z], the last one is the current position
    quat = dic['robot0_base_quat'][-1]   # [x, y, z, w], the last one is the current orientation
    yaw = np.arctan2(2*(quat[3]*quat[2] + quat[0]*quat[1]), 1 - 2*(quat[1]**2 + quat[2]**2)) # quat to yaw
    se2 = np.array([pos[0], pos[1], yaw])
    return se2

def obs_to_SE3(dic):
    """
    Convert observation dictionary to SE3 pose as 4x4 transformation matrix
    """
    pos = dic['robot0_base_pos'][-1] # [x, y, z]
    quat = dic['robot0_base_quat'][-1]   # [x, y, z, w]
    
    # Convert quaternion to rotation matrix
    x, y, z, w = quat
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    
    return T
