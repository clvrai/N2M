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
    if algorithm_name == "act":
        pos = dic['robot0_base_pos']
        quat = dic['robot0_base_quat']
    else:
        pos = dic['robot0_base_pos'][-1] # [x, y, z]
        quat = dic['robot0_base_quat'][-1]   # [x, y, z, w]
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

class NaviPolicy:
    def __init__(self, navi_mode=NAVI_MODE, max_vel=0.5):
        self.navi_mode = navi_mode
        self.goal = None
        self.dt = 0.05  # time step

        # PI controller gains
        self.Kp_x = 20.0
        self.Kp_y = 20.0
        self.Kp_theta = 0.5
        
        self.Ki_x = 10.
        self.Ki_y = 10.
        self.Ki_theta = 10.

        # Sliding window size for integral terms
        self.window_size = 5
        self.error_window_x = []
        self.error_window_y = []
        self.error_window_theta = []

        # Integral windup limits (optional, for safety)
        self.int_limit = 2.0

        # max velocity
        self.max_vel = max_vel
        self.max_omega = 0.2
        
        self.pos_error_limit = 0.005
        self.vel_limit = 0.05
        
        self.done = False

    def reset(self):
        self.goal = None
        self.done = False
        self.error_window_x = []
        self.error_window_y = []
        self.error_window_theta = []

    def set_goal_without_reset(self, goal:np.ndarray):
        self.goal = goal
        
    def set_goal_with_reset(self, goal:np.ndarray):
        """
        Set goal position [x_g, y_g, theta_g].
        """
        self.reset()
        self.goal = goal

    def step(self, obs:np.ndarray):
        """
        Compute control command [x_vel, y_vel, omega] given current observation [x, y, theta].
        The velocity command is in the robot-centric frame!
        """
        if self.goal is None:
            return np.array([0.0, 0.0, 0.0])

        x, y, theta = obs
        x_g, y_g, theta_g = self.goal

        # Compute world-frame errors
        dx = x_g - x
        dy = y_g - y
        dtheta = normalize_angle(theta_g - theta)

        # Project (dx, dy) from world frame to robot frame
        # [dx_robot]   [ cos(-theta)  -sin(-theta) ] [dx]
        # [dy_robot] = [ sin(-theta)   cos(-theta) ] [dy]
        # which is equivalent to:
        # dx_r = cos(theta) * dx + sin(theta) * dy
        # dy_r = -sin(theta) * dx + cos(theta) * dy
        cos_theta = np.cos(-theta)
        sin_theta = np.sin(-theta)
        dx_robot = cos_theta * dx - sin_theta * dy
        dy_robot = sin_theta * dx + cos_theta * dy

        # Update sliding window for errors
        self.error_window_x.append(dx_robot * self.dt)
        self.error_window_y.append(dy_robot * self.dt)
        self.error_window_theta.append(dtheta * self.dt)
        
        # Keep only the last window_size elements
        if len(self.error_window_x) > self.window_size:
            self.error_window_x.pop(0)
            self.error_window_y.pop(0)
            self.error_window_theta.pop(0)
        
        # Calculate integral terms as sum of error window
        int_x = sum(self.error_window_x)
        int_y = sum(self.error_window_y)
        int_theta = sum(self.error_window_theta)

        # Clamp integral terms
        int_x = np.clip(int_x, -self.int_limit, self.int_limit)
        int_y = np.clip(int_y, -self.int_limit, self.int_limit)
        int_theta = np.clip(int_theta, -self.int_limit, self.int_limit)

        # PI control law in robot frame
        x_vel = self.Kp_x * dx_robot + self.Ki_x * int_x
        y_vel = self.Kp_y * dy_robot + self.Ki_y * int_y
        omega = self.Kp_theta * dtheta + self.Ki_theta * int_theta

        # Clip velocities
        x_vel = np.clip(x_vel, -self.max_vel, self.max_vel)
        y_vel = np.clip(y_vel, -self.max_vel, self.max_vel)
        omega = np.clip(omega, -self.max_omega, self.max_omega)
        
        # print(colored("dx: {:.3f}, dy: {:.3f}, dtheta: {:.3f}".format(dx, dy, dtheta), "yellow"))
        # print(colored("x_vel: {:.3f}, y_vel: {:.3f}, omega: {:.3f}".format(x_vel, y_vel, omega), "yellow"))
        if abs(dx) < self.pos_error_limit and abs(dy) < self.pos_error_limit and abs(dtheta) < self.pos_error_limit:
            if abs(x_vel) < self.vel_limit and abs(y_vel) < self.vel_limit and abs(omega) < self.vel_limit:
                self.done = True

        return np.array([x_vel, y_vel, omega])