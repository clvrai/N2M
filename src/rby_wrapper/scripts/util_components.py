import logging
import threading
import time
import numpy as np
import rby1_sdk as rby


GRIPPER_DIRECTION = False

class Gripper:
    """
    Class for gripper
    """

    def __init__(self):
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([1, 1])
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q: np.typing.NDArray = None
        self._running = False
        self._thread = None
        self._motor_states = None
        self._motor_states_thread = None

    def initialize(self, verbose=False):
        rv = True
        for dev_id in [0, 1]:
            if not self.bus.ping(dev_id):
                if verbose:
                    logging.error(f"Dynamixel ID {dev_id} is not active")
                rv = False
            else:
                if verbose:
                    logging.info(f"Dynamixel ID {dev_id} is active")
        if rv:
            logging.info("Servo on gripper")
            self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        return rv

    def set_operating_mode(self, mode):
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 0.5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q
            # A small value (e.g., 5) was too short and failed to detect limits properly, so a reasonably larger value was chosen.
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        return True

    def get_motor_states(self):
        return self._motor_states
    
    def set_target(self, normalized_q):
        # self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            logging.error("Cannot set target. min_q or max_q is not valid.")
            return

        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q
            
        
    # ===== These functions are used for RBY1 ros2 manager, which uses *timer and topic* to manage the gripper =====
    # ===== This way is preferred because it is more efficient and easier to manage =====
    def update_motor_states(self):
        motor_states = self.bus.group_fast_sync_read_encoder([0, 1])
        if motor_states is not None:
            self._motor_states = motor_states

    def loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        if self.target_q is not None:
            target_q = self.target_q.copy()
            self.bus.group_sync_write_send_position(
                [(dev_id, q) for dev_id, q in enumerate(target_q.tolist())]
            )
    
    # ===== These functions are used for teleoperation, which uses *threads* to manage the gripper =====
    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.loop_thread, daemon=True)
            self._thread.start()
    
    def start_motor_states_update(self):
        if self._motor_states_thread is None or not self._motor_states_thread.is_alive():
            self._motor_states_thread = threading.Thread(target=self.update_motor_states_thread, daemon=True)
            self._motor_states_thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        
        if self._motor_states_thread is not None:
            self._motor_states_thread.join(timeout=1.0)
            self._motor_states_thread = None

    def update_motor_states_thread(self):
        while self._running:
            motor_states = self.bus.group_fast_sync_read_encoder([0, 1])
            if motor_states is not None:
                self._motor_states = motor_states
            time.sleep(0.01)  # maximum 100Hz

    def loop_thread(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            if self.target_q is not None:
                target_q = self.target_q.copy()
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(target_q.tolist())]
                )
            time.sleep(0.01)

            

