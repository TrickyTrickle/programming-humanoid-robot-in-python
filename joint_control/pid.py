'''In this exercise you need to implement the PID controller for joints of robot.

* Task:
    1. complete the control function in PIDController with prediction
    2. adjust PID parameters for NAO in simulation

* Hints:
    1. the motor in simulation can simple modelled by angle(t) = angle(t-1) + speed * dt
    2. use self.y to buffer model prediction
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'software_installation'))

import numpy as np
from collections import deque
from spark_agent import SparkAgent, JOINT_CMD_NAMES


class PIDController(object):
    '''a discretized PID controller, it controls an array of servos,
       e.g. input is an array and output is also an array
    '''
    def __init__(self, dt, size):
        '''
        @param dt: step time
        @param size: number of control values
        @param delay: delay in number of steps
        '''
        self.dt = dt
        self.u = np.zeros(size)
        self.e1 = np.zeros(size)
        self.e2 = np.zeros(size)
        # ADJUST PARAMETERS BELOW
        delay = size - 1
        self.Kp = 5
        self.Ki = 0.2
        self.Kd = 0.2
        self.y = deque(np.zeros(size*2))
        self.z = deque(np.zeros(size*2))
        self.clamp = deque(np.zeros(size))

    def set_delay(self, delay):
        '''
        @param delay: delay in number of steps
        '''
        self.y = deque(self.y, delay + 1)

    """
    def control(self, target, sensor):

    for n in range(len(target)):
        target_val = target[n]
        sensor_val = sensor[n]

        error = target_val - sensor_val

        I_error = self.z[(n*2)+1]

        if self.clamp[n] == 0:
            I_error += error * self.dt
        else:
            I_error += 0

        self.z[(n*2)+1] = I_error

        D_error = (error - self.z[n*2]) / self.dt

        self.z[n*2] = error

        PID = self.Kp * error + self.Ki * I_error + self.Kd * D_error

        if PID > 1 or PID < 0:
            self.u[n] = 1
            signs = np.sign([error, PID])

            if signs[0] == signs[1]:
                self.clamp[n] = 1
            else:
                self.clamp[n] = 0
        else:
            self.u[n] = PID
            self.clamp[n] = 0

    return self.u
    """

    def control(self, target, sensor):
        '''apply PID control
        @param target: reference values
        @param sensor: current values from sensor
        @return control signal
        '''

        for n in range(len(target)):
            target_val = target[n]
            sensor_val = sensor[n]

            error = target_val - sensor_val

            I_error = self.z[(n*2)+1]
            I_error += error * self.dt
            self.z[(n*2)+1] = I_error

            D_error = (error - self.z[n*2]) / self.dt

            self.z[n*2] = error

            PID = self.Kp * error + self.Ki * I_error + self.Kd * D_error

            self.u[n] = PID

        return self.u


class PIDAgent(SparkAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PIDAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.joint_names = JOINT_CMD_NAMES.keys()
        number_of_joints = len(self.joint_names)
        self.joint_controller = PIDController(dt=0.01, size=number_of_joints)
        self.target_joints = {k: 0 for k in self.joint_names}

    def think(self, perception):
        action = super(PIDAgent, self).think(perception)
        '''calculate control vector (speeds) from
        perception.joint:   current joints' positions (dict: joint_id -> position (current))
        self.target_joints: target positions (dict: joint_id -> position (target)) '''
        joint_angles = np.asarray(
            [perception.joint[joint_id]  for joint_id in JOINT_CMD_NAMES])
        target_angles = np.asarray([self.target_joints.get(joint_id,
            perception.joint[joint_id]) for joint_id in JOINT_CMD_NAMES])
        u = self.joint_controller.control(target_angles, joint_angles)
        action.speed = dict(zip(JOINT_CMD_NAMES.keys(), u))  # dict: joint_id -> speed
        return action


if __name__ == '__main__':
    agent = PIDAgent()
    agent.target_joints['HeadYaw'] = 1.0
    agent.run()
