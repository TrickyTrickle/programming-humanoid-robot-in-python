'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity
from numpy import dot, sin, cos, pi, arccos
from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                        'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'],
                        'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                        'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
                        'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll']
                       }

        # links defines all links from the documentation
        # [joint1, joint2, x, y, z, d, theta, alpha, a]
        # x, y, z are relative to joint1
        # theta is in degrees
        # alpha is in degrees
        self.links = [  ['Torso', 'HeadYaw', 0.00, 0.00, 126.50, 126.5, 0, 0, 0.00],
                        ['HeadYaw', 'HeadPitch', 0.00, 0.00, 0.00, 0.00, 0, 270, 0.00],
                        ['Torso', 'LShoulderPitch', 0.00, 98.00, 100.00, 100.00, 90, 270, 98.00],
                        ['LShoulderPitch', 'LShoulderRoll', 0.00, 0.00, 0.00, 0, 270, 90, 0.00],
                        ['LShoulderRoll', 'LElbowYaw', 105.00, 15.00, 0.00, 105.00, 90, 270, 15.00],
                        ['LElbowYaw', 'LElbowRoll', 0.00, 0.00, 0.00, 0, 0, 90, 0.00],
                        ['LElbowRoll', 'LWristYaw', 55.95, 0.00, 0.00, 55.95, 0, 270, 0.00],
                        ['Torso', 'RShoulderPitch', 0.00, -98.00, 100.00, 100.00, 90, 270, 0.00],
                        ['RShoulderPitch', 'RShoulderRoll', 0.00, 0.00, 0.00, 0, 270, 90, 0.00],
                        ['RShoulderRoll', 'RElbowYaw', 105.00, -15.00, 0.00, 105.00, 90, 270, -15.00],
                        ['RElbowYaw', 'RElbowRoll', 0.00, 0.00, 0.00, 0, 0, 90, 0.00],
                        ['RElbowRoll', 'RWristYaw', 55.95, 0.00, 0.00, 55.95, 0, 270, 0.00],
                        ['Torso', 'LHipYawPitch', 0.00, 50.00, -85.00, -85.00, 0, 0, 0.00],
                        ['LHipYawPitch', 'LHipRoll', 0.00, 0.00, 0.00, 0, 0, 90, 0.00],
                        ['LHipRoll', 'LHipPitch', 0.00, 0.00, 0.00, 0, 90, 90, 0.00],
                        ['LHipPitch', 'LKneePitch', 0.00, 0.00, -100.00, 0, 0, 0, -100.00],
                        ['LKneePitch', 'LAnklePitch', 0.00, 0.00, -102.90, 0, 0, 0, -102.90],
                        ['LAnklePitch', 'LAnkleRoll', 0.00, 0.00, 0.00, 0, 270, 270, 0.00],
                        ['Torso', 'RHipYawPitch', 0.00, -50.00, -85.00, -85, 0, 0, 0.00],
                        ['RHipYawPitch', 'RHipRoll', 0.00, 0.00, 0.00, 0, 0, 90, 0.00],
                        ['RHipRoll', 'RHipPitch', 0.00, 0.00, 0.00, 0, 90, 90, 0.00],
                        ['RHipPitch', 'RKneePitch', 0.00, 0.00, -100.00, 0, 0, 0, -100.00],
                        ['RKneePitch', 'RAnklePitch', 0.00, 0.00, -102.90, 0, 0, 0, -102.90],
                        ['RAnklePitch', 'RAnkleRoll', 0.00, 0.00, 0.00, 0, 270, 270, 0]]

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)


    def find_d(self, joint_name):
        for link in self.links:
            if link[1] == joint_name:
                return link[-4]

        return None

    def find_a(self, joint_name):
        for link in self.links:
            if link[1] == joint_name:
                return link[-1]

        return None

    def find_theta(self, joint_name):
        for link in self.links:
            if link[1] == joint_name:
                return ((link[-3]/180.0)*pi)

        return None

    def find_alpha(self, joint_name):
        for link in self.links:
            if link[1] == joint_name:
                return ((link[-2]/180.0)*pi)

        return None


    def from_trans(self, trans):
        x = trans[0, -1]
        y = trans[1, -1]
        z = trans[2, -1]
        theta = arccos(trans[0,0])

        return [x, y, z, theta]

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        theta = self.find_theta(joint_name) + joint_angle
        d = self.find_d(joint_name)
        a = self.find_a(joint_name)
        alpha = self.find_alpha(joint_name)

        T = matrix([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                    [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                    [0         ,  sin(alpha)           ,  cos(alpha)            , d          ],
                    [0         ,  0                    ,  0                     , 1          ]])

        return T

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                T = dot(T, Tl)

                self.transforms[joint] = T

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
