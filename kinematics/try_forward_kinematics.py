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
from numpy import dot
from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):

    X = 0
    Y = 1
    Z = 2

    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains =  {'Head': ['HeadYaw', 'HeadPitch'],
                        'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw'],
                        'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                        'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
                        'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
                       }

        # Maps all chains to the relevant links
        self.chains_links = {'Head':'Head', 'LArm':'Arms', 'LLeg':'Legs', 'RLeg':'Legs', 'RArm':'Arms'}

        # links defines all links from the documentation
        self.links = {  ['Torso', 'HeadYaw', 0.00, 0.00, 126.50],
                        ['HeadYaw', 'HeadPitch', 0.00, 0.00, 0.00],
                        ['Torso', 'LShoulderPitch', 0.00, 98.00, 100.00],
                        ['LShoulderPitch', 'LShoulderRoll', 0.00, 0.00, 0.00],
                        ['LShoulderRoll', 'LElbowYaw', 105.00, 15.00, 0.00],
                        ['LElbowYaw', 'LElbowRoll', 0.00, 0.00, 0.00],
                        ['LElbowRoll', 'LWristYaw', 55.95, 0.00, 0.00],
                        ['Torso', 'RShoulderPitch', 0.00, -98.00, 100.00],
                        ['RShoulderPitch', 'RShoulderRoll', 0.00, 0.00, 0.00],
                        ['RShoulderRoll', 'RElbowYaw', 105.00, -15.00, 0.00],
                        ['RElbowYaw', 'RElbowRoll', 0.00, 0.00, 0.00],
                        ['RElbowRoll', 'RWristYaw', 55.95, 0.00, 0.00],
                        ['Torso', 'LHipYawPitch', 0.00, 50.00, -85.00],
                        ['LHipYawPitch', 'LHipRoll', 0.00, 0.00, 0.00],
                        ['LHipRoll', 'LHipPitch', 0.00, 0.00, 0.00],
                        ['LHipPitch', 'LKneePitch', 0.00, 0.00, -100.00],
                        ['LKneePitch', 'LAnklePitch', 0.00, 0.00, -102.90],
                        ['LAnklePitch', 'LAnkleRoll', 0.00, 0.00, 0.00],
                        ['Torso', 'RHipYawPitch', 0.00, -50.00, -85.00],
                        ['RHipYawPitch', 'RHipRoll', 0.00, 0.00, 0.00],
                        ['RHipRoll', 'RHipPitch', 0.00, 0.00, 0.00],
                        ['RHipPitch', 'RKneePitch', 0.00, 0.00, -100.00],
                        ['RKneePitch', 'RAnklePitch', 0.00, 0.00, -102.90],
                        ['RAnklePitch', 'RAnkleRoll', 0.00, 0.00, 0.00]
                     }


        # TODO: Remove probably
        self.lengths = {'ShoulderOffsetY' : 98.00,
                        'ElbowOffsetY'    : 15.00,
                        'UpperArmLength'  : 105.00,
                        'LowerArmLength'  : 55.95,
                        'ShoulderOffsetZ' : 100.00,
                        'HandOffsetX'     : 57.75,
                        'HandOffsetZ'     : 12.31
                        }

    def find_chain(self, joint_name):
        for chain in self.chains:
            if joint_name in chain.values:
                return chain

        return None

    def find_chain_number(self, chain, joint_name):
        for i in range(len(chain.values)):
            if chain[i] == joint_name:
                return i

        return None


    def get_d(self, link_name, old_z, old_joint, new_joint):
        link = self.links[link_name]

        for joint_link in link:
            if joint_link[0] == old_joint and joint_link[1] == new_joint:
                 return joint_link[2+old_z] # 2 is the offset for the array where the coordinates begin

        return None

    def think(self, perception):
        print("here")
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        T = identity(4)

        chain = find_chain(joint_name)
        n = find_chain_number(chain, joint_name)

        if n == 0:
            previous_joint_name = 'Torso'
            old_z = self.Z
        else:
            previous_joint_name = chain[n-1]
            old_z =



        theta = joint_angle
        d = get_d(self.links[self.chains_links[chain.key]], old_z, previous_joint_name, joint_name)
        alpha =
        r =

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

                print(Joint, ": ", T)
        sys.exit("Done after one full run")

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    print("here1")
    agent.run()
