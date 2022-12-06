'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
from numpy import matrix, linalg, asarray
from scipy.linalg import pinv


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        max_step = 0.1
        lambda_ = 1

        joint_angles = []
        chain = self.chains[effector_name]
        theta = []

        for i in range(len(chain)):
            theta.append(self.perception.joint[chain[i]])

        target = matrix([self.from_trans(transform)]).T
        for i in range(1000):
            self.forward_kinematics(self.perception.joint)
            forward_trans = [identity(4), identity(4)]
            for i in range(len(chain)):
                forward_trans.append(self.transforms[chain[i]])

            Te = matrix([self.from_trans(forward_trans[-1])]).T

            e = target - Te
            e[e > max_step] = max_step
            e[e < -max_step] = -max_step

            T = matrix([self.from_trans(i) for i in forward_trans[1:-1]]).T
            J = Te - T
            dT = Te - T

            J[0, :] = -dT[2, :]
            J[1, :] = dT[1, :]
            J[2, :] = dT[0, :]

            J[-1, :] = 1
            d_theta = lambda_ * pinv(J) * e
            theta += asarray(d_theta.T)[0]


            if  linalg.norm(d_theta) < 1e-1:
                joint_angles = theta
                break

        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        joint_angles = self.inverse_kinematics(effector_name, transform)

        chain = self.chains[effector_name]
        names = list()
        times = list()
        keys = list()

        for i in range(len(chain)):
            names.append(chain[i])
            times.append([i*1 + 1])
            keys.append([[joint_angles[i], [0, 0, 0],[0, 0, 0]]])


        self.keyframes = (names, times, keys)  # the result joint angles have to fill in
        print("Done")

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[1, -1] = 0.05
    T[2, -1] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()
