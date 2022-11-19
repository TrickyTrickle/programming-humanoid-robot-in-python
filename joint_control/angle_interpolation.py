'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import wipe_forehead


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.keyframe_time = 0
        self.currently_executing_keyframe_motion = False
        self.current_keyframe = None
        self.keyframe_start_time = 0
        self.keyframe_total_time = 0

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        if 'LHipYawPitch' in target_joints:
            target_joints['RHipYawPitch'] = target_joints['LHipYawPitch'] # copy missing joint in keyframes
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def get_total_keyframe_time(self, keyframe_times):
        time = 0
        for joint_times in keyframe_times:
            if joint_times[-1] > time:
                time = joint_times[-1]
        return time


    def cubic_bezier(self, keys, times, current_time, perception, key_name):
        n = -1
        current_angle = 0

        for j in range(len(times)):
            if times[j] > current_time:
                n = j
                break

        if n == -1:
            current_angle = None
        else:
            relative_time = current_time/times[-1]

            # if n == 0 get the current angle of the joint and do the interpoaltion with that as the starting point
            # if the joint name is not in the list for perception use 0 as the starting point
            # if != 0 do a standart cubic bezier
            if n == 0:
                if key_name in perception.joint:
                    current_angle = (1 - relative_time)**2 * perception.joint[key_name] + (1 - relative_time) * 2 * relative_time * (keys[n][0] + keys[n][1][2]) + relative_time**2 * keys[n][0]
                else:
                    current_angle = (1 - relative_time) * 2 * relative_time * (keys[n][0] + keys[n][1][2]) + relative_time**2 * keys[n][0]
            else:
                current_angle = (1 - relative_time)**3 * keys[n-1][0] + (1 - relative_time)**2 * 3 * relative_time * (keys[n-1][0] + keys[n-1][2][2]) + (1 - relative_time) * 3 * relative_time**2 * (keys[n][0] + keys[n][1][2]) + relative_time**3 * keys[n][0]

        return current_angle

    def angle_interpolation(self, keyframes, perception):
        if keyframes != ([], [], []) and keyframes != self.current_keyframe:
            self.currently_executing_keyframe_motion = True
            self.current_keyframe = keyframes
            self.keyframe_start_time = perception.time
            self.keyframe_time = 0
            self.keyframe_total_time = self.get_total_keyframe_time(keyframes[1])

        elif perception.time - self.keyframe_start_time > self.keyframe_total_time and self.currently_executing_keyframe_motion == True:
            self.currently_executing_keyframe_motion = False
            self.keyframe_start_time = 0
            self.keyframe_time = 0
            self.keyframe_total_time = 0
            self.current_keyframe = None
            self.keyframes = ([], [], [])

        self.keyframe_time = perception.time - self.keyframe_start_time
        target_joints = {}

        for n in range(len(keyframes[0])):
            if keyframes[1][n][-1] >= self.keyframe_time:

                current_angle = self.cubic_bezier(keyframes[2][n], keyframes[1][n], self.keyframe_time, perception, keyframes[0][n])

                if current_angle == None:
                    if not (keyframes[0][n] in perception.joint):
                        continue;
                    else:
                        current_angle = perception.joint[keyframes[0][n]]

                target_joints[keyframes[0][n]] = current_angle

        return target_joints






if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = wipe_forehead("asd")  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
