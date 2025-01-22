# Copyright 2018 CNRS

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import time
import numpy as np
from math import atan2, cos, sin
from pinocchio import centerOfMass, forwardKinematics
from cop_des import CoPDes
from com_trajectory import ComTrajectory
from inverse_kinematics import InverseKinematics
from tools import Constant, Piecewise

def rot_z(theta) : 
    return np.array([np.array([cos(theta), -sin(theta), 0]),
                     np.array([sin(theta), cos(theta), 0]),
                     np.array([0, 0, 1])])

# Computes the trajectory of a swing foot.
#
# Input data are
#  - initial and final time of the trajectory,
#  - initial and final pose of the foot,
#  - maximal height of the foot,
#
# The trajectory is polynomial with zero velocities at start and end.
# The orientation of the foot is kept as in intial pose.
class SwingFootTrajectory(object):
    def __init__(self, t_init, t_end, init, end, height):
        assert(init[2] == end[2])
        self.t_init = t_init
        self.t_end = t_end
        self.height = height
        # Write your code here
        self.init = init
        self.end = end 

    def __call__(self, t):
        # write your code here
        ti = self.t_init
        dt = t - ti
        T  = self.t_end - self.t_init

        # x coordinate
        x0, x1 = self.init[0], self.end[0]
        dx     = x0 - x1
        a0, a1, a2, a3 = x0, 0, -3*dx/T**2, 2*dx/T**3
        x = a3 * dt**3 + a2 * dt**2 + a1*dt + a0
        
        # y coordinate
        y0, y1= self.init[1], self.end[1]
        dy = y0 - y1
        z0 = self.init[2]
        h  = self.height + z0
        c0, c1, c2, c3 = y0, 0, -3*dy/T**2, 2*dy/T**3 
        y  = c3 * dt**3 + c2 * dt**2 + c1*dt + c0
        
        # z coordinate
        tmp_b = (16*(h-z0))/T**4
        b0, b1, b2, b3, b4 = z0, 0, tmp_b*T**2, -2*tmp_b*T, tmp_b
        z = b4 * dt**4 + b3 * dt**3 + b2 * dt**2 + b1 * dt**1 + b0
        
        return [x,y,z]

# Computes a walking whole-body motion
#
# Input data are
#  - an initial configuration of the robot,
#  - a sequence of step positions (x,y,theta) on the ground,
#  - a mapping from time to R corresponding to the desired orientation of the
#    waist. If not provided, keep constant orientation.
#
class WalkingMotion(object):
    step_height = 0.05
    single_support_time = 0.05 
    double_support_time = 0.01
    def __init__(self, robot):
        self.robot = robot

    def compute(self, q0, steps, waisOrientation=None):
        # Test input data
        if len(steps) < 4:
            raise RuntimeError("sequence of step should be of length at least 4 instead of " +
                               f"{len(steps)}")
        # Copy steps in order to avoid modifying the input list.
        steps_ = steps[:]
        # Compute offset between waist and center of mass since we control the center of mass
        # indirectly by controlling the waist.
        data = self.robot.model.createData()
        forwardKinematics(self.robot.model, data, q0)
        com = centerOfMass(self.robot.model, data, q0)
        waist_pose = data.oMi[self.robot.waistJointId]
        com_offset = waist_pose.translation - com
        # Trajectory of left and right feet
        self.lf_traj = Piecewise()
        self.rf_traj = Piecewise()
        
        # Write your code here
        # Initialization
        t = 0
        single_support_time, double_support_time = self.single_support_time, self.double_support_time
        start_left, start_right = data.oMi[self.robot.leftFootJointId].translation, data.oMi [self.robot.rightFootJointId].translation
        
        current_left, current_right = np.array(start_left), np.array(start_right)
        step_left, step_right=[step for step in steps if step[1]==-0.1],[step for step in steps if step[1]==0.1]
        
        self.rf_traj.segments.append(Constant(t, t+double_support_time, start_right))
        self.lf_traj.segments.append(Constant(t, t+double_support_time,  start_left))
        t += double_support_time

        for i in range(len(step_left)) : 
            if i != len(step_left) -1 : 
                end_right = np.array([step_right[i+1][0], step_right[i][1], current_right[2]])
            end_left = np.array([step_left[i][0], step_left[i][1], current_left[2]])
        
            # On déplace le pied gauche
            self.lf_traj.segments.append(SwingFootTrajectory(t,t+single_support_time, current_left, end_left, self.step_height))   
            current_left = end_left
            self.rf_traj.segments.append(Constant(t, t+single_support_time, current_right))
            t += single_support_time
            
            # Les deux pieds sont au sol
            self.rf_traj.segments.append(Constant(t, t+double_support_time, current_right))
            self.lf_traj.segments.append(Constant(t, t+double_support_time, current_left))
            t += double_support_time
            
            # On déplace le pied droit
            self.rf_traj.segments.append(SwingFootTrajectory(t,t+single_support_time, current_right, end_right, self.step_height))
            current_right = end_right
            self.lf_traj.segments.append(Constant(t, t+single_support_time,current_left))
            t += single_support_time
            
            # Les deux pieds sont au sol
            self.rf_traj.segments.append(Constant(t, t+double_support_time, current_right))
            self.lf_traj.segments.append(Constant(t, t+double_support_time, current_left))
            t += double_support_time

        self.com_trajectory = ComTrajectory(com[0:2], steps, np.array([1.6,0.0]), com[2])
        X = self.com_trajectory.compute() 
        times      = 0.01 * np.arange(len(X)//2)
        com        = np.array(list(map(self.com_trajectory, times)))
        right_foot = np.array(list(map(self.rf_traj, times)))
        left_foot  = np.array(list(map(self.lf_traj, times)))
        
        configs = []
        cop_des = np.array(list(map(self.com_trajectory.cop_des, times)))
       
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.plot(times, left_foot[:,0], label="x left foot")
        ax1.plot(times, right_foot[:,0], label="x right foot")
        ax1.plot(times, cop_des[:,0], label="x CoPdes")
        ax1.legend()
        ax2.plot(times, left_foot[:,1], label="y left foot")
        ax2.plot(times, right_foot[:,1], label="y right foot")
        ax2.plot(times, cop_des[:,1], label="y CoPdes")
        ax2.legend()
        ax3.plot(times, left_foot[:,2], label="z left foot")
        ax3.plot(times, right_foot[:,2], label="z right foot")
        ax3.legend()
        plt.show()
        '''

        # div = len(right_foot) // len(theta)

        i   = 0
        for t in range(len(right_foot)) : 
            ik = InverseKinematics (self.robot)
            ik.rightFootRefPose.translation = np.array (right_foot[t])
            ik.leftFootRefPose.translation  = np.array (left_foot[t])
            ik.waistRefPose.translation     = np.array (com[t] + com_offset)
            
            '''
            # waist Rotation
            rot = rot_z(theta[i])
            ik.rightFootRefPose.rotation = rot
            ik.leftFootRefPose.rotation  = rot
            ik.waistRefPose.rotation     = rot

            if  t % div == 0 and i < len(theta)-1: 
                i+=1
            '''
            q0 = neutral(robot.model)
            q0 [robot.name_to_config_index["leg_right_4_joint"]] = .2
            q0 [robot.name_to_config_index["leg_left_4_joint"]]  = .2
            q0 [robot.name_to_config_index["arm_left_2_joint"]]  = .2
            q0 [robot.name_to_config_index["arm_right_2_joint"]] = -.2

            q = ik.solve (q0)
            configs.append(q)

        return configs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from talos import Robot
    from pinocchio import neutral
    import numpy as np
    from inverse_kinematics import InverseKinematics
    from real_trajectory import RealTrajectory
    import eigenpy

    robot = Robot ()
    ik = InverseKinematics (robot)
    ik.rightFootRefPose.translation = np.array ([0, -0.1, 0.1])
    ik.leftFootRefPose.translation = np.array ([0, 0.1, 0.1])
    ik.waistRefPose.translation = np.array ([0, 0, 0.95])

    q0 = neutral (robot.model)
    q0 [robot.name_to_config_index["leg_right_4_joint"]] = .2
    q0 [robot.name_to_config_index["leg_left_4_joint"]] = .2
    q0 [robot.name_to_config_index["arm_left_2_joint"]] = .2
    q0 [robot.name_to_config_index["arm_right_2_joint"]] = -.2
    q = ik.solve (q0)
    robot.display(q)
    wm = WalkingMotion(robot)
    # First two values correspond to initial position of feet
    # Last two values correspond to final position of feet
    
    #start = np.array([0,0,0])
    #end   = np.array([5,0,0])
    #rt    = RealTrajectory(start, end)
    #steps, theta = rt.compute()
 
    steps = [np.array([0, -.1, 0.]), np.array([0.4, .1, 0.]),
             np.array([0.8, -.1, 0.]), np.array([1.2, .1, 0.]),
             np.array([1.6, -.1, 0.]), np.array([1.6, .1, 0.])]
 
    configs = wm.compute(q, steps)
    
    for q in configs:
        time.sleep(1e-2)
        robot.display(q)
    
    delta_t = wm.com_trajectory.delta_t
    times = delta_t*np.arange(wm.com_trajectory.N+1)
    left_foot = np.array(list(map(wm.lf_traj, times)))
    right_foot = np.array(list(map(wm.rf_traj, times)))
    cop_des = np.array(list(map(wm.com_trajectory.cop_des, times)))

    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.plot(times, left_foot[:,0], label="x left foot")
    ax1.plot(times, right_foot[:,0], label="x right foot")
    ax1.plot(times, cop_des[:,0], label="x CoPdes")
    ax1.legend()
    ax2.plot(times, left_foot[:,1], label="y left foot")
    ax2.plot(times, right_foot[:,1], label="y right foot")
    ax2.plot(times, cop_des[:,1], label="y CoPdes")
    ax2.legend()
    ax3.plot(times, left_foot[:,2], label="z left foot")
    ax3.plot(times, right_foot[:,2], label="z right foot")
    ax3.legend()
    plt.show()
    '''
