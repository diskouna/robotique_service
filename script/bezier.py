# Copyright 2024 CNRS

# Author: Florent Lamiraux

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
from pinocchio import forwardKinematics
from talos import Robot
from scipy.optimize import fmin_slsqp
from cop_des import CoPDes
from walking_motion_2 import WalkingMotion

class Bezier(object):
    """
    Bezier curve with any number of control points
    Evaluation is performed with de Casteljau algorithm.
    """
    def __init__(self, controlPoints):
        self.controlPoints = controlPoints

    def __call__(self, t):
        cp = self.controlPoints[:]
        while len(cp) > 1:
            cp1 = list()
            for p0, p1 in zip(cp, cp[1:]):
                cp1.append((1-t)*p0 + t*p1)
            cp = cp1[:]
        return cp[0]

    def derivative(self):
        """
        Return the derivative as a new Bezier curve
        """
        n = len(self.controlPoints) - 1
        cp = list()
        for P0, P1 in zip(self.controlPoints, self.controlPoints[1:]):
            cp.append(n*(P1-P0))
        return Bezier(cp)

def simpson(f, t_init, t_end, n_intervals):
    """
    Computation of an integral with Simpson formula
    """
    l = (t_end - t_init)/n_intervals
    t0 = t_init
    res = f(t0)/6
    for i in range(n_intervals):
        t1 = t0 + .5*l
        t2 = t0 + l
        res += 2/3*f(t1) + 1/3*f(t2)
        t0 = t2
    res -= f(t_end)/6
    res *= l
    return res

class Integrand(object):
    """
    Computes the integrand defining the integral cost for a given Bezier curve
    and a given parameter t as

         1     2           2
    I = --- (v   + alpha v  )
         2     T           N

    where
      - v  and v  are the tangent and normal velocities.
         T      N
    """
    alpha = 8

    def __init__(self, bezier):
        self.function = bezier
        self.derivative = bezier.derivative()

    def __call__(self, t):
        B_t = self.function(t)
        B_dot_t = self.derivative(t)
        theta = B_t[2]
        
        # Tangent velocity 
        v_T = np.dot([ np.cos(theta), np.sin(theta), 0], B_dot_t)
        # Normal velocity
        v_N = np.dot([-np.sin(theta), np.cos(theta), 0], B_dot_t)
        
        # Return integrand value
        return 0.5 * (v_T**2 + self.alpha * v_N**2)

class SlidingMotion(object):
    """
    Defines a sliding motion of the robot using Bezier curve and minimizing
    an integral cost favoring forward motions
    """
    beta = 100
    stepLength = .25
    def __init__(self, robot, q0, end):
        """ Constructor

        - input: q0 initial configuration of the robot,
        - end: end configuration specified as (x, y, theta) for the position
                and orientation in the plane.
        """
        self.robot = robot
        self.q0    = q0
        self.end   = end
        self.controlPoints = np.linspace(q0[:3], end, 6)
        self.bezier = Bezier(self.controlPoints)
        self.derivative = self.bezier.derivative()
        self.steps = list()
        self.bezier_path = list()

    def cost(self, X):
        """
        Compute the cost of a trajectory represented by a Bezier curve
        """
        assert(len(X.shape) == 1)
        self.bezier.controlPoints = X.reshape((6, 3))
        integrand = Integrand(self.bezier)
        integral_value = simpson(integrand, 0, 1, 100) # with n_intervals = 100
        
        theta_0 = self.bezier(0)[2]# TODO: q0
        theta_1 = end[2]
        B_dot_0, B_dot_1 = self.derivative(0), self.derivative(1)

        cost_boundary_start = np.dot([-np.sin(theta_0), np.cos(theta_0), 0], B_dot_0)
        cost_boundary_end   = np.dot([-np.sin(theta_1), np.cos(theta_1), 0], B_dot_1)
        
        cost_boundary = cost_boundary_start**2 + cost_boundary_end**2

        return integral_value + self.beta * cost_boundary

    def boundaryConstraints(self, X):
        """
        Computes the scalar product of the x-y velocity at the beginning 
        (resp. at the end) of the trajectory with the unit vector of initial
        (resp. end) orientation.
        """
        self.bezier.controlPoints = X.reshape((6, 3))
        B_dot_start = self.bezier.derivative()(0)
        B_dot_end   = self.bezier.derivative()(1)

        c_start = np.dot([-np.sin(self.q0[2]), np.cos(self.q0[2]), 0], B_dot_start)
        c_end = np.dot([-np.sin(self.end[2]), np.cos(self.end[2]), 0], B_dot_end)

        return [c_start, c_end]

    def solve(self):
        """
        Solve the optimization problem. Initialize with a straight line
        """
        x0 = self.controlPoints.flatten()

        result = fmin_slsqp(
            func=self.cost,
            x0=x0,
            f_eqcons=self.boundaryConstraints
        )
        self.controlPoints = result.reshape((6, 3))
        return self.controlPoints

    def leftFootPose(self, pose):
        res = np.zeros(3)   
        res[:2] = pose[:2] + np.array([-0.1 * sin(pose[2]), 0.1 * cos(pose[2])])
        res[2] = pose[2]
        return res

    def rightFootPose(self, pose):
        res = np.zeros(3)
        res[:2] = pose[:2] + np.array([0.1 * sin(pose[2]), -0.1 * cos(pose[2])])
        res[2] = pose[2]
        return res

    def computeMotion(self):
        configs = list()
        self.solve()

        steps = list()
        ts = np.linspace(0, 1, 10)
        
        for t in ts:
            pose = self.bezier(t)
            self.bezier_path.append(pose)
            if len(configs) % 2 == 0:
                steps.append(self.rightFootPose(pose))
            else:
                steps.append(self.leftFootPose(pose))
        self.steps = steps     
        
        wm = WalkingMotion(self.robot)
        configs = wm.compute(q0, steps)

        return configs 

if __name__ == '__main__':
    from talos import Robot
    robot = Robot()
    q0 = np.array([
        0.00000000e+00, 0.00000000e+00, 9.50023790e-01, 3.04115703e-04,
        0.00000000e+00, 0.00000000e+00, 9.99999957e-01, 0.00000000e+00,
        2.24440496e-02, -5.88127845e-01, 1.21572430e+00, -6.27580400e-01,
        -2.29184434e-02, 0.00000000e+00, -2.95804462e-02, -5.88175279e-01,
        1.21608861e+00, -6.27902977e-01, 2.91293666e-02, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 2.00000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, -2.00000000e-01, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
    end = np.array([2, 1, 1.57])
    sm = SlidingMotion(robot, q0, end)
    configs = sm.computeMotion()
    for q in configs:
        time.sleep(1e-2)
        robot.display(q)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    times = 1e-2*np.arange(101)
    #X = np.array(list(map(sm.slidingPath, times)))
    st_rf = sm.steps[0::2]
    st_lf = sm.steps[1::2]

    rf_x = [elmt[0] for elmt in st_rf]
    rf_y = [elmt[1] for elmt in st_rf]

    lf_x = [elmt[0] for elmt in st_lf]
    lf_y = [elmt[1] for elmt in st_lf]

    ax1.plot(rf_x, rf_y, 'o', label="right foot x-y path")
    ax2.plot(lf_x, lf_y, '*', label="left foot x-y path")

    plt.show()
