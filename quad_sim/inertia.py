#!/usr/bin/env python

"""
Computing inertias of bodies.
Coordinate frame:
x - forward; y - left; z - up
The same coord frame is used for quads
All default inertias of objects are with respect to COM
Source of inertias: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
WARNIGN: The h,w,l of the BoxLink are defined DIFFERENTLY compare to Wiki
"""

import numpy as np
import copy

def rotate_I(I, R):
    """
    Rotating inertia tensor I
    R - rotation matrix
    """
    return R @ I @ R.T

def translate_I(I, m, xyz):
    """
    Offsetting inertia tensor I by [x,y,z].T
    relative to COM
    """
    x,y,z = xyz[0], xyz[1], xyz[2]
    I_new = np.zeros([3,3])
    I_new[0][0] = I[0][0] + m * (y**2 + z**2)
    I_new[1][1] = I[1][1] + m * (x**2 + z**2)
    I_new[2][2] = I[2][2] + m * (x**2 + y**2)
    I_new[0][1] = I_new[1][0] = I[0][1] + m * x * y
    I_new[0][2] = I_new[2][0] = I[0][1] + m * x * z
    I_new[1][2] = I_new[2][1] = I[1][2] + m * y * z
    return I_new

def deg2rad(deg):
    return deg / 180. * np.pi

class SphereLink():
    """
    Box object
    """
    type = "sphere"
    def __init__(self, r, m=None, density=None, name="sphere"):
        """
        m = mass
        dx = dy = dz = diameter = 2 * r
        """
        self.name = name
        self.r = r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    @property
    def I_com(self):
        r = self.r
        return np.array([
            [2/5. * self.m * r **2, 0., 0.],
            [0., 2/5. * self.m * r **2, 0.],
            [0., 0., 2/5. * self.m * r **2],
        ])

    def compute_m(self, density):
        return density * 4./3. * np.pi * self.r ** 3


class BoxLink():
    """
    Box object
    """
    type = "box"
    def __init__(self, l, w, h, m=None, density=None, name="box"):
        """
        m = mass
        dx = length = l
        dy = width = l
        dz = height = h
        """
        self.name = name
        self.l, self.w, self.h = l, w, h
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    @property
    def I_com(self):
        l ,w, h = self.l, self.w, self.h
        return np.array([
            [1/12. * self.m * (h**2 + w**2), 0., 0.],
            [0., 1/12. * self.m * (l**2 +  h**2), 0.],
            [0., 0., 1/12. * self.m * (w**2 + l**2)],
        ])
    
    def compute_m(self, density):
        return density * self.l * self.w * self.h

class RodLink():
    """
    Rod == Horizontal Cylinder
    """
    type = "rod"
    def __init__(self, l, r=0.002, m=None, density=None, name="rod"):
        """
        m = mass
        dx = length
        dy = dz = diameter == height
        """
        self.name = name
        self.l = l
        self.r = r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    @property
    def I_com(self):
        return np.array([
            [1/12. * self.m * self.l**2, 0., 0.],
            [0., 0., 0.],
            [0., 0., 1/12. * self.m * self.l**2],
        ])

    def compute_m(self, density):
        return density * np.pi * self.l * self.r ** 2

class CylinderLink():
    """
    Vertical Cylinder
    """
    type = "cylinder"
    def __init__(self, h, r, m=None, density=None, name="cylinder"):
        """
        m = mass
        dz = height = h
        dy = dx = 2*radius = 2*r = diameter
        """
        self.name = name
        self.h, self.r = h, r
        if m is None:
            self.m = self.compute_m(density)
        else:
            self.m = m
    
    @property
    def I_com(self):
        h, r = self.h, self.r
        return np.array([
            [1/12. * self.m * (3*r**2 + h**2), 0., 0.],
            [0., 1/12. * self.m * (3*r**2 + h**2), 0.],
            [0., 0., 0.5 * self.m * r**2],
        ])
    def compute_m(self, density):
        return density * np.pi * self.h * self.r ** 2

class LinkPose(object):
    def __init__(self, R=None, xyz=None, alpha=None):
        """
        One can provide either:
        R - rotation matrix or 
        alpha - angle of roation in a xy (horizontal) plane [degrees]
        xyz - offset
        """
        if xyz is not None:
            self.xyz = np.array(xyz)
        else:
            self.xyz = np.zeros(3)
        if R is not None:
            self.R = R
        elif alpha:
            self.R = np.array([
                [np.cos(alpha), -np.sin(alpha), 0.],
                [np.sin(alpha), np.cos(alpha), 0.],
                [0., 0., 1.]
            ])
        else:
            self.R = np.eye(3)


class QuadLink(object):
    """
    Quadrotor link set to compute inertia.
    Initial coordinate system assumes being in the middle of the central body.
    Orientation of axes: x - forward; y - left; z - up
    arm_angle == |/ , i.e. between the x axis and the axis of the arm
    Quadrotor assumes X configuration.
    """
    def __init__(self, params=None, verbose=False):
        # PARAMETERS (CrazyFlie by default)
        self.motors_num = 4
        self.params = {}
        self.params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        self.params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        self.params["arms"] = {"w":0.005, "h":0.005, "m":0.001}
        self.params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
        self.params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}

        self.params["arms_pos"] = {"angle": 45., "z": 0.}

        self.params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1.}
        self.params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
        if params is not None:
            self.params.update(params)
        else:
            print("WARN: since params is None the CrazyFlie params will be used")

        # Printing all params
        if verbose:
            print("######################################################")
            print("QUAD PARAMETERS:")
            [print(key,":", val) for key,val in self.params.items()]
            print("######################################################")
        
        # Dependent parameters
        self.arm_angle = deg2rad(self.params["arms_pos"]["angle"])
        if self.arm_angle == 0.:
            self.arm_angle = 0.01
        self.motor_xyz = np.array(self.params["motor_pos"]["xyz"])
        delta_y = self.motor_xyz[1] - self.params["body"]["w"] / 2.
        if "l" not in self.params["arms"]:
            self.arm_length =  delta_y / np.sin(self.arm_angle)
            self.params["arms"]["l"] = self.arm_length
        else:
            self.arm_length = self.params["arms"]["l"]
        # print("Arm length: ", self.arm_length, "angle: ", self.arm_angle)

        # Vectors of coordinates of the COMs of arms, s.t. their ends will be exactly at motors locations
        self.arm_xyz = np.array([ self.motor_xyz[0] - delta_y /(2 * np.tan(self.arm_angle)),
                                 self.motor_xyz[1] - delta_y / 2,
                                 self.params["arms_pos"]["z"] ])
        

        # X signs according to clockwise starting front-left
        # i.e. the list bodies are counting clockwise: front_right, back_right, back_left, front_left
        # See CrazyFlie doc for more details: https://wiki.bitcraze.io/projects:crazyflie2:userguide:assembly
        self.x_sign = np.array([1, -1, -1, 1])
        self.y_sign = np.array([-1, -1, 1, 1])
        self.sign_mx = np.array([self.x_sign, self.y_sign, np.array([1., 1., 1., 1.])])
        self.motors_coord = self.sign_mx * self.motor_xyz[:, None]
        self.props_coord = copy.deepcopy(self.motors_coord)
        self.props_coord[2,:] = (self.props_coord[2,:] + self.params["motors"]["h"] / 2. + self.params["propellers"]["h"])
        self.arm_angles = [
            -self.arm_angle, 
             self.arm_angle, 
            -self.arm_angle, 
             self.arm_angle]
        self.arms_coord = self.sign_mx * self.arm_xyz[:, None]

        # First defining the bodies
        self.body =  BoxLink(**self.params["body"], name="body") # Central body 
        self.payload = BoxLink(**self.params["payload"], name="payload") # Could include battery
        self.arms  = [BoxLink(**self.params["arms"], name="arm_%d" % i) for i in range(self.motors_num)] # Just arms
        self.motors =  [CylinderLink(**self.params["motors"], name="motor_%d" % i) for i in range(self.motors_num)] # The motors itself
        self.props =  [CylinderLink(**self.params["propellers"], name="prop_%d" % i) for i in range(self.motors_num)] # Propellers
        
        self.links = [self.body, self.payload] + self.arms + self.motors + self.props

        # print("######################################################")
        # print("Inertias:")
        # [print(link.I_com, "\n") for link in self.links]
        # print("######################################################")

        # Defining locations of all bodies
        self.body_pose = LinkPose()
        self.payload_pose = LinkPose(xyz=list(self.params["payload_pos"]["xy"]) + [np.sign(self.params["payload_pos"]["z_sign"])*(self.body.h + self.payload.h) / 2])
        self.arms_pose = [LinkPose(alpha=self.arm_angles[i], xyz=self.arms_coord[:, i]) 
                            for i in range(self.motors_num)]
        self.motors_pos = [LinkPose(xyz=self.motors_coord[:, i]) 
                            for i in range(self.motors_num)]
        self.props_pos = [LinkPose(xyz=self.props_coord[:, i]) 
                    for i in range(self.motors_num)]
        
        self.poses = [self.body_pose, self.payload_pose] + self.arms_pose + self.motors_pos + self.props_pos

        # Recomputing the center of mass of the new system of bodies
        masses = [link.m for link in self.links]
        self.com = sum([ masses[i] * pose.xyz for i, pose in enumerate(self.poses)]) / self.m

        # Recomputing corrections on posess with the respect to the new system
        self.poses_init = copy.deepcopy(self.poses)
        for pose in self.poses:
            pose.xyz -= self.com
        
        if verbose:
            print("Initial poses: ")
            [print(pose.xyz) for pose in self.poses_init]
            print("###################################")
            print("Final poses: ")
            [print(pose.xyz) for pose in self.poses]
            print("###################################")

        # Computing inertias
        self.links_I = []
        for link_i, link in enumerate(self.links):
            I_rot = rotate_I(I=link.I_com, R=self.poses[link_i].R)
            I_trans = translate_I(I=I_rot, m=link.m, xyz=self.poses[link_i].xyz)
            self.links_I.append(I_trans)
        
        # Total inertia
        self.I_com = sum(self.links_I)

        # Propeller poses
        self.prop_pos = np.array([pose.xyz for pose in self.motors_pos])
    
    @property
    def m(self):
        return np.sum([link.m for link in self.links]) 


if __name__ == "__main__":
    import time
    start_time = time.time()
    import argparse
    import yaml
    from gym_art.quadrotor.quad_models import *

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c',"--config",
        help="Config file to test"
    )
    args = parser.parse_args()

    def report(quad):
        print("Time:", time.time()-start_time)
        print("Quad inertia: \n", quad.I_com)
        print("Quad mass:", quad.m)
        print("Quad arm_xyz:", quad.arm_xyz)
        print("Quad COM: ", quad.com)
        print("Quad arm_length: ", quad.arm_length)
        print("Quad prop_pos: \n", quad.prop_pos, "shape:", quad.prop_pos.shape)

    ## CrazyFlie parameters
    # params = {}
    # params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
    # params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
    # params["arms"] = {"l": 0.022, "w":0.005, "h":0.005, "m":0.001}
    # params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
    # params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}
    
    # params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
    # params["arms_pos"] = {"angle": 45., "z": 0.}
    # params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    quad_crazyflie = QuadLink(params=crazyflie_params()["geom"], verbose=True)
    print("Crazyflie: ")
    report(quad_crazyflie)

    ## Aztec params
    geom_params = {}
    geom_params["body"] = {"l": 0.1, "w": 0.1, "h": 0.085, "m": 0.5}
    geom_params["payload"] = {"l": 0.12, "w": 0.12, "h": 0.04, "m": 0.1}
    geom_params["arms"] = {"l": 0.1, "w":0.015, "h":0.015, "m":0.025} #0.17 total arm
    geom_params["motors"] = {"h":0.02, "r":0.025, "m":0.02}
    geom_params["propellers"] = {"h":0.01, "r":0.1, "m":0.009}
    
    geom_params["motor_pos"] = {"xyz": [0.12, 0.12, 0.]}
    geom_params["arms_pos"] = {"angle": 45., "z": 0.}
    geom_params["payload_pos"] = {"xy": [0., 0.], "z_sign": -1}

    quad = QuadLink(params=geom_params, verbose=True)
    print("Aztec: ")
    report(quad)

    ## Crazyflie with lowered inertia
    quad = QuadLink(params=crazyflie_lowinertia_params()["geom"], verbose=True)
    print("Crazyflie lowered inertia: ")
    print("factor: ", quad_crazyflie.I_com / quad.I_com)
    report(quad)


    ## Random params
    if args.config is not None:
        yaml_stream = open(args.config, 'r')
        params_load = yaml.load(yaml_stream)

        quad_load = QuadLink(params=params_load, verbose=True)
        print("Loaded quad: %s" % args.config)
        report(quad_load)



################################################
## BUGS

#   geom:
#    body:
#      l: 0.03606089911004016
#      w: 0.0335657274378426
#      h: 0.006102479549661156
#      m: 0.0062767052677894074
#    payload:
#      l: 0.03838432440273057
#      w: 0.023816859339232426
#      h: 0.0070768000745466235
#      m: 0.011730161186989083
#    arms:
#      l: 0.03186274210023081
#      w: 0.004925117851085423
#      h: 0.003791035497312349
#      m: 0.0006186316884703438
#    motors:
#      h: 0.014114172356543832
#      r: 0.004954090517124884
#      m: 0.00019911121838799071
#    propellers:
#      h: 0.0003473493979756195
#      r: 0.018981731819806787
#      m: 0.0010610363239981538
#    motor_pos:
#      xyz: [0.0371103 0.0300385 0.       ]
#    arms_pos:
#      angle: 0.0
#      z: 0.0
#    payload_pos:
#      xy: [0. 0.]
#      z_sign: 0.9631780811491029
#  damp:
#    vel: 0.0013677304461958925
#    omega_quadratic: 0.013233401772593535
#  noise:
#    thrust_noise_ratio: 0.00970067152725083
#  motor:
#    thrust_to_weight: 2.928398587847958
#    torque_to_thrust: 0.07219217705972501

# self.dynamics.inertia
# array([7.80390772e-06,            nan,            nan])
