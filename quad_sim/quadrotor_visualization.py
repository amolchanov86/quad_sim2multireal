import numpy as np
from numpy.linalg import norm
import copy

import gym_art.quadrotor.rendering3d as r3d
from gym_art.quadrotor.quad_utils import *

# for visualization.
# a rough attempt at a reasonable third-person camera
# that looks "over the quadrotor's shoulder" from behind
class ChaseCamera(object):
    def __init__(self, view_dist=4):
        self.view_dist = view_dist

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


# for visualization.
# In case we have vertical control only we use a side view camera
class SideCamera(object):
    def __init__(self, view_dist):
        self.view_dist = view_dist

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        # eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        eye = self.pos_smooth + self.view_dist * np.array([0, 1, 0])
        center = self.pos_smooth
        return eye, center, up


# using our rendering3d.py to draw the scene in 3D.
# this class deals both with map and mapless cases.
class Quadrotor3DScene(object):
    def __init__(self, w, h,
        quad_arm=None, model=None, obstacles=True, visible=True, resizable=True, goal_diameter=None, viewpoint='chase', obs_hw=[64,64]):

        self.window_target = None
        self.window_w, self.window_h = w , h
        self.resizable = resizable
        self.viepoint = viewpoint
        self.obs_hw = copy.deepcopy(obs_hw)

        # self.world_box = 40.0
        self.quad_arm = quad_arm
        self.obstacles = obstacles
        self.model = model

        if goal_diameter:
            self.goal_forced_diameter = goal_diameter
        else:
            self.goal_forced_diameter = None
        self.update_goal_diameter()
        
        if self.viepoint == 'chase':
            self.chase_cam = ChaseCamera(view_dist=self.diameter * 15)
        elif self.viepoint == 'side':
            self.chase_cam = SideCamera(view_dist=self.diameter * 15)

        self.scene = None
        self.window_target = None
        self.obs_target = None
        self.video_target = None

    def update_goal_diameter(self):
        if self.quad_arm is not None:
            self.diameter = 2 * self.quad_arm
        else:
            self.diameter = 2 * np.linalg.norm(self.model.params["motor_pos"]["xyz"][:2])
         
        if self.goal_forced_diameter:
            self.goal_diameter = self.goal_forced_diameter
        else:
            self.goal_diameter = self.diameter


    def _make_scene(self):
        # if target is None:
        #     self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
        #     self.obs_target = r3d.FBOTarget(self.obs_hw[0], self.obs_hw[1])
        #     self.video_target = r3d.FBOTarget(self.window_h, self.window_h)

        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        if self.model is not None:
            self.quad_transform = self._quadrotor_3dmodel(self.model)
        else:
            self.quad_transform = self._quadrotor_simple_3dmodel(self.diameter)

        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*self.diameter, 32))

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.random_textype(), (0.15, 0.25),
            r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.update_goal_diameter()
        self.chase_cam.view_dist = self.diameter * 15

        self.create_goal(goal=(0,0,0))
        
        bodies = [r3d.BackToFront([floor, self.shadow_transform]),
            self.goal_transform, self.quad_transform] + self.goal_arrows
        
            

        if self.obstacles:
            bodies += self.obstacles.bodies

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)

        self.scene = r3d.Scene(batches=[batch], bgcolor=(0,0,0))
        self.scene.initialize()

    def create_goal(self, goal):
        ## Goal
        self.goal_transform = r3d.transform_and_color(np.eye(4),
            (0.85, 0.55, 0), r3d.sphere(self.goal_diameter/2, 18))
        
        goal_arr_len, goal_arr_r, goal_arr_sect  = 1.5 * self.goal_diameter, 0.02 * self.goal_diameter, 10
        self.goal_arrows = []

        self.goal_arrows_rot = []
        self.goal_arrows_rot.append(np.array([[0,0,1],[0,1,0],[-1,0,0]]))
        self.goal_arrows_rot.append(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
        self.goal_arrows_rot.append(np.eye(3))

        self.goal_arrows.append(r3d.transform_and_color(
            np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]), 
            (1., 0., 0.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))
        self.goal_arrows.append(r3d.transform_and_color(
            np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]), 
            (0., 1., 0.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))
        self.goal_arrows.append(r3d.transform_and_color(
            np.eye(4), 
            (0., 0., 1.), r3d.arrow(goal_arr_r, goal_arr_len, goal_arr_sect)))

    def update_goal(self, goal):
        self.goal_transform.set_transform(r3d.translate(goal[0:3]))

        self.goal_arrows[0].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[0]))
        self.goal_arrows[1].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[1]))
        self.goal_arrows[2].set_transform(r3d.trans_and_rot(goal[0:3], self.goal_arrows_rot[2]))


    def update_model(self, model):
        self.model = model
        if self.video_target is not None:
            self.video_target.finish()
            self.video_target = None
        if self.obs_target is not None:
            self.obs_target.finish()
            self.obs_target = None
        if self.window_target:
            self._make_scene()

    def _quadrotor_3dmodel(self, model):
        # params["body"] = {"l": 0.03, "w": 0.03, "h": 0.004, "m": 0.005}
        # params["payload"] = {"l": 0.035, "w": 0.02, "h": 0.008, "m": 0.01}
        # params["arms"] = {"l": 0.022, "w":0.005, "h":0.005, "m":0.001}
        # params["motors"] = {"h":0.02, "r":0.0035, "m":0.0015}
        # params["propellers"] = {"h":0.002, "r":0.022, "m":0.00075}
        
        # params["motor_pos"] = {"xyz": [0.065/2, 0.065/2, 0.]}
        # params["arms_pos"] = {"angle": 45., "z": 0.}
        # params["payload_pos"] = {"xy": [0., 0.], "z_sign": 1}

        ## PROPELLERS 
        # "X" propeller configuration, start fwd left, go clockwise
        # IDs: https://wiki.bitcraze.io/projects:crazyflie2:userguide:assembly
        link_colors = {
            "body": (0.67843137, 1. , 0.18431373),
            "payload": (0., 0., 1.),
            "prop_0":(1,0,0), "prop_1":(0,1,0), "prop_2":(0,1,0), "prop_3": (1,0,0),
            "motor_0":(0,0,0), "motor_1":(0,0,0), "motor_2":(0,0,0), "motor_3": (0,0,0),
            "arm_0":(0,0,1), "arm_1":(0,0,1), "arm_2":(0,0,1), "arm_3": (0,0,1),
            }
        
        links = []
        for i, link in enumerate(model.links):
            xyz, R, color = model.poses[i].xyz, model.poses[i].R, link_colors[link.name]
            rot = np.eye(4)
            rot[:3,:3] = R
            # print("LINK: ", link.name, "R:", rot, end=" ")
            if link.name[:4] == "prop":
                prop_r = link.r
                color = 0.5 * np.array(color) + 0.2
            if link.type == "box":
                # print("Type: Box")
                link_transf = r3d.transform_and_color(
                    np.matmul(r3d.translate(xyz), rot), color, 
                              r3d.box(link.l, link.w, link.h))
            elif link.type == "cylinder":
                # print("Type: Cylinder")
                link_transf = r3d.transform_and_color(r3d.translate(xyz), color,
                    r3d.cylinder(link.r, link.h, 32))
            links.append(link_transf)


        ## ARROWS
        arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12*prop_r, 2.5*prop_r, 16))
        links.append(arrow)

        self.have_state = False
        return r3d.Transform(np.eye(4), links)

    def _quadrotor_simple_3dmodel(self, diam):
        r = diam / 2
        prop_r = 0.3 * diam
        prop_h = prop_r / 15.0

        # "X" propeller configuration, start fwd left, go clockwise
        rr = r * np.sqrt(2)/2
        deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
        colors = ((1,0,0), (1,0,0), (0,1,0), (0,1,0))
        def disc(translation, color):
            color = 0.5 * np.array(list(color)) + 0.2
            disc = r3d.transform_and_color(r3d.translate(translation), color,
                r3d.cylinder(prop_r, prop_h, 32))
            return disc
        props = [disc(d, c) for d, c in zip(deltas, colors)]

        arm_thicc = diam / 20.0
        arm_color = (0.6, 0.6, 0.6)
        arms = r3d.transform_and_color(
            np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
            [r3d.box(diam/10, diam, arm_thicc), r3d.box(diam, diam/10, arm_thicc)])

        arrow = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12*prop_r, 2.5*prop_r, 16))

        bodies = props + [arms, arrow]
        self.have_state = False
        return r3d.Transform(np.eye(4), bodies)

    # TODO allow resampling obstacles?
    def reset(self, goal, dynamics):
        self.chase_cam.reset(goal[0:3], dynamics.pos, dynamics.vel)
        self.update_state(dynamics, goal)

    def update_state(self, dynamics, goal):
        if self.scene:
            self.chase_cam.step(dynamics.pos, dynamics.vel)
            self.have_state = True
            self.fpv_lookat = dynamics.look_at()
            
            self.update_goal(goal=goal)

            matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
            self.quad_transform.set_transform_nocollide(matrix)

            shadow_pos = 0 + dynamics.pos
            shadow_pos[2] = 0.001 # avoid z-fighting
            matrix = r3d.translate(shadow_pos)
            self.shadow_transform.set_transform_nocollide(matrix)

    def render_chase(self, dynamics, goal, mode="human"):
        if mode == "human":
            if self.window_target is None: 
                self.window_target = r3d.WindowTarget(self.window_w, self.window_h, resizable=self.resizable)
                self._make_scene()
            self.update_state(dynamics=dynamics, goal=goal)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.window_target)
            return None
        elif mode == "rgb_array":
            if self.video_target is None:
                self.video_target = r3d.FBOTarget(self.window_h, self.window_h)
                self._make_scene()
            self.update_state(dynamics=dynamics, goal=goal)
            self.cam3p.look_at(*self.chase_cam.look_at())
            r3d.draw(self.scene, self.cam3p, self.video_target)
            return np.flipud(self.video_target.read())

    def render_obs(self, dynamics, goal):
        if self.obs_target is None: 
            self.obs_target = r3d.FBOTarget(self.obs_hw[0], self.obs_hw[1])
            self._make_scene()
        self.update_state(dynamics=dynamics, goal=goal)
        self.cam1p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam1p, self.obs_target)
        return np.flipud(self.obs_target.read())