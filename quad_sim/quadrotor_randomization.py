import numpy as np
from numpy.linalg import norm
from copy import deepcopy

from gym_art.quadrotor.quad_utils import *
from gym_art.quadrotor.quad_models import *

def clip_params_positive(params):
    def clip_positive(key, item):
        return np.clip(item, a_min=0., a_max=None)
    walk_dict(params, clip_positive)
    return params

def check_quad_param_limits(params, params_init=None):
    ## Body parameters (like lengths and masses) are always positive
    for key in ["body", "payload", "arms", "motors", "propellers"]:
        params["geom"][key] = clip_params_positive(params["geom"][key])

    params["geom"]["motor_pos"]["xyz"][:2] = np.clip(params["geom"]["motor_pos"]["xyz"][:2], a_min=0.005, a_max=None)
    body_w = params["geom"]["body"]["w"]
    params["geom"]["payload_pos"]["xy"] = np.clip(params["geom"]["payload_pos"]["xy"], a_min=-body_w/4., a_max=body_w/4.)    
    params["geom"]["arms_pos"]["angle"] = np.clip(params["geom"]["arms_pos"]["angle"], a_min=0., a_max=90.)    
    
    ## Damping parameters
    params["damp"]["vel"] = np.clip(params["damp"]["vel"], a_min=0.00000, a_max=1.)
    params["damp"]["omega_quadratic"] = np.clip(params["damp"]["omega_quadratic"], a_min=0.00000, a_max=1.)
    
    ## Motor parameters
    params["motor"]["thrust_to_weight"] = np.clip(params["motor"]["thrust_to_weight"], a_min=1.2, a_max=None)
    params["motor"]["torque_to_thrust"] = np.clip(params["motor"]["torque_to_thrust"], a_min=0.001, a_max=1.)
    params["motor"]["linearity"] = np.clip(params["motor"]["linearity"], a_min=0., a_max=1.)
    params["motor"]["assymetry"] = np.clip(params["motor"]["assymetry"], a_min=0.9, a_max=1.1)
    params["motor"]["C_drag"] = np.clip(params["motor"]["C_drag"], a_min=0., a_max=None)
    params["motor"]["C_roll"] = np.clip(params["motor"]["C_roll"], a_min=0., a_max=None)
    params["motor"]["damp_time_up"] = np.clip(params["motor"]["damp_time_up"], a_min=0., a_max=None)
    params["motor"]["damp_time_down"] = np.clip(params["motor"]["damp_time_down"], a_min=0., a_max=None)

    ## Make sure propellers make sense in size
    if params_init is not None:
        r0 = params_init["geom"]["propellers"]["r"]
        t2w, t2w0 = params_init["motor"]["thrust_to_weight"], params["motor"]["thrust_to_weight"]
        params["geom"]["propellers"]["r"] = r0 * (t2w/t2w0)**0.5

    return params

def get_dyn_randomization_params(quad_params, noise_ratio=0., noise_ratio_params=None):
    """
    The function updates noise params
    Args:
        noise_ratio (float): ratio of change relative to the nominal values
        noise_ratio_params (dict): if for some parameters you want to have different ratios relative to noise_ratio,
            you can provided it through this dictionary
    Returns:
        noise_params dictionary
    """
    ## Setting the initial noise ratios (nominal ones)
    noise_params = deepcopy(quad_params)
    def set_noise_ratio(key, item):
        if isinstance(item, str):
            return None
        else:
            return noise_ratio
    
    walk_dict(noise_params, set_noise_ratio)

    ## Updating noise ratios
    if noise_ratio_params is not None:
        # noise_params.update(noise_ratio_params)
        dict_update_existing(noise_params, noise_ratio_params)
    return noise_params


def perturb_dyn_parameters(params, noise_params, sampler="normal"):
    """
    The function samples around nominal parameters provided noise parameters
    Args:
        params (dict): dictionary of quadrotor parameters
        noise_params (dict): dictionary of noise parameters with the same hierarchy as params, but
            contains ratio of deviation from the params
    Returns:
        dict: modified parameters
    """
    ## Sampling parameters
    def sample_normal(key, param_val, ratio):
        #2*ratio since 2std contain 98% of all samples
        param_val_sample = np.random.normal(loc=param_val, scale=np.abs((ratio/2)*np.array(param_val)))
        return param_val_sample, ratio
    
    def sample_uniform(key, param_val, ratio):
        param_val = np.array(param_val)
        return np.random.uniform(low=param_val - param_val*ratio, high=param_val + param_val*ratio), ratio

    sample_param = locals()["sample_" + sampler]

    params_new = deepcopy(params)
    walk_2dict(params_new, noise_params, sample_param)

    ## Fixing a few parameters if they go out of allowed limits
    params_new = check_quad_param_limits(params_new, params)
    # print_dic(params_new)

    return params_new

def sample_random_dyn():
    """
    The function samples parameters for all possible quadrotors
    Args:
        scale (float): scale of sampling
    Returns:
        dict: sampled quadrotor parameters
    """
    ###################################################################
    ## DENSITIES (body, payload, arms, motors, propellers)
    # Crazyflie estimated body / payload / arms / motors / props density: 1388.9 / 1785.7 / 1777.8 / 1948.8 / 246.6 kg/m^3
    # Hummingbird estimated body / payload / arms / motors/ props density: 588.2 / 173.6 / 1111.1 / 509.3 / 246.6 kg/m^3
    geom_params = {}
    dens_val = np.random.uniform(
        low=[500., 200., 500., 500., 200.], 
        high=[2000., 2000., 2000., 4500., 300.])
    
    geom_params["body"] = {"density": dens_val[0]}
    geom_params["payload"] = {"density": dens_val[1]}
    geom_params["arms"] = {"density": dens_val[2]}
    geom_params["motors"] = {"density": dens_val[3]}
    geom_params["propellers"] = {"density": dens_val[4]}

    ###################################################################
    ## GEOMETRIES
    # MOTORS (and overal size)
    total_w = np.random.uniform(low=0.05, high=0.2)
    total_l = np.clip(np.random.normal(loc=1., scale=0.1), a_min=1.0, a_max=None) * total_w
    motor_z = np.random.normal(loc=0., scale=total_w / 8.)
    geom_params["motor_pos"] = {"xyz": [total_w / 2., total_l / 2., motor_z]}
    geom_params["motors"]["r"] = total_w * np.random.normal(loc=0.1, scale=0.01)
    geom_params["motors"]["h"] = geom_params["motors"]["r"] * np.random.normal(loc=1.0, scale=0.05)
    
    # BODY
    w_low, w_high = 0.25, 0.5
    w_coeff = np.random.uniform(low=w_low, high=w_high)
    geom_params["body"]["w"] = w_coeff * total_w
    ## Promotes more elangeted bodies when they are more narrow
    l_scale = (1. - (w_coeff - w_low) / (w_high - w_low))
    geom_params["body"]["l"] =  np.clip(np.random.normal(loc=1., scale=l_scale), a_min=1.0, a_max=None) * geom_params["body"]["w"]
    geom_params["body"]["h"] =  np.random.uniform(low=0.1, high=1.5) * geom_params["body"]["w"]

    # PAYLOAD
    pl_scl = np.random.uniform(low=0.25, high=1.0, size=3)
    geom_params["payload"]["w"] =  pl_scl[0] * geom_params["body"]["w"]
    geom_params["payload"]["l"] =  pl_scl[1] * geom_params["body"]["l"]
    geom_params["payload"]["h"] =  pl_scl[2] * geom_params["body"]["h"]
    geom_params["payload_pos"] = {
            "xy": np.random.normal(loc=0., scale=geom_params["body"]["w"] / 10., size=2), 
            "z_sign": np.sign(np.random.uniform(low=-1, high=1))}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    # ARMS
    geom_params["arms"]["w"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["arms"]["h"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["arms_pos"] = {"angle": np.random.normal(loc=45., scale=10.), "z": motor_z - geom_params["motors"]["h"]/2.}
    
    # PROPS
    thrust_to_weight = np.random.uniform(low=1.8, high=2.5)
    # thrust_to_weight = np.random.uniform(low=2.3, high=2.5)
    geom_params["propellers"]["h"] = 0.01
    geom_params["propellers"]["r"] = (0.3) * total_w * (thrust_to_weight / 2.0)**0.5
    
    ## Damping parameters
    # damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    # damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    # damp_params = {
    #     "vel": 0.001 * damp_vel_scale, 
    #     "omega_quadratic": 0.015 * damp_omega_scale}
    damp_params = {
        "vel": 0.0, 
        "omega_quadratic": 0.0}

    ## Noise parameters
    noise_params = {}
    noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.01, high=0.05) #0.01
    
    ## Motor parameters
    damp_time_up = np.random.uniform(low=0.15, high=0.2)
    damp_time_down_scale = np.random.uniform(low=1.0, high=1.0)
    motor_params = {"thrust_to_weight" : thrust_to_weight,
                    "torque_to_thrust": np.random.uniform(low=0.005, high=0.025), #0.05 originally
                    "assymetry": np.random.uniform(low=0.9, high=1.1, size=4),
                    "linearity": 1.0,
                    "C_drag": 0.,
                    "C_roll": 0.,
                    "damp_time_up": damp_time_up,
                    "damp_time_down": damp_time_down_scale * damp_time_up
                    # "linearity": np.random.normal(loc=0.5, scale=0.1)
                    }

    ## Summarizing
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }

    ## Checking everything
    params = check_quad_param_limits(params=params)
    return params

def sample_random_dyn_nodelay():
    params = sample_random_dyn()
    params["motor"]["damp_time_up"] = 0.
    params["motor"]["damp_time_down"] = 0.
    return params

def sample_random_thrust2weight_15_25():
    params = sample_random_dyn()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=1.5, high=2.5)
    # params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.8, high=2.8)
    return params

def sample_random_thrust2weight_15_35():
    params = sample_random_dyn()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=1.5, high=3.5)
    return params

def sample_random_thrust2weight_20_30():
    params = sample_random_dyn()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.0, high=3.0)
    return params

def sample_random_thrust2weight_20_40():
    params = sample_random_dyn()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.0, high=4.0)
    return params

def sample_random_thrust2weight_20_50():
    params = sample_random_dyn()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.0, high=5.0)
    return params

def sample_random_with_linearity():
    params = sample_random_dyn()
    params["motor"]["linearity"] = np.random.uniform(low=0., high=1.)
    return params



def sample_crazyflie_thrust2weight_18_25():
    params = crazyflie_params()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=1.8, high=2.5)
    return params

def sample_crazyflie_thrust2weight_15_25():
    params = crazyflie_params()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=1.5, high=2.5)
    return params

def sample_crazyflie_thrust2weight_15_35():
    params = crazyflie_params()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=1.5, high=3.5)
    return params

def sample_crazyflie_thrust2weight_20_30():
    params = crazyflie_params()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.0, high=3.0)
    return params

def sample_crazyflie_thrust2weight_20_40():
    params = crazyflie_params()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.0, high=4.0)
    return params

def sample_crazyflie_thrust2weight_20_50():
    params = crazyflie_params()
    params["motor"]["thrust_to_weight"] = np.random.uniform(low=2.0, high=5.0)
    return params


def sample_random_dyn_lowinertia():
    """
    The function samples parameters for all possible quadrotors
    Args:
        scale (float): scale of sampling
    Returns:
        dict: sampled quadrotor parameters
    """
    ###################################################################
    ## DENSITIES (body, payload, arms, motors, propellers)
    # Crazyflie estimated body / payload / arms / motors / props density: 1388.9 / 1785.7 / 1777.8 / 1948.8 / 246.6 kg/m^3
    # Hummingbird estimated body / payload / arms / motors/ props density: 588.2 / 173.6 / 1111.1 / 509.3 / 246.6 kg/m^3
    geom_params = {}
    dens_val = np.random.uniform(
        low=[1500., 1500., 150., 150., 15.], 
        high=[2500., 2500., 250., 250., 25.])
    
    geom_params["body"] = {"density": dens_val[0]}
    geom_params["payload"] = {"density": dens_val[1]}
    geom_params["arms"] = {"density": dens_val[2]}
    geom_params["motors"] = {"density": dens_val[3]}
    geom_params["propellers"] = {"density": dens_val[4]}

    ###################################################################
    ## GEOMETRIES
    # MOTORS (and overal size)
    total_w = np.random.uniform(low=0.05, high=0.2)
    total_l = np.clip(np.random.normal(loc=1., scale=0.1), a_min=1.0, a_max=None) * total_w
    motor_z = np.random.normal(loc=0., scale=total_w / 8.)
    geom_params["motor_pos"] = {"xyz": [total_w / 2., total_l / 2., motor_z]}
    geom_params["motors"]["r"] = total_w * np.random.normal(loc=0.1, scale=0.01)
    geom_params["motors"]["h"] = geom_params["motors"]["r"] * np.random.normal(loc=1.0, scale=0.05)
    
    # BODY
    w_low, w_high = 0.2, 0.4
    w_coeff = np.random.uniform(low=w_low, high=w_high)
    geom_params["body"]["w"] = w_coeff * total_w
    ## Promotes more elangeted bodies when they are more narrow
    l_scale = (1. - (w_coeff - w_low) / (w_high - w_low))
    geom_params["body"]["l"] =  np.clip(np.random.normal(loc=1., scale=l_scale), a_min=1.0, a_max=2.0) * geom_params["body"]["w"]
    geom_params["body"]["h"] =  np.random.uniform(low=0.25, high=1.0) * geom_params["body"]["w"]

    # PAYLOAD
    pl_scl = np.random.uniform(low=0.50, high=1.0, size=2)
    pl_scl_h = np.random.uniform(low=0.25, high=0.75, size=1)
    geom_params["payload"]["w"] =  pl_scl[0] * geom_params["body"]["w"]
    geom_params["payload"]["l"] =  pl_scl[1] * geom_params["body"]["l"]
    geom_params["payload"]["h"] =  pl_scl_h[0] * geom_params["body"]["h"]
    geom_params["payload_pos"] = {
            "xy": np.random.normal(loc=0., scale=geom_params["body"]["w"] / 10., size=2), 
            "z_sign": np.sign(np.random.uniform(low=-1, high=1))}
    # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    # ARMS
    geom_params["arms"]["w"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["arms"]["h"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    geom_params["arms_pos"] = {"angle": np.random.normal(loc=45., scale=10.), "z": motor_z - geom_params["motors"]["h"]/2.}
    
    # PROPS
    thrust_to_weight = np.random.uniform(low=1.8, high=2.5)
    geom_params["propellers"]["h"] = 0.01
    geom_params["propellers"]["r"] = (0.3) * total_w * (thrust_to_weight / 2.0)**0.5
    
    ## Damping parameters
    # damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    # damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    # damp_params = {
    #     "vel": 0.001 * damp_vel_scale, 
    #     "omega_quadratic": 0.015 * damp_omega_scale}
    damp_params = {
        "vel": 0.0, 
        "omega_quadratic": 0.0}

    ## Noise parameters
    noise_params = {}
    noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.05, high=0.1) #0.01
    
    ## Motor parameters
    damp_time_up = np.random.uniform(low=0.1, high=0.2)
    damp_time_down_scale = np.random.uniform(low=1.0, high=2.0)
    motor_params = {"thrust_to_weight" : thrust_to_weight,
                    "torque_to_thrust": np.random.uniform(low=0.005, high=0.02), #0.05 originally
                    "assymetry": np.random.uniform(low=0.9, high=1.1, size=4),
                    "linearity": 1.0,
                    "C_drag": 0.,
                    "C_roll": 0.,
                    "damp_time_up": damp_time_up,
                    "damp_time_down": damp_time_down_scale * damp_time_up
                    # "linearity": np.random.normal(loc=0.5, scale=0.1)
                    }

    ## Summarizing
    params = {
        "geom": geom_params, 
        "damp": damp_params, 
        "noise": noise_params,
        "motor": motor_params
    }

    ## Checking everything
    params = check_quad_param_limits(params=params)
    return params



    # def sample_random_nondim_dyn():
    #     """
    #     The function samples parameters for all possible non-dimensional quadrotors
    #     Args:
    #         scale (float): scale of sampling
    #     Returns:
    #         dict: sampled quadrotor parameters
    #     """
    #     ###################################################################
    #     ## DENSITIES (body, payload, arms, motors, propellers)
    #     # Crazyflie estimated body / payload / arms / motors / props density: 1388.9 / 1785.7 / 1777.8 / 1948.8 / 246.6 kg/m^3
    #     # Hummingbird estimated body / payload / arms / motors/ props density: 588.2 / 173.6 / 1111.1 / 509.3 / 246.6 kg/m^3
    #     geom_params = {}
       
    #     geom_params["body"] = {"mass": 1.0}
    #     geom_params["payload"] = {"mass": 0}
    #     geom_params["arms"]    = {"mass": 0.}
    #     geom_params["motors"]  = {"mass": 0.}
    #     geom_params["propellers"] = {"mass": 0.}

    #     ###################################################################
    #     ## GEOMETRIES
    #     # MOTORS (and overal size)
    #     roll_authority = np.random.uniform(low=600, high=1200) #for our current low inertia CF ~ 1050
    #     pitch_authority = np.random.uniform(low=0.8, high=1.0) * roll_authority
    #     total_w = np.random.uniform(low=0.5, high=0.5)
    #     total_l = total_w
    #     motor_z = np.random.normal(loc=0., scale=total_w / 8.)
    #     geom_params["motor_pos"] = {"xyz": [total_w / 2., total_l / 2., motor_z]}
    #     geom_params["motors"]["r"] = total_w * np.random.normal(loc=0.1, scale=0.01)
    #     geom_params["motors"]["h"] = geom_params["motors"]["r"] * np.random.normal(loc=1.0, scale=0.05)

    #     # BODY
    #     geom_params["body"]["w"] = np.random.uniform(low=1.0, high=1.0)
    #     ## Promotes more elangeted bodies when they are more narrow
    #     geom_params["body"]["l"] =  np.random.uniform(low=1.0, high=2.0) * geom_params["body"]["w"]
    #     geom_params["body"]["h"] =  np.random.uniform(low=0.1, high=1.0) * geom_params["body"]["w"]
        


    #     # PAYLOAD
    #     pl_scl = np.random.uniform(low=0.25, high=1.0, size=3)
    #     geom_params["payload"]["w"] =  pl_scl[0] * geom_params["body"]["w"]
    #     geom_params["payload"]["l"] =  pl_scl[1] * geom_params["body"]["l"]
    #     geom_params["payload"]["h"] =  pl_scl[2] * geom_params["body"]["h"]
    #     geom_params["payload_pos"] = {
    #             "xy": np.random.normal(loc=0., scale=geom_params["body"]["w"] / 10., size=2), 
    #             "z_sign": np.sign(np.random.uniform(low=-1, high=1))}
    #     # z_sing corresponds to location (+1 - on top of the body, -1 - on the bottom of the body)

    #     # ARMS
    #     geom_params["arms"]["w"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    #     geom_params["arms"]["h"] = total_w * np.random.normal(loc=0.05, scale=0.005)
    #     geom_params["arms_pos"] = {"angle": np.random.normal(loc=45., scale=10.), "z": motor_z - geom_params["motors"]["h"]/2.}
        
    #     # PROPS
    #     thrust_to_weight = np.random.uniform(low=1.8, high=2.5)
    #     geom_params["propellers"]["h"] = 0.01
    #     geom_params["propellers"]["r"] = (0.3) * total_w * (thrust_to_weight / 2.0)**0.5
        
    #     ## Damping parameters
    #     # damp_vel_scale = np.random.uniform(low=0.01, high=2.)
    #     # damp_omega_scale = damp_vel_scale * np.random.uniform(low=0.75, high=1.25)
    #     # damp_params = {
    #     #     "vel": 0.001 * damp_vel_scale, 
    #     #     "omega_quadratic": 0.015 * damp_omega_scale}
    #     damp_params = {
    #         "vel": 0.0, 
    #         "omega_quadratic": 0.0}

    #     ## Noise parameters
    #     noise_params = {}
    #     noise_params["thrust_noise_ratio"] = np.random.uniform(low=0.01, high=0.05) #0.01
        
    #     ## Motor parameters
    #     damp_time_up = np.random.uniform(low=0.1, high=0.2)
    #     damp_time_down_scale = np.random.uniform(low=1.0, high=2.0)
    #     motor_params = {"thrust_to_weight" : thrust_to_weight,
    #                     "torque_to_thrust": np.random.uniform(low=0.005, high=0.025), #0.05 originally
    #                     "assymetry": np.random.uniform(low=0.9, high=1.1, size=4),
    #                     "linearity": 1.0,
    #                     "C_drag": 0.,
    #                     "C_roll": 0.,
    #                     "damp_time_up": damp_time_up,
    #                     "damp_time_down": damp_time_down_scale * damp_time_up
    #                     # "linearity": np.random.normal(loc=0.5, scale=0.1)
    #                     }

    #     ## Summarizing
    #     params = {
    #         "geom": geom_params, 
    #         "damp": damp_params, 
    #         "noise": noise_params,
    #         "motor": motor_params
    #     }

    #     ## Checking everything
    #     params = check_quad_param_limits(params=params)
    #     return params