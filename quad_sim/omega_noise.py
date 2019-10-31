#!/usr/bin/env python
import numpy as np
from numpy.random import normal 
from numpy.random import uniform
import matplotlib.pyplot as plt
from math import exp


'''
[w, x, y, z]
'''
def quat2rot(quat):
	assert quat.shape == (4,)

	rot = np.zeros((3, 3))

	rot[0][0] = 1 - 2 * quat[2] * quat[2] - 2 * quat[3] * quat[3]
	rot[0][1] = 2 * quat[1] * quat[2] - 2 * quat[3] * quat[0]
	rot[0][2] = 2 * quat[1] * quat[3] + 2 * quat[2] * quat[0]
	rot[1][0] = 2 * quat[1] * quat[2] + 2 * quat[3] * quat[0]
	rot[1][1] = 1 - 2 * quat[1] * quat[1] - 2 * quat[3] * quat[3]
	rot[1][2] = 2 * quat[2] * quat[3] - 2 * quat[1] * quat[0]
	rot[2][0] = 2 * quat[1] * quat[3] - 2 * quat[2] * quat[0]
	rot[2][1] = 2 * quat[2] * quat[3] + 2 * quat[1] * quat[0]
	rot[2][2] = 1 - 2 * quat[1] * quat[1] - 2 * quat[2] * quat[2]

	return rot


'''
http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
'''
def rot2quat(rot):
	assert rot.shape == (3, 3)

	trace = np.trace(rot)
	if trace > 0:
		S = (trace + 1.0)**0.5 * 2
		qw = 0.25 * S
		qx = (rot[2][1] - rot[1][2]) / S 
		qy = (rot[0][2] - rot[2][0]) / S
		qz = (rot[1][0] - rot[0][1]) / S
	elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
		S = (1.0 + rot[0][0] - rot[1][1] - rot[2][2])**0.5 * 2
		qw = (rot[2][1] - rot[1][2]) / S
		qx = 0.25 * S 
		qy = (rot[0][1] + rot[1][0]) / S
		qz = (rot[0][2] + rot[2][0]) / S
	elif rot[1][1] > rot[2][2]:
		S = (1.0 + rot[1][1] - rot[0][0] - rot[2][2])**0.5 * 2
		qw = (rot[0][2] - rot[2][0]) / S 
		qx = (rot[0][1] + rot[1][0]) / S
		qy = 0.25 * S
		qz = (rot[1][2] + rot[2][1]) / S
	else:
		S = (1.0 + rot[2][2] - rot[0][0] - rot[1][1])**0.5 * 2
		qw = (rot[1][0] - rot[0][1]) / S
		qx = (rot[0][2] + rot[2][0]) / S 
		qy = (rot[1][2] + rot[2][1]) / S
		qz = 0.25 * S

	return np.array([qw, qx, qy, qz])

def quat_from_small_angle(theta):
	assert theta.shape == (3,)

	q_squared = np.linalg.norm(theta)**2 / 4.0
	if q_squared < 1:
		q_theta = np.array([(1 - q_squared)**0.5, theta[0] * 0.5, theta[1] * 0.5, theta[2] * 0.5])
	else:
		w = 1.0 / (1 + q_squared)**0.5
		f = 0.5 * w
		q_theta = np.array([w, theta[0] * f, theta[1] * f, theta[2] * f])

	q_theta = q_theta / np.linalg.norm(q_theta)

	return q_theta

class Noise:
	def __init__(self, noise_normal_position=np.zeros(3), noise_uniform_position=np.zeros(3), 
					   noise_normal_linear_velocity=np.zeros(3), noise_uniform_linear_velocity=np.zeros(3), 
					   noise_normal_theta=np.zeros(3), noise_uniform_theta=np.zeros(3), 
					   noise_density=0.000175, random_walk=0.0105, bias_correlation_time=1000, turn_on_bias_sigma=0.09,
					   measurement_delay=0): 


		"""
		Args:
		noise_normal_position: numpy array of 3 elements representing the 
				standard deviation of the position noise, the noise is centered at 0
		noise_uniform_position: numpy array of 3 elements representing the upper range 
				of the uniform distribution
		noise_normal_linear_velocity: numpy array of 3 elements representing the 
				standard deviation of the linear velocity noise, the noise is centered at 0
		noise_uniform_linear_velocity: numpy array of 3 elements representing the upper range 
				of the uniform distribution
		noise_normal_theta: numpy array of 3 elements representing the 
				standard deviation of the orientation noise, the noise is centered at 0
		noise_uniform_theta: numpy array of 3 elements representing the upper range 
				of the uniform distribution
		noise_density: gyroscope noise, MPU-9250 spec
		random_walk: gyroscope noise, MPU-9250 spec
		bias_correlation_time: gyroscope noise, MPU-9250 spec
		turn_on_bias_sigma: gyroscope noise, MPU-9250 spec
		measurement_delay: integer, # time steps to delay messages 
		"""

		self.noise_normal_position = noise_normal_position
		self.noise_uniform_position = noise_uniform_position

		self.noise_normal_linear_velocity = noise_normal_linear_velocity
		self.noise_uniform_linear_velocity = noise_uniform_linear_velocity

		self.noise_normal_theta = noise_uniform_theta
		self.noise_uniform_theta = noise_uniform_theta

		self.noise_density = noise_density
		self.random_walk = random_walk
		self.bias_correlation_time = bias_correlation_time
		self.turn_on_bias_sigma = turn_on_bias_sigma
		self.gyroscope_bias = np.zeros(3)

		self.measurement_delay = measurement_delay

		self.previous_pos = np.zeros(3)
		self.previous_vel = np.zeros(3)
		self.previous_rot = np.zeros((3, 3))
		self.previous_omega = np.zeros(3)

		self.states_queue = []

		self.omega_noise_comp = []
	

	"""
	args: 
	pos: ground truth of the position
	vel: grond truth if the linear velocity
	rot: ground truth of the orientation in rotational matrix
	omega: ground truth of the angular velocity
	dt: integration step
	"""
	def add_noise(self, pos, vel, rot, omega, dt):
		assert pos.shape == (3,)
		assert vel.shape == (3,)
		assert rot.shape == (3,3)
		assert omega.shape == (3,)

		# add noise to position measurement
		noisy_pos = np.zeros(3)
		noisy_pos[0] = pos[0] + \
					   normal(0, self.noise_normal_position[0]) + \
					   uniform(-self.noise_uniform_position[0], self.noise_uniform_position[0])
		noisy_pos[1] = pos[1] + \
					   normal(0, self.noise_normal_position[1]) + \
					   uniform(-self.noise_uniform_position[1], self.noise_uniform_position[1])
		noisy_pos[2] = pos[2] + \
					   normal(0, self.noise_normal_position[2]) + \
					   uniform(-self.noise_uniform_position[2], self.noise_uniform_position[2])

		# add noise to linear velocity
		noisy_vel = np.zeros(3)		
		noisy_vel[0] = vel[0] + \
					   normal(0, self.noise_normal_linear_velocity[0]) + \
					   uniform(-self.noise_uniform_linear_velocity[0], self.noise_uniform_linear_velocity[0])
		noisy_vel[1] = vel[1] + \
					   normal(0, self.noise_normal_linear_velocity[1]) + \
					   uniform(-self.noise_uniform_linear_velocity[1], self.noise_uniform_linear_velocity[1])
		noisy_vel[2] = vel[2] + \
					   normal(0, self.noise_normal_linear_velocity[2]) + \
					   uniform(-self.noise_uniform_linear_velocity[2], self.noise_uniform_linear_velocity[2])

		# add noise to orientation
		quat = rot2quat(rot)

		theta = np.zeros(3)
		theta[0] = normal(0, self.noise_normal_theta[0]) + uniform(-self.noise_uniform_theta[0], self.noise_uniform_theta[0])
		theta[1] = normal(0, self.noise_normal_theta[1]) + uniform(-self.noise_uniform_theta[1], self.noise_uniform_theta[1])
		theta[2] = normal(0, self.noise_normal_theta[2]) + uniform(-self.noise_uniform_theta[2], self.noise_uniform_theta[2])

		# convert theta to quaternion
		quat_theta = quat_from_small_angle(theta)

		noisy_quat = np.zeros(4)
		## quat * quat_theta
		noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[3] 
		noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[2] 
		noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[1] 
		noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[0]

		# TODO: make sure the rotational matrix is orthogonal
		noisy_rot = quat2rot(noisy_quat)

		noisy_omega = self.add_noise_to_omega(omega, dt)

		self.states_queue.append((noisy_pos, noisy_vel, noisy_rot, noisy_omega))

		if self.measurement_delay > len(self.states_queue):
			return np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3)

		return self.states_queue.pop(0)

	## copy from rotorS imu plugin
	def add_noise_to_omega(self, omega, dt):
		assert omega.shape == (3,)

		sigma_g_d = self.noise_density / (dt**0.5)
		sigma_b_g_d = (-(sigma_g_d**2) * (self.bias_correlation_time / 2) * (exp(-2*dt/self.bias_correlation_time) - 1))**0.5
		pi_g_d = exp(-dt / self.bias_correlation_time)

		self.gyroscope_bias = pi_g_d * self.gyroscope_bias + sigma_b_g_d * normal(0, 1, 3)

		# noise = self.gyroscope_bias + self.random_walk * normal(0, 1, 3) + self.turn_on_bias_sigma * normal(0, 1, 3)
		noise = self.gyroscope_bias + self.random_walk * normal(0, 1, 3) #+ self.turn_on_bias_sigma * normal(0, 1, 3)
		noise_comp = np.array([np.linalg.norm(self.gyroscope_bias), np.linalg.norm(self.random_walk * normal(0, 1, 3)), np.linalg.norm(self.turn_on_bias_sigma * normal(0, 1, 3))]) / np.linalg.norm(noise)
		
		self.omega_noise_comp.append(np.abs(noise_comp))
		print("Noise components: ", np.mean( np.array(self.omega_noise_comp), axis=0))

		omega = omega + noise
		return omega



def test():
	nnp = np.array([0.05, 0.05, 0.05])
	nup = np.array([0.05, 0.05, 0.05])

	nnlv = np.array([0.05, 0.05, 0.05])
	nulv = np.array([0.05, 0.05, 0.05])

	nnt = np.array([0.1, 0.1, 0.1])
	nut = np.array([0.1, 0.1, 0.1])

	noise_generator = Noise(noise_normal_position=nnp, noise_uniform_position=nup, 
					   noise_normal_linear_velocity=nnlv, noise_uniform_linear_velocity=nulv, 
					   noise_normal_theta=nnt, noise_uniform_theta=nut, measurement_delay=10)
	dt = 0.005

	n = 1000

	ground_truth_pos_x = np.sin(np.linspace(-20, 20, num=n))
	ground_truth_pos_y = np.sin(np.linspace(-20, 20, num=n))
	ground_truth_pos_z = np.sin(np.linspace(-20, 20, num=n))

	ground_truth_vel_x = np.cos(np.linspace(-20, 20, num=n))
	ground_truth_vel_y = np.cos(np.linspace(-20, 20, num=n))
	ground_truth_vel_z = np.cos(np.linspace(-20, 20, num=n))

	ground_truth_omega_x = np.zeros(n)
	ground_truth_omega_y = np.zeros(n)
	ground_truth_omega_z = np.zeros(n)

	rot = np.eye(3)
	ground_truth_orientation_x = np.sin(np.linspace(-20, 20, num=n)) # np.ones(n)
	ground_truth_orientation_y = np.cos(np.linspace(-20, 20, num=n)) # np.ones(n)
	ground_truth_orientation_z = np.sin(np.linspace(-20, 20, num=n)) # np.ones(n)
	

	ground_truth_pos = np.column_stack((ground_truth_pos_x, ground_truth_pos_y, ground_truth_pos_z))
	ground_truth_vel = np.column_stack((ground_truth_vel_x, ground_truth_vel_y, ground_truth_vel_z))
	ground_truth_omega = np.column_stack((ground_truth_omega_x, ground_truth_omega_y, ground_truth_omega_z))
	ground_truth_orientation = np.column_stack((ground_truth_orientation_x, ground_truth_orientation_y, ground_truth_orientation_z))
	## normalize the orientation
	norms = np.linalg.norm(ground_truth_orientation, axis=1)
	ground_truth_orientation = ground_truth_orientation / norms[:, None]
	print(ground_truth_orientation)

	for i in range(n):
		noisy_pos, noisy_vel, noisy_rot, noisy_omega = noise_generator.add_noise(
			ground_truth_pos[i], ground_truth_vel[i], rot, ground_truth_omega[i], dt)

		# test if the noisy rotation matrix is orthogonal
		if np.allclose(noisy_rot.T, np.linalg.inv(noisy_rot), 1e-5) != True:
			print('Non-orthogonal rotation matrix:')
			print(noisy_rot.T)
			print(np.linalg.inv(noisy_rot))
			exit(1)

		noisy_orientation = np.matmul(noisy_rot, ground_truth_orientation[i])

		if i == 0:
			noisy_pos_ = np.array([noisy_pos])
			noisy_vel_ = np.array([noisy_vel])
			noisy_omega_ = np.array([noisy_omega])
			noisy_orientation_ = np.array([noisy_orientation])
		else:
			noisy_pos_ = np.concatenate((noisy_pos_, np.array([noisy_pos])))
			noisy_vel_ = np.concatenate((noisy_vel_, np.array([noisy_vel])))
			noisy_omega_ = np.concatenate((noisy_omega_, np.array([noisy_omega])))
			noisy_orientation_ = np.concatenate((noisy_orientation_, np.array([noisy_orientation])))

	## TODO: plot
	plt.subplot(431)
	plt.ylabel('pos x')
	plt.plot(range(n), noisy_pos_[:, 0], c='red')
	plt.plot(range(n), ground_truth_pos[:, 0], c='green')
	plt.subplot(432)
	plt.ylabel('pos y')
	plt.plot(range(n), noisy_pos_[:, 1], c='red')
	plt.plot(range(n), ground_truth_pos[:, 1], c='green')
	plt.subplot(433)
	plt.ylabel('pos z')
	plt.plot(range(n), noisy_pos_[:, 2], c='red')
	plt.plot(range(n), ground_truth_pos[:, 2], c='green')

	plt.subplot(434)
	plt.ylabel('vel x')
	plt.plot(range(n), noisy_vel_[:, 0], c='red')
	plt.plot(range(n), ground_truth_vel[:, 0], c='green')
	plt.subplot(435)
	plt.ylabel('vel y')
	plt.plot(range(n), noisy_vel_[:, 1], c='red')
	plt.plot(range(n), ground_truth_vel[:, 1], c='green')
	plt.subplot(436)
	plt.ylabel('vel z')
	plt.plot(range(n), noisy_vel_[:, 2], c='red')
	plt.plot(range(n), ground_truth_vel[:, 2], c='green')

	plt.subplot(437)
	plt.ylabel('omega x')
	plt.plot(range(n), noisy_omega_[:, 0], c='red')
	plt.plot(range(n), ground_truth_omega[:, 0], c='green')
	plt.subplot(438)
	plt.ylabel('omega y')
	plt.plot(range(n), noisy_omega_[:, 1], c='red')
	plt.plot(range(n), ground_truth_omega[:, 1], c='green')
	plt.subplot(439)
	plt.ylabel('omega z')
	plt.plot(range(n), noisy_omega_[:, 2], c='red')
	plt.plot(range(n), ground_truth_omega[:, 2], c='green')

	plt.subplot(4, 3, 10)
	plt.ylabel('orientation in x')
	plt.plot(range(n), noisy_orientation_[:, 0], c='red')
	plt.plot(range(n), ground_truth_orientation[:, 0], c='green')
	plt.subplot(4, 3, 11)
	plt.ylabel('orientation in y')
	plt.plot(range(n), noisy_orientation_[:, 1], c='red')
	plt.plot(range(n), ground_truth_orientation[:, 1], c='green')
	plt.subplot(4, 3, 12)
	plt.ylabel('orientation in z')
	plt.plot(range(n), noisy_orientation_[:, 2], c='red')
	plt.plot(range(n), ground_truth_orientation[:, 2], c='green')

	plt.show()

if __name__ == '__main__':
	test()