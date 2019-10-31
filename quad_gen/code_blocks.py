#!/usr/bin/env python

headers_controller_nn = """
#include "math3d.h"
#include "stabilizer_types.h"
#include <math.h>
#include "controller_nn.h"

"""

headers_network_evaluate = """
#include "network_evaluate.h"

"""

constants = """

#define MAX_THRUST 0.1597
// PWM to thrust coefficients
#define A 2.130295e-11
#define B 1.032633e-6
#define C 5.484560e-4

"""

controller_init_function = """

void controllerNNInit(void) {}

"""

controller_test_function = """

bool controllerNNTest(void) {
	return true;
}

"""

linear_activation = """

float linear(float num) {
	return num;
}

"""

sigmoid_activation = """

float sigmoid(float num) {
	return 1 / (1 + exp(-num));
}

"""

relu_activation = """

float relu(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

"""

scaling = """
// range of action -1 ... 1, need to scale to range 0 .. 1
float scale(float v) {
	return 0.5f * (v + 1);
}

"""

clipping = """

float clip(float v, float min, float max) {
	if (v < min) return min;
	if (v > max) return max;
	return v;
}

"""

controller_entry = """
static control_t_n control_n;
static struct mat33 rot;
static float state_array[18];

void controllerNN(control_t *control, 
				  setpoint_t *setpoint, 
				  const sensorData_t *sensors, 
				  const state_t *state, 
				  const uint32_t tick)
{
	control->enableDirectThrust = true;
	if (!RATE_DO_EXECUTE(RATE_250_HZ, tick)) {
		return;
	}

	// Orientation
	struct quat q = mkquat(state->attitudeQuaternion.x, 
						   state->attitudeQuaternion.y, 
						   state->attitudeQuaternion.z, 
						   state->attitudeQuaternion.w);
	rot = quat2rotmat(q);

	// angular velocity
	float omega_roll = radians(sensors->gyro.x);
	float omega_pitch = radians(sensors->gyro.y);
	float omega_yaw = radians(sensors->gyro.z);

	// the state vector
	state_array[0] = state->position.x - setpoint->position.x;
	state_array[1] = state->position.y - setpoint->position.y;
	state_array[2] = state->position.z - setpoint->position.z;
	state_array[3] = state->velocity.x;
	state_array[4] = state->velocity.y;
	state_array[5] = state->velocity.z;
	state_array[6] = rot.m[0][0];
	state_array[7] = rot.m[0][1];
	state_array[8] = rot.m[0][2];
	state_array[9] = rot.m[1][0];
	state_array[10] = rot.m[1][1];
	state_array[11] = rot.m[1][2];
	state_array[12] = rot.m[2][0];
	state_array[13] = rot.m[2][1];
	state_array[14] = rot.m[2][2];
	state_array[15] = omega_roll;
	state_array[16] = omega_pitch;
	state_array[17] = omega_yaw;
	state_array[18] = state->position.z;


	// run the neural neural network
	networkEvaluate(&control_n, state_array);

	// convert thrusts to directly to PWM
	// need to hack the firmware (stablizer.c and power_distribution_stock.c)
	int PWM_0, PWM_1, PWM_2, PWM_3; 
	thrusts2PWM(&control_n, &PWM_0, &PWM_1, &PWM_2, &PWM_3);

	control->motorRatios[0] = PWM_0;
	control->motorRatios[1] = PWM_1;
	control->motorRatios[2] = PWM_2;
	control->motorRatios[3] = PWM_3;
}


/*
 * Crazyflie rotors positions (as opposed to what the neural network assumes): 
 *							x 
 *							|
 *						3	|	0
 *				 (thrust 0)	|	(thrust 1)
 *				  y <---------------
 *					 		|
 *					 	2	|	1
 *			     (thrust 3) |	(thrust 2)
 *	(thrust projection must align with each motor, e.g thrust 1 is on rotor 0)
 */
void thrusts2PWM(struct control_t_n *control_n, 
	int *PWM_0, int *PWM_1, int *PWM_2, int *PWM_3){

	/*
	// scaling and cliping
	control_n->thrust_0 = MAX_THRUST * clip(scale(control_n->thrust_0), 0.0, 1.0);
	control_n->thrust_1 = MAX_THRUST * clip(scale(control_n->thrust_1), 0.0, 1.0);
	control_n->thrust_2 = MAX_THRUST * clip(scale(control_n->thrust_2), 0.0, 1.0);
	control_n->thrust_3 = MAX_THRUST * clip(scale(control_n->thrust_3), 0.0, 1.0);

	// motor 0
	*PWM_0 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_1))) / (2 * A);
	// motor 1
	*PWM_1 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_2))) / (2 * A);
	// motor 2
	*PWM_2 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_3))) / (2 * A);
	// motor 3 
	*PWM_3 = (int)(-B + sqrt(B * B - 4 * A * (C - control_n->thrust_0))) / (2 * A);
	*/

	// scaling and cliping
	control_n->thrust_0 = UINT16_MAX * clip(scale(control_n->thrust_0), 0.0, 1.0);
	control_n->thrust_1 = UINT16_MAX * clip(scale(control_n->thrust_1), 0.0, 1.0);
	control_n->thrust_2 = UINT16_MAX * clip(scale(control_n->thrust_2), 0.0, 1.0);
	control_n->thrust_3 = UINT16_MAX * clip(scale(control_n->thrust_3), 0.0, 1.0);

	// motor 0
	*PWM_0 = control_n->thrust_0;
	// motor 1
	*PWM_1 = control_n->thrust_1;
	// motor 2
	*PWM_2 = control_n->thrust_2;
	// motor 3 
	*PWM_3 = control_n->thrust_3;
}

"""

log_group = """

LOG_GROUP_START(ctrlNN)
LOG_ADD(LOG_FLOAT, thrust0, &output_2[0])
LOG_ADD(LOG_FLOAT, thrust1, &output_2[1])
LOG_ADD(LOG_FLOAT, thrust2, &output_2[2])
LOG_ADD(LOG_FLOAT, thrust3, &output_2[3])
LOG_GROUP_STOP(ctrlNN)

"""