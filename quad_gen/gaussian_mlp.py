import argparse
import numpy as np 
import os
# import tensorflow as tf

from code_blocks import (
	headers_network_evaluate,
	linear_activation,
	sigmoid_activation,
	relu_activation,
)

def generate(policy, sess, output_path=None):
	"""
	Generate mlp model source code given a policy object
	Args:
		policy [policy object]: the trained policy
		sess [tf.Session] a tensorflow session
		output_path [str]: the path of the generated code (should include the file name)
	"""
	# TODO: check if the policy is really a mlp policy
	trainable_list = policy.get_params()

	trainable_shapes = []
	trainable_evals = []
	for tf_trainable in trainable_list:
		trainable_shapes.append(tf_trainable.shape)
		trainable_evals.append(tf_trainable.eval(session=sess))

	"""
	To account for the last matrix which stores the std, 
	the # of layers must be subtracted by 1
	"""
	n_layers = len(trainable_shapes) - 1
	weights = []	# strings
	biases = []		# strings
	outputs = []	# strings

	structure = """static const int structure["""+str(int(n_layers/2))+"""][2] = {"""

	n_weight = 0
	n_bias = 0
	for n in range(n_layers): 
		shape = trainable_shapes[n]
		
		if len(shape) == 2:
			## it is a weight matrix
			weight = """static const float layer_"""+str(n_weight)+"""_weight["""+str(shape[0])+"""]["""+str(shape[1])+"""] = {"""
			for row in trainable_evals[n]:
				weight += """{"""
				for num in row:
					weight += str(num) + ""","""
				# get rid of the comma after the last number
				weight = weight[:-1]
				weight += """},"""
			# get rid of the comma after the last curly bracket
			weight = weight[:-1]
			weight += """};\n"""
			weights.append(weight)
			n_weight += 1

			# augment the structure array
			structure += """{"""+str(shape[0])+""", """+str(shape[1])+"""},"""

		elif len(shape) == 1:
			## it is a bias vector 
			bias = """static const float layer_"""+str(n_bias)+"""_bias["""+str(shape[0])+"""] = {"""
			for num in trainable_evals[n]:
				bias += str(num) + ""","""
			# get rid of the comma after the last number
			bias = bias[:-1]
			bias += """};\n"""
			biases.append(bias)

			## add the output arrays
			output = """static float output_"""+str(n_bias)+"""["""+str(shape[0])+"""];\n"""
			outputs.append(output)

			n_bias += 1

	# complete the structure array
	## get rid of the comma after the last curly bracket
	structure = structure[:-1] 
	structure += """};\n"""

	"""
	Multiple for loops to do matrix multiplication
	 - assuming using tanh activation
	"""
	for_loops = []	# strings 

	# the first hidden layer
	input_for_loop = """
		for (int i = 0; i < structure[0][1]; i++) {
			output_0[i] = 0;
			for (int j = 0; j < structure[0][0]; j++) {
				output_0[i] += state_array[j] * layer_0_weight[j][i];
			}
			output_0[i] += layer_0_bias[i];
			output_0[i] = tanhf(output_0[i]);
		}
	"""
	for_loops.append(input_for_loop)

	# rest of the hidden layers
	for n in range(1, int(n_layers/2)-1):
		for_loop = """
		for (int i = 0; i < structure["""+str(n)+"""][1]; i++) {
			output_"""+str(n)+"""[i] = 0;
			for (int j = 0; j < structure["""+str(n)+"""][0]; j++) {
				output_"""+str(n)+"""[i] += output_"""+str(n-1)+"""[j] * layer_"""+str(n)+"""_weight[j][i];
			}
			output_"""+str(n)+"""[i] += layer_"""+str(n)+"""_bias[i];
			output_"""+str(n)+"""[i] = tanhf(output_"""+str(n)+"""[i]);
		}
		"""
		for_loops.append(for_loop)

	n = int(n_layers/2)-1
	# the last hidden layer which is supposed to have no non-linearity
	output_for_loop = """
		for (int i = 0; i < structure["""+str(n)+"""][1]; i++) {
			output_"""+str(n)+"""[i] = 0;
			for (int j = 0; j < structure["""+str(n)+"""][0]; j++) {
				output_"""+str(n)+"""[i] += output_"""+str(n-1)+"""[j] * layer_"""+str(n)+"""_weight[j][i];
			}
			output_"""+str(n)+"""[i] += layer_"""+str(n)+"""_bias[i];
		}
		"""
	for_loops.append(output_for_loop)

	## assign network outputs to control
	assignment = """
		control_n->thrust_0 = output_"""+str(n)+"""[0];
		control_n->thrust_1 = output_"""+str(n)+"""[1];
		control_n->thrust_2 = output_"""+str(n)+"""[2];
		control_n->thrust_3 = output_"""+str(n)+"""[3];	
	"""

	## construct the network evaluation function
	controller_eval = """
	void networkEvaluate(struct control_t_n *control_n, const float *state_array) {
	"""
	for code in for_loops:
		controller_eval += code 
	## assignment to control_n
	controller_eval += assignment

	## closing bracket
	controller_eval += """
	}
	"""

	## combine the all the codes
	source = ""
	## headers
	source += headers_network_evaluate
	## helper functions
	source += linear_activation
	source += sigmoid_activation
	source += relu_activation
	## the network evaluation function
	source += structure
	for output in outputs:
		source += output 
	for weight in weights:
		source += weight 
	for bias in biases:
		source += bias
	source += controller_eval

	## add log group for logging
	# source += log_group

	if output_path:
		with open(output_path, 'w') as f:
			f.write(source)

	return source