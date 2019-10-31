#!/usr/bin/env python

import argparse
import csv
import joblib
import os
import re
import sys
import shutil
import tensorflow as tf
import yaml

import gaussian_mlp as mlp

def subdir(root_dir):
	"""
	return all subdirectories as a list
	Args:
		folder [str]: a root directory
	rtype:
		List[str]
	"""
	return [sub_f.path for sub_f in os.scandir(root_dir) if sub_f.is_dir() ]

def read_txt_to_get_dirs(root_dir, txt):
	"""
	Read a txt file containing directories to models
	Assuming that all the directories are sub directories
	(i.e. the full path to a model is root + '/' + sub_dir)
	"""
	sub_dirs = []
	root_dir = root_dir.strip().rstrip(os.sep)
	with open(txt, 'r') as f:
		for line in f:
			line = line.strip().lstrip(os.sep)
			print(root_dir + '/' + line)
			assert os.path.isdir(root_dir + '/' + line) == True
			sub_dirs.append(root_dir + '/' + line)
	return sub_dirs

def analyze_seeds(experiment):
	'''
	Args:
		experiment [str]: root directory of a single experiment containing multiple seeds
	rtype [str]:
		the directory of the seed with the highest average reward
	'''

	assert os.path.isdir(experiment) == True

	seeds = subdir(experiment)

	highest_reward = -float('inf')
	for seed_dir in seeds:
		## check if it is a seed directory
		seed_dir_split = seed_dir.split('/')
		if not re.search(r'^seed_*', seed_dir_split[-1]):
			print('Experiment %s has seed folder that is named incorrectly... terminating...' % seed_dir)
			exit(1)
		else:
			with open(seed_dir + '/progress.csv', 'r') as csvfile:
				progress_reader = csv.DictReader(csvfile)
				main_reward_latest = list(progress_reader)[-1]['rewards/rew_main_avg']
				if highest_reward <= float(main_reward_latest):
					target_seed = seed_dir
					best_seed = seed_dir_split[-1]
					highest_reward = float(main_reward_latest)


	print('Best seed: %s' % best_seed)
	return target_seed

def save_result(model_dir, out_dir, osi=False, absolute_path=False):
	"""
	Save the params.pkl file and the config file.
	Convert the graph to source code and save
	Args:
		model_dir [str]: the directory of which the pickle file is located
		out_dir [str]: the root directory of which the model should be saved
		osi [bool]: indicates whether the model is an osi
		absolute_path [bool]: (default False) indicated whether the out_dir 
			has been modified to the desired sub location
	"""
	model_dir = model_dir.rstrip(os.sep)
	out_dir = out_dir.rstrip(os.sep)
	
	if not absolute_path:
		## the out_dir is still the out_dir provided at the command line
		## try to append the correct sub_dir to it
		desired_sub_p = model_dir.split('/')[-5:]
		desired_sub_p = '/'.join(desired_sub_p)
		out_dir += '/' + desired_sub_p

	try:
		os.makedirs(out_dir, exist_ok=True)
	except FileExistsError:
		# directory already exists
		pass

	shutil.copyfile(model_dir + '/params.pkl', out_dir + '/params.pkl')
	shutil.copyfile(model_dir + '/config.yml', out_dir + '/config.yml')

	tf.reset_default_graph()
	with tf.Session() as sess:

		print("extrating parameters from file %s ..." % model_dir + '/params.pkl')
		pkl_params = joblib.load(model_dir + '/params.pkl')
		policy = pkl_params['policy']

		mlp.generate(policy, sess, out_dir+'/network_evaluate.c')

def copy_by_best_seed(root_dir, out_dir):
	"""
	TODO: write comments
	"""
	print('Searching root %s ...' % root_dir)
	print('================================')
	subdirs = subdir(root_dir)

	for experiment in subdirs:
		print('Searching subdir %s ... Analyzing seeds' % experiment)
		## grad the seed with the highest average reward
		target_seed = analyze_seeds(experiment)
		save_result(target_seed, out_dir)

def copy_by_txt(root_dir, out_dir, txt):
	"""
	Copy the models specified in a txt file
	All the models must be located under the root_dir
	Args:
		root_dir [str]: the root directory
		out_dir [str]: the output directory [will create one if it doesn't exist]
		txt [str]: the txt file specifying the model relative directories
	"""
	print('searching root %s ...' % root_dir)
	print('================================')

	subdirs = read_txt_to_get_dirs(root_dir, txt)
	for experiment in subdirs:
		print('copying params.pkl from %s to %s...' % (experiment, out_dir))
		save_result(experiment, out_dir)


def traverse_root(root_dir, out_dir):
	"""
	Recursively search for pickle files of models and 
	covert the model if found

	Args:
		root_dir [str]: the root directory
		out_dir [str]: the output directory [will create one if it doesn't exist]
		osi [bool]: to indicate if a model is an osi
	"""
	subdirs = subdir(root_dir)
	for path in subdirs:
		path = path.rstrip(os.sep)
		if os.path.isfile(path + '/params.pkl') == True:
			save_path = '/'.join([i for i in path.split('/')[-5:]]) ## -5 is picked appropriately
			save_path = out_dir.rstrip(os.sep)+'/'+save_path
			print('copying params.pkl from %s to %s...' % (path, save_path))
			save_result(path, save_path, absolute_path=True)
		else:
			traverse_root(path, out_dir)

def main(args):
	if args.mode == 0:
		copy_by_txt(args.root_dir, args.out_dir, args.txt)
	elif args.mode == 1:
		copy_by_best_seed(args.root_dir, args.out_dir)
	elif args.mode == 2:
		traverse_root(args.root_dir, args.out_dir)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument(
		'mode',
		type=int,
		default=2,
		help='select a mode to copy file.\n'
			 '0: a txt file with dirs\n'
			 '1: a root where all the experiments are stored and select the best seeds.\n'
			 '2: a root dir where all the subdirs that contain plk file will be copied.\n', 
	)

	parser.add_argument(
		'root_dir',
		type=str,
		help='Root dir of the experiments'
	)

	parser.add_argument(
		'out_dir', 
		type=str,
		help='dir to save the experiments'
	)

	parser.add_argument(
		'-txt',
		type=str,
		help='txt file that contains all the models'
	)

	args = parser.parse_args() 

	main(args)