import tensorflow as tf

import queue
from queue import *

import json


import numpy as np
import tensorflow as tf
import itertools
import os

import time
import numpy as np
# import theano
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import threading
import csv


from env import Env
from machine import Machine
from task_dist import Task_Dist







class Agent():
	def __init__(self,session,pa):
		
		# session: the tensorflow session
		self.sess=session
		

		optimizer=tf.train.AdamOptimizer(pa.lr_rate)
		
		self.optimizer=optimizer
		self.action_size=pa.num_actions
		with tf.variable_scope('network'):
			# store the state,policy and value for the network

			self.state,self.policy,self.value,self.fc1=self.build_model(pa)
			# get the weights for the network 
			self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')

			# Placeholders for the action, advantage and target value
			self.action = tf.placeholder('int32', name='action')
			self.target_value = tf.placeholder('float32', name='target_value')
			self.advantages = tf.placeholder('float32', name='advantages')

		with tf.variable_scope('optimizer'):
			# Compute the one hot vectors for each action given.
			action_one_hot=tf.one_hot(self.action,self.action_size,1.0,0.0)
			 # Clip the policy output to avoid zeros and ones -- these don't play well with taking log.
			min_policy=0.000001
			max_policy=0.999999
			self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))

			# For a given state and action, compute the log of the policy at
			# that action for that state. This also works on batches.
			self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices=1)
			# define policy loss and value loss
			self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)
			self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))
			# entropy loss introduced to encourage exploration
			self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))
			self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.001
			 # Compute the gradient of the loss with respect to all the weights,
			# and create a list of tuples consisting of the gradient to apply to
			# the weight.
			self.grads = tf.gradients(self.loss, self.weights)
		#	self.grads, _ = tf.clip_by_global_norm(self.grads,20)
		#	grads, _ = tf.clip_by_global_norm(grads)
			self.grads_vars = list(zip(self.grads, self.weights))

			# Create an operator to apply the gradients using the optimizer.
			# Note that apply_gradients is the second part of minimize() for the
			# optimizer, so will minimize the loss.
			self.train_op = optimizer.apply_gradients(self.grads_vars)




	def build_model(self,pa):

		state = tf.placeholder('float32', shape=(None,pa.state_len
			,pa.state_width,1), name='state')

		# flatten the network 
		with tf.variable_scope('flatten'):
			flatten=tf.contrib.layers.flatten(inputs=state)
		# fully connected layer with 20 hidden units

		with tf.variable_scope('fc1'):
			fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=20,
			activation_fn=tf.nn.relu,
			weights_initializer=tf.contrib.layers.xavier_initializer(),
			biases_initializer=None)
		


		# the policy output
		with tf.variable_scope('policy'):
			policy = tf.contrib.layers.fully_connected(inputs=fc1,
			num_outputs=pa.num_actions, activation_fn=tf.nn.softmax,
			weights_initializer=tf.contrib.layers.xavier_initializer(),
			biases_initializer=None)


		with tf.variable_scope('value'):
			value = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1,
				activation_fn=None,
				weights_initializer=tf.contrib.layers.xavier_initializer(),
				biases_initializer=None)

		return state, policy, value,fc1

	# helper functions for getting policy, value and training 

	def get_value(self, state):
		return self.sess.run(self.value, {self.state: state}).flatten()

	def get_policy_and_value(self, state):
		policy, value = self.sess.run([self.policy, self.value], {self.state:
		state})
		return policy.flatten(), value.flatten()

	# Train the network on the given states and rewards
	def train(self, states, actions, target_values, advantages):
		
		x=self.sess.run(self.grads, feed_dict={
			self.state: states,
			self.action: actions,
			self.target_value: target_values,
			self.advantages: advantages
		})
		for c in x:
			print('Grad',np.count_nonzero(np.array(c)))
		print(self.sess.run(self.fc1, feed_dict={
			self.state: states,
			self.action: actions,
			self.target_value: target_values,
			self.advantages: advantages
		}))
		# Training
		self.sess.run(self.train_op, feed_dict={
			self.state: states,
			self.action: actions,
			self.target_value: target_values,
			self.advantages: advantages
		})

# function to visualize he state	

def visualize_state(mat, pa, path=None):
	with open('colormap.csv') as f:
		reader = csv.reader(f)
		data = [tuple([float(l) for l in r]) for r in reader]
	cmap = colors.ListedColormap(data)
	norm = colors.Normalize(vmin=0, vmax=1, clip=True)
	mat -= np.floor(mat)
	for i in range(pa.num_machines):
		mat[:,pa.machine_res_cap*(i+1)-1]=1

	new_mat = np.ones((mat.shape[0]+2, mat.shape[1]+2))
	new_mat[1:-1, 1:-1] = mat


	plt.figure("STATE")
	plt.xlabel('RESOURCE')
	plt.ylabel('TIME')
	plt.imshow(new_mat, interpolation='nearest', cmap=cmap, norm=norm)
	if path is not None:
		plt.savefig(str(path+'.jpg'))
	plt.pause(0.001)



def test(it,pa ,agent,env,workloads):
	logline = str(it) + '\n'
	for ex in range(pa.num_test_ex):
		env.workload_seq=workloads[ex+pa.num_ex]

		finished_episode_len=0
		counter=0
		env.reset()
		terminal=True
		# for i in range(20):
		env.generate_workload()
			# env.seq_id += 1
		print('Testing : ', env.workload_seq)
		terminal=False
		state = env.observe()
		indices=[]
		acts = []
		policies = []
		values=[]
		ex_rewards=[]
		cpu_crs = [0]*5
		cpu_crs_max=[0]*5
		mem_crs = [0]*5
		mem_crs_max=[0]*5
		c_utils = ''
		m_utils = ''
		json_array = []
		
		suffer=[]
		while not terminal:

			policy, value = agent.get_policy_and_value(np.expand_dims(np.expand_dims(state,axis=0),axis=3))
			values.append(value)	
			#	print(agent.action_size,policy.shape)
			action_idx = np.argmax(policy)
		#	visualize_state(state, pa, '/home/shanka/trajs_demo_distribution/episode_{0}_{1}'.format(int(ex),int(counter)))
			acts.append(action_idx)
			policies.append(policy)

			# #################json
			# json_all_machines = []
			# for k, machine in enumerate(env.machines):
			# 	# print(k)
			# 	json_machine_array = []
			# 	for task in machine.running_tasks:
			# 		json_task = {}
			# 		json_task['name'] = task.service
			# 		if len(task.cpu_util) < pa.hist_wind_len:
			# 			json_task['util'] = [0 for x in range(pa.hist_wind_len)]
			# 			json_task['util'][-len(task.cpu_util):] = task.cpu_util[-len(task.cpu_util):]
			# 			# print(len(json_task['util']))
			# 		else:
			# 			json_task['util'] = task.cpu_util[-pa.hist_wind_len:]
			# 		json_machine_array.append(json_task)
			# 	json_all_machines.append(json_machine_array)
			# json_incoming_tasks = []
			# x = []
			# for task in env.waiting_tasks:
			# 	x.append(task.service)
			# json_all_machines.append(x)
			# myFormattedList = [ (int(elem*100)) for elem in policy ]
			# json_all_machines.append(list(myFormattedList))

			# json_all_machines.append(str(action_idx))
			# # # if len(json_array) > 0:
			# # #     if status == 'Allocation_Success':
			# # #         x = []
			# # #         z = env.waiting_tasks[-len(prev_waiting_tasks)+1:]
			# # #         for task in z:
			# # #             x.append(task.service)
			# # #         json_array[-1].append(x)
			# # #     else:
			# # #         x = []
			# # #         z = env.waiting_tasks[-len(prev_waiting_tasks):]
			# # #         for task in z:
			# # #             x.append(task.service)
			# # #         json_array[-1].append(x)
			# json_array.append(json_all_machines)
			
			# prev_waiting_tasks = env.waiting_tasks
			#################

			# Take the action and get the next state, reward and terminal.
			state, ex_rewards, terminal, status= env.step(action_idx,counter,ex_rewards)
			counter+=1
	#			rews.append(reward)

			if  not(status == 'Backlog_Empty'):
				finished_episode_len = counter
				indices.append(counter-1)
		#	print('Status>>',terminal)
			
		#	print('Policy',policy)
		#	print('Value',value)
		#	print('Action>>',action_idx)
		#	print('Reward>>',batch_rewards[counter-1])
		#	print('finished_episode_len>>',finished_episode_len)
			 ##############logs
			# for k, machine in enumerate(env.machines):
			# 	crs_this_time=[0]*5
			# 	for i in range(len(machine.running_tasks)):
			# 		for j in range(i+1, len(machine.running_tasks)):
			# 			task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
			# 			if task_i != task_j:
			# 				crs[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
			# 				crs_this_time[k]+=pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
			# 	crs_max[k] = max(crs_max[k], crs_this_time[k])
			#################
			c_util = ''
			m_util = ''
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.cpus_left >= 0:
						c_util+=str(machine.total_cpus - machine.cpus_left) +','
					else:
						c_util+=str(machine.total_cpus) +','
						suffer.append(abs(machine.cpus_left))
				else:
					c_util += str(0) + ','
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.mems_left >= 0:
						m_util+=str(machine.total_mems - machine.mems_left) +','
					else:
						m_util+=str(machine.total_mems) +','
						suffer.append(abs(machine.mems_left))
				else:
					m_util += str(0) + ','

				cpu_crs_this_time = [0]*pa.num_machines
				mem_crs_this_time = [0]*pa.num_machines
				for i in range(len(machine.running_tasks)):
					for j in range(i+1, len(machine.running_tasks)):
						task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
						if task_i != task_j:
							cpu_crs[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
							cpu_crs_this_time[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
							mem_crs[k] += pa.interference_penalty * (task_i.mem_util[-1] * task_j.mem_util[-1]) * (-1)
							mem_crs_this_time[k] += pa.interference_penalty * (task_i.mem_util[-1] * task_j.mem_util[-1]) * (-1)
				cpu_crs_max[k] = max(cpu_crs_max[k], cpu_crs_this_time[k])
				mem_crs_max[k] = max(mem_crs_max[k], mem_crs_this_time[k])
				#################
			c_utils += c_util + '|'
			m_utils += m_util + '|'
		z=[]
		for c in indices:
				
			z.append(ex_rewards[c])
		logline += str(str(counter-1)+'|'+str(c_utils) + str(finished_episode_len)) + '\n' + str(sum(z)) + '\n' + str(sum(suffer))  +'\n'
		logline +=str(m_utils) +'\n'
		for i in range(len(env.machines)):
			logline += str(cpu_crs[i]) + ','
		logline = logline[:-1] + '\n'
		for i in range(len(env.machines)):
			logline += str(cpu_crs_max[i]) + ','
		logline = logline[:-1]
		logline += '\n'

		for i in range(len(env.machines)):
			logline += str(mem_crs[i]) + ','
		logline = logline[:-1] + '\n'
		for i in range(len(env.machines)):
			logline += str(mem_crs_max[i]) + ','
		logline = logline[:-1]
		logline += '\n'


				# crs_this_time = [0]*pa.num_machines
				# # for i in range(len(machine.running_tasks)):
				# # 	for j in range(i+1, len(machine.running_tasks)):
				# # 		task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
				# # 		if task_i != task_j:
				# # 			crs[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
				# # 			crs_this_time[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
				# # crs_max[k] = max(crs_max[k], crs_this_time[k])
			#################
			# if len(util)>0:
		# 	# 	utils += sum(util)/len(util)
		# for i in range(len(env.machines)):
		# 	logline += str(max(utils[i])) + ','
		# #logline+=str(sum(values)/len(values))+','
		# # z=[]
		# # #for c in indices:
				
		# # 		z.append(ex_rewards[c])
		# # logline += str(sum(z)) + '\n' + '\n' + str(sum(suffer))  +'\n' + str(finished_episode_len) + '\n'
		# # for i in range(len(env.machines)):
		# # 	logline += str(crs_max[i]) + ','
		# logline += '\n'






		# for i in range(len(env.machines)):
		# 	logline += str(crs[i]) + ','
		# logline += str(sum(values) / float(len(values))) + ','

		# logline += str(sum(ex_rewards)) + '\n'
		# for i in range(len(env.machines)):
		# 	logline += str(crs_max[i]) + ','
		# logline += '\n'

		if it%20==0:
		#	print('Example number : ',ex)
			x=[]
			y=[]
			z=[]
			w=[]
			for c in indices:
				x.append(acts[c])
				y.append(policies[c])
				z.append(ex_rewards[c])
			print('Test Actions : ', x[:len(indices)])
			print(y[:len(indices)])
			print('Reward : ',z[:len(indices)])
			print('Total reward : ', sum(z[:len(indices)]))
		# print('Reward : ', sum(rews))
		# with open('/home/shanka/json_logs_distribution/'+str(ex)+'.json_', 'w') as json_file:
		# 	json.dump(json_array, json_file)
	return logline




def async_trainer(agent, env,pa, sess,saver,workloads):
	print ("Training Starts")
	
	it=0
#	T = T_queue.get()
#	T_queue.put(T+1)
#	t = 0

#	last_verbose = T
#	last_time = time()
#	last_target_update = T
	
	terminal = True
	timer_start=time.time()
	#open the log file for writing relevant outputs
	logs = open('/home/dell/logs_cpu_mem_40ex', 'a')
	# loglines = ''
	while it < pa.num_epochs:
#		t_start = t
	#	batch_states = []
	#	batch_rewards = []
	#	batch_actions = []
	#	baseline_values = []\
	 # stores states ,actions,target values  and advantages for all examples 
		k=[]
		l=[]
		m=[]
		n=[]
		
		for ex in range(pa.num_ex):
			# example wise states, actions and rewards and values 
			ex_states = []
			ex_rewards = []
			ex_actions = []
			ex_baseline_values = []
			# get the workload sequence for the given example 
			env.workload_seq = workloads[ex]
	#		print('Example>>',workloads[ex])
			finished_episode_len=0
			counter=0
			if terminal:
				terminal = False
				obs=[]
				rews=[]
				acts=[]
				env.reset()
				# for i in range(20):
				# get the workload  at the current timestep 
				env.generate_workload()
					# env.seq_id += 1
				# get the new state
				state = env.observe()
				
				indices=[]
			
			while not terminal :
				obs.append(state)
				
			
				# save the states 
				ex_states.append(np.expand_dims(np.expand_dims(state,axis=0),axis=3))

				# Choose an action randomly according to the policy
				# probabilities. We do this anyway to prevent us having to compute
				# the baseline value separately.
			#	print('State>>',state)
				policy, value = agent.get_policy_and_value(np.expand_dims(np.expand_dims(state,axis=0),axis=3))
				
			#	print(agent.action_size,policy.shape)
				action_idx = np.random.choice(agent.action_size, p=policy)
				acts.append(action_idx)

				# Take the action and get the next state, reward and terminal.
				state, ex_rewards, terminal, status= env.step(action_idx,counter,ex_rewards)
				counter+=1
	#			rews.append(reward)

			# Store those indices when backlog queue is not empty
				if not(status == 'Backlog_Empty'):
					finished_episode_len = counter
					indices.append(counter-1)


			#	print('Status >>',status)

				 #print('Time_Step>>',T)
			#	print('Policy',policy)
				# print('Value',value)
			#	print('Action>>',action_idx)
			#	print('Reward>>',ex_rewards[counter-1])
				#print('finished_episode_len>>',finished_episode_len)
				# Update counters
			#	T += 1
		#		T = T_queue.get()
		#		T_queue.put(T+1)

				
			
				
			#	batch_rewards.append(reward)
				
				# save the actions and values for each state
				ex_actions.append(action_idx)
				ex_baseline_values.append(value[0])


			target_value = 0
			# If the last state was terminal, just put R = 0. Else we want the
			# estimated value of the last state.
			if not terminal:
				target_value = agent.get_value(np.expand_dims(np.expand_dims(state,axis=0),axis=3))[0]
			last_R = target_value

			# Compute the sampled n-step discounted reward
		#	new_rews=[]
		#	for g in range(finished_episode_len-1):
		#		new_rews.append(sum(batch_rewards[g:g+pa.hist_wind_len]))
		#	new_rews = batch_rewards[:finished_episode_len-1]
		#	new_rews.append(sum(batch_rewards[finished_episode_len-1:]))
		#	print('Example No >>',ex)
		#	print('Time_Step>>',T)
		#	print('Example Rewards >>',ex_rewards[:finished_episode_len])
		#	print('Actions >>',ex_actions[:finished_episode_len])
		#	print('finished_episode_len >>',finished_episode_len)

		# get the valid states , actions , rewards and values corresponding to indices 
			final_states=[]
			final_actions=[]
			final_rewards=[]
			final_values=[]
			for c in indices:
				final_states.append(ex_states[c])
				final_actions.append(ex_actions[c])
				final_rewards.append(ex_rewards[c])
				final_values.append(ex_baseline_values[c])
			# compute the target values for each state based on the discounted rewards
			ex_target_values = []
			for reward in reversed(final_rewards):
				target_value = reward + pa.discount * target_value
				ex_target_values.append(target_value)
			# Reverse the example target values, so they are in the correct order
			# again.
			ex_target_values.reverse()

			# Compute the estimated value of each state(the advantage function)
			ex_advantages = np.array(ex_target_values) - np.array(final_values)
		#	print('adv',batch_advantages.shape)
		#	print('Stack',np.vstack(ex_states).shape)
		#	print('Actions',np.array(ex_actions).shape)
		#	print('Targets',np.array(ex_target_values).shape)
		#	print('Indices>>',indices)
		#	print('finished_episode_len>>',finished_episode_len)

			#print('Actions>>',final_actions)
			a1=np.vstack(final_states)
			a2=np.array(final_actions)
			a3=np.array(ex_target_values)
			a4=ex_advantages	
			# store the states, action, target values and advantages for all the examples 

			for i in range(a1.shape[0]):
				k.append(a1[i])
				l.append(a2[i])
				m.append(a3[i])
				n.append(a4[i])

			
		batch_states=np.array(k)
		batch_actions=np.array(l)
		batch_target_values=np.array(m)
		batch_advantages=np.array(n)
	#	print('Batch States>>',batch_states.shape)
	#	print('Batch Actions>>',batch_actions.shape)	
		# Apply asynchronous gradient update
	#	agent.train(np.vstack(batch_states[:finished_episode_len]), np.array(batch_actions[:finished_episode_len]), np.array(batch_target_values[:finished_episode_len]),
	#	batch_advantages)
			# if itr==pa.batch_size or ex==pa.num_ex-1:
			# 	batch_states=np.array(k)
			# 	batch_actions=np.array(l)
			# 	batch_target_values=np.array(m)
			# 	batch_advantages=np.array(n)
		agent.train(batch_states, batch_actions, batch_target_values,
		batch_advantages)
				
		timer_end=time.time()
		print ("-----------------")
		print ("Iteration: \t %i" % it)
		print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
		print ("-----------------")
		it+=1
		

		# if already model checkpoints are saved then restore the models from the saved directory 
		# call the test function to write into the log file 
		if it%1==0:
			print('Testing')
			# change this loop iteration,depending on the save freq and total no of checkpoints
			for ite in range(8900, 10160, 20):
				print(ite)
				saver.restore(sess, "/home/dell/actor_critic_cpu_mem_dist_40ex/model.ckpt-"+str(ite))
				logline=test(ite,pa,agent,env,workloads)
				logs.write(logline)
				logs.flush()
		
				os.fsync(logs.fileno())
			break
			# 	test(it,pa,agent,env,workloads)
			# 	print('File>>>',i)
		
			# break
			# 	print('File>>>',i)
		
	# saves the model checkpoint after every 20 iterations and writes into log file after
	# calling the test function 	
	#	loglines += logline
	#	if it % 20 == 0:
	#		saver.save(sess, '/home/dell/actor_critic_model_results_dist_cpu_mem_grad20/model.ckpt', global_step=it)
	#		logline=test(it,pa,agent,env,workloads)
	#   		logs.write(logline)
	#   		logs.flush()
	#   		os.fsync(logs.fileno())
	# logs.close()

	global training_finished
	training_finished = True





# def a3c(pa, restore=None, save_path=None):

# 	task_dist = Task_Dist()
# 	workloads = task_dist.gen_seq_workload()

# # Create the environments
# 	envs = []
# 	for ex in range(pa.num_ex):
# 		env = Env(0, 1)
# 		env.workload_seq = workloads[ex]
# 		envs.append(env)
		

# 	# Also create an environment for evaluating the agent
# #    evaluation_env = CustomGym(gym_name)

# 	# Create the tensorflow session and use it in context
	
# 	with tf.Session() as sess:

# 		# Create the agent
# 		agent = Agent(sess, pa)
		

# 		# Create a saver, and only keep 2 checkpoints.
# 		saver = tf.train.Saver(max_to_keep=500)

# 		T_queue = Queue()

# 		# Either restore the parameters or don't.
# 		if restore is not None:
# 			saver.restore(sess, save_path + '-' + str(restore))
# 			last_T = restore
# 			print ("T was:", last_T)
# 			T_queue.put(last_T)
# 		else:
# 			sess.run(tf.global_variables_initializer())
# 			T_queue.put(0)

# 	#	summary = Summary(save_path, agent)

# 		# Create a process for each worker
# 		processes = []
# 		for i in range(pa.num_ex):
# 			processes.append(threading.Thread(target=async_trainer, args=(agent,
# 			envs[i], pa,sess, i, T_queue, saver, save_path)))

# 		# Create a process to evaluate the agent
# 	   # processes.append(threading.Thread(target=evaluator, args=(agent,
# 		#evaluation_env, sess, T_queue,)))

# 		# Start all the processes
# 		for p in processes:
# 			p.daemon = True
# 			p.start()

# 		# Until training is finished
# 	#	while not training_finished:
# 	#		sleep(0.01)

# 		# Join the processes, so we get this thread back.
# 		for p in processes:
# 			p.join()





#T_MAX = 100000000
# Use this many threads
#NUM_THREADS = 1
# Initial learning rate for Adam
#	INITIAL_LEARNING_RATE = 1e-4
# The discount factor
#	DISCOUNT_FACTOR = 0.99
# Evaluate the agent and print out average reward every this many steps
#	VERBOSE_EVERY = 40000
# Update the parameters in each thread after this many steps in that thread
#I_ASYNC_UPDATE = 5
# Use this global variable to exit the training loop in each thread once we've finished.
training_finished = False

def main():
	import params
	import sys
	task_dist = Task_Dist()
	workloads = task_dist.gen_seq_workload()

	pa=params.Params()
	env = Env(0, 1)
#	env.workload_seq = workloads[0]
	import os
	import sys, getopt
	import threading
	import tensorflow as tf
	import numpy as np
	from time import time, sleep

	
	import random
	#from agent import Agent

	# Train for this many steps
#	T_MAX = 100000000
	# Use this many threads
	#NUM_THREADS = 8
	# Initial learning rate for Adam
#	INITIAL_LEARNING_RATE = 1e-4
	# The discount factor
#	DISCOUNT_FACTOR = 0.99
	# Evaluate the agent and print out average reward every this many steps
#	VERBOSE_EVERY = 40000
	# Update the parameters in each thread after this many steps in that thread
	#I_ASYNC_UPDATE = 5
	# Use this global variable to exit the training loop in each thread once we've finished.
	training_finished = False
	tf.reset_default_graph()
	sess = tf.Session()
	

	
	agent=Agent(sess,pa)
#	a3c(pa)
	sess.run(tf.global_variables_initializer())
	# create a tensorflow saver pbject and keep 50000 checkpoints
	saver = tf.train.Saver(max_to_keep=50000)
	async_trainer(agent,env,pa,sess,saver,workloads)	


if __name__ =='__main__':
	main()
