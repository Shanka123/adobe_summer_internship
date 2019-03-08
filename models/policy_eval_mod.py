import time
import numpy as np
import theano
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import threading
import csv
#import simulation environment file
import policy_network

from multiprocessing import Process
from multiprocessing import Manager
import os,json

from env import Env
from machine import Machine
from task_dist import Task_Dist

def init_accums(pg_learner):  # in rmsprop
	accums = []
	params = pg_learner.get_params()
	for param in params:
		accum = np.zeros(param.shape, dtype=param.dtype)
		accums.append(accum)
	return accums

def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):

	assert len(grads) == len(params)
	assert len(grads) == len(accums)
	for dim in range(len(grads)):
		accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
		params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))
	return accums,params

def discount(x, gamma):
	"""
	Given vector x, computes a vector y such that
	y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
	"""
	out = np.zeros(len(x))
	out[-1] = x[-1]
	for i in reversed(range(len(x)-1)):
		out[i] = x[i] + gamma*out[i+1]
	assert x.ndim >= 1
	# More efficient version:
	# scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
	return out

def get_entropy(vec):
	entropy = - np.sum(vec * np.log(vec))
	if np.isnan(entropy):
		entropy = 0
	return entropy

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
	plt.pause(0.01)

def get_traj(agent, env, episode_max_length):
	"""
	Run agent-environment loop for one whole episode (trajectory)
	Return dictionary of results
	"""
	env.reset()

	#call reset function of simulator
	obs = []
	acts = []
	rews = []
	final_obs=[]
	final_acts=[]
	final_rews=[]
	indices=[]
	#entropy = []
	info = []
	probs = []
	# for i in range(20):
	env.generate_workload()
		# env.seq_id += 1

	ob = env.observe()
	#call the observe function from simulator
	art_a = []
	finished_episode_len = 0
	for _ in range(episode_max_length):
		act_prob = agent.get_one_act_prob(ob)
		csprob_n = np.cumsum(act_prob)
		a = (csprob_n > np.random.rand()).argmax()

		obs.append(ob)  # store the ob at current decision making step
		acts.append(a)
		probs.append(act_prob)
		# plt1 = visualize_state(ob)
		# print('State at %d : ' % env.cur_time)
		# np.set_printoptions(linewidth=40*5, precision = 2, threshold=np.nan)
		# print(ob)
		# print(a+1)
		ob, rews, done, status= env.step(a, _, rews)
		# print(rews)
		#call the step function from simulator

	#    entropy.append(get_entropy(act_prob))
		if status == 'Allocation_Success':
			finished_episode_len = _ + 1
		if status != 'Backlog_Empty':
			indices.append(_)
		if done: break
	for c in indices:
		final_obs.append(obs[c])
		final_acts.append(acts[c])
		final_rews.append(rews[c])	
		

	new_rews = rews[:finished_episode_len]

	return {'reward': np.array(final_rews),
			'ob': np.array(final_obs),
			'action': np.array(final_acts),
			'prob' : probs[:finished_episode_len]
			}



def concatenate_all_ob(trajs, pa):

	timesteps_total = 0
	for i in range(len(trajs)):
		timesteps_total += len(trajs[i]['reward'])

	all_ob = np.zeros(
		(timesteps_total, 1, pa.state_len, pa.state_width),
		dtype=theano.config.floatX)

	timesteps = 0
	for i in range(len(trajs)):
		for j in range(len(trajs[i]['reward'])):
			all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
			timesteps += 1

	return all_ob



def concatenate_all_ob_across_examples(all_ob, pa):
	num_ex = len(all_ob)
	total_samp = 0
	for i in range(num_ex):
		total_samp += all_ob[i].shape[0]

	all_ob_contact = np.zeros(
		(total_samp, 1, pa.state_len, pa.state_width),
		dtype=theano.config.floatX)

	total_samp = 0

	for i in range(num_ex):
		prev_samp = total_samp
		total_samp += all_ob[i].shape[0]
		all_ob_contact[prev_samp : total_samp, :, :, :] = all_ob[i]

	return all_ob_contact

def get_traj_worker(pg_learner, env, pa, result):
	trajs = []
	for i in range(pa.num_seq_per_batch):
		traj = get_traj(pg_learner, env, pa.episode_max_length)
		trajs.append(traj)
		# print(traj['action'])
		# print(traj['reward'], sum(traj['reward']))

	all_ob = concatenate_all_ob(trajs, pa)

	# Compute discounted sums of rewards
	rets = [discount(traj["reward"], pa.discount) for traj in trajs]
	maxlen = max(len(ret) for ret in rets)
	padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

	# Compute time-dependent baseline
	baseline = np.mean(padded_rets, axis=0)

	# Compute advantage function
	advs = [ret - baseline[:len(ret)] for ret in rets]
	all_action = np.concatenate([traj["action"] for traj in trajs])
	all_adv = np.concatenate(advs)

	all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
	all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

	result.append({"all_ob": all_ob,
				   "all_action": all_action,
				   "all_adv": all_adv,
				   "all_eprews": all_eprews,
				   "all_eplens": all_eplens})

def launch(pa, pg_resume=None, save_freq = 50, render=False, repre='image', end='no_new_job', test_only=False):

	task_dist = Task_Dist()
	workloads = task_dist.gen_seq_workload()
	if test_only:
		test(0, pa, pg_resume, workloads)
		return

	pg_learners = []
	envs = []
	accums=[]
	for ex in range(pa.num_ex):
		print("-prepare for env-", ex)
		env = Env(0, 1)
		env.workload_seq = workloads[ex]
		envs.append(env)

	for ex in range(pa.batch_size + 1):  # last worker for updating the parameters
		print("-prepare for worker-", ex)
		pg_learner = policy_network.PGLearner(pa)

		if pg_resume is not None:
			net_handle = open(pg_resume, 'rb')
			net_params = pickle.load(net_handle)
			pg_learner.set_net_params(net_params)

		pg_learners.append(pg_learner)
	for idx in range(pa.batch_size):
		accums.append(init_accums(pg_learners[idx]))

	print ('Preparing for Training from Scratch...')

	#   ref_discount_rews=slow_down_cdf.launch(pa,pg_resume=None,render=False,repre=repre,end=end)
	all_test_rews = []
	timer_start=time.time()


	logs = open('/tmp/logs', 'a')
	loglines = ''
	for iteration in range(1, pa.num_epochs+1):
		ps = []  # threads
		manager = Manager()  # managing return results
		manager_result = manager.list([])

		ex_indices = list(range(pa.num_ex))
	#	np.random.shuffle(ex_indices)

		all_ob=[]
		all_action=[]
		grads_all = []
		eprews = []
		eplens = []
		all_adv=[]
		all_eprews=[]
		all_eplens=[]

		ex_counter = 0
		for ex in range(pa.num_ex):
			ex_idx = ex_indices[ex]
			p = Process(target=get_traj_worker,
						args=(pg_learners[ex_counter], envs[ex_idx], pa, manager_result, ))
			ps.append(p)

			ex_counter += 1

			if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

				print(ex+1, "out of", pa.num_ex)

				ex_counter = 0

				for p in ps:
					p.start()

				for p in ps:
					p.join()

				result = []  # convert list from shared memory
				for r in manager_result:
					result.append(r)

				ps = []
				manager_result = manager.list([])
				all_ob=[]
				all_action=[]
				all_adv=[]
			#	all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
			#	all_action = np.concatenate([r["all_action"] for r in result])
			#	all_adv = np.concatenate([r["all_adv"] for r in result])
				for idx in range(pa.batch_size):
					all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result[idx:idx+1]], pa)
					all_action = np.concatenate([r["all_action"] for r in result[idx:idx+1]])
					all_adv = np.concatenate([r["all_adv"] for r in result[idx:idx+1]])
					

				# Do policy gradient update step, using the first agent
				# put the new parameter in the last 'worker', then propagate the update at the end
					grads = pg_learners[idx].get_grad(all_ob,all_action,all_adv)
					params = pg_learners[idx].get_params()

					accums[idx],params=rmsprop_updates_outside(grads, params, accums[idx], pa.lr_rate, pa.rms_rho, pa.rms_eps)
					pg_learners[idx].set_net_params(params)
			#	grads_all.append(grads)

				all_eprews.extend([r["all_eprews"] for r in result])

				eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
				eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

		# assemble gradients
	#	grads = grads_all[0]
	#	for i in range(1, len(grads_all)):
	#		for j in range(len(grads)):
	#			grads[j] += grads_all[i][j]

		# propagate network parameters to others
	#	params = pg_learners[pa.batch_size].get_params()

	#	rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

	#	for i in range(pa.batch_size + 1):
	#		pg_learners[i].set_net_params(params)

		timer_end=time.time()
		print ("-----------------")
		print ("Iteration: \t %i" % iteration)
		print ("NumTrajs: \t %i" % len(eprews))
		print ("NumTimesteps: \t %i" % np.sum(eplens))
		print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
		print ("-----------------")
		# time.sleep(5)
	#	pg_resume = '/tmp/case2_multi_cpu_mem_dist_20_moretraj/%s.pkl_' % str(iteration)
	#	if iteration % 2 == 0:
	#		param_file = open(pg_resume, 'wb')
	#		pickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
	#		param_file.close()

		if iteration % 2 == 0:
			logline = test(iteration, pa, pg_resume, workloads, pg_learners)
			loglines += logline
			if iteration % 2 == 0:
				logs.write(loglines)
				logs.flush()
				os.fsync(logs.fileno())
				loglines = ''
	logs.close()

def test(it, pa ,pg_resume, workloads, pg_learners, episode_max_length=200):

	# if pg_learner is None:
	# 	pg_learner=policy_network.PGLearner(pa)
	# 	if pg_resume is not None:
	# 		net_handle = open(pg_resume, 'rb')
	# 		net_params = pickle.load(net_handle)
	# 		pg_learner.set_net_params(net_params)

	env = Env(0, 1)
	logline = str(it) + '\n'
	for ex in range(pa.num_ex):
		env.reset()
		env.workload_seq = workloads[ex]
		env.generate_workload()
		print('Testing : ', env.workload_seq)
		ob = env.observe()
		acts = []
		probs = []
		crs = [0]*pa.num_machines
		crs_max = [0]*pa.num_machines
		rews = []
		final_obs=[]
		final_acts=[]
		final_rews=[]
		indices=[]
		json_array = []
		utils = 0
		suffer = []
		for _ in range(episode_max_length):
			act_prob = pg_learners[ex].get_one_act_prob(ob)
			csprob_n = np.cumsum(act_prob)
			a = np.argmax(act_prob)

			#################json
			json_all_machines = []
			for k, machine in enumerate(env.machines):
				# print(k)
				json_machine_array = []
				for task in machine.running_tasks:
					json_task = {}
					json_task['name'] = task.service
					if len(task.cpu_util) < pa.hist_wind_len:
						json_task['util'] = [0 for x in range(pa.hist_wind_len)]
						json_task['util'][-len(task.cpu_util):] = task.cpu_util[-len(task.cpu_util):]
						# print(len(json_task['util']))
					else:
						json_task['util'] = task.cpu_util[-pa.hist_wind_len:]
					json_machine_array.append(json_task)
				json_all_machines.append(json_machine_array)
			json_incoming_tasks = []
			x = []
			for task in env.waiting_tasks:
				x.append(task.service)
			json_all_machines.append(x)
			json_all_machines.append(str(a))
			# if len(json_array) > 0:
			#     if status == 'Allocation_Success':
			#         x = []
			#         z = env.waiting_tasks[-len(prev_waiting_tasks)+1:]
			#         for task in z:
			#             x.append(task.service)
			#         json_array[-1].append(x)
			#     else:
			#         x = []
			#         z = env.waiting_tasks[-len(prev_waiting_tasks):]
			#         for task in z:
			#             x.append(task.service)
			#         json_array[-1].append(x)
			json_array.append(json_all_machines)
			
			# prev_waiting_tasks = env.waiting_tasks
			#################

			# plt1 = visualize_state(ob, pa, '/tmp/trajs/'+str(_)+'.jpg')
			# if _ < sum([len(i) for i in workloads[0]]):
			# 	print('Agent action: ', a)
			# 	man_act = input('Manual Action    :   ')
			# 	if man_act:
			# 		a = int(man_act)
			ob, rews, done, status= env.step(a, _, rews)
			acts.append(a)
			probs.append(act_prob)
			if status == 'Allocation_Success':
				finished_episode_len = _ + 1
			if status !='Backlog_Empty':
				indices.append(_)
			if done:
				break
			##############logs
			util = []
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.cpus_left >= 0:
						util.append(machine.total_cpus - machine.cpus_left)
					else:
						util.append(machine.total_cpus)
						suffer.append(abs(machine.cpus_left))
				crs_this_time = [0]*pa.num_machines
				for i in range(len(machine.running_tasks)):
					for j in range(i+1, len(machine.running_tasks)):
						task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
						if task_i != task_j and len(task_i.cpu_util)>0 and len(task_j.cpu_util)>0:
							crs[k] += pa.interference_penalty_cpu * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
							crs_this_time[k] += pa.interference_penalty_cpu * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
				crs_max[k] = max(crs_max[k], crs_this_time[k])
			#################
			utils += sum(util)/len(util)
		for c in indices:
			final_acts.append(acts[c])
			final_rews.append(rews[c])
		for i in range(len(env.machines)):
			logline += str(crs[i]) + ','
		logline += str(sum(rews)) + '\n' + str(utils) + '\n' + str(sum(suffer))  +'\n' + str(finished_episode_len) + '\n'
		for i in range(len(env.machines)):
			logline += str(crs_max[i]) + ','
		logline += '\n'
		if it % 2 == 0:
			print('Test Actions: ',final_acts)
			print(probs[:finished_episode_len])
			print('Reward : ', final_rews)
			print('Full Reward: ',rews)
			print('Reward : ', sum(rews))


		# with open('/home/rnehra/json_logs/'+str(ex)+'.json', 'w') as json_file:
		# 	json.dump(json_array, json_file)

	return logline

def test2(pa):
	def ex_test(pg_learner, env, pa, result):
		env.reset()
		env.generate_workload()
		ob = env.observe()
		acts = []
		probs = []
		cpu_crs = [0]*pa.num_machines
		cpu_crs_max = [0]*pa.num_machines
		mem_crs = [0]*pa.num_machines
		mem_crs_max = [0]*pa.num_machines
		rews = []
		c_utils = ''
		m_utils= ''
		suffer = []
		finished_episode_len = 0
		logline = ''
		for _ in range(pa.episode_max_length):
			act_prob = pg_learner.get_one_act_prob(ob)
			csprob_n = np.cumsum(act_prob)
			a = np.argmax(act_prob)
			ob, rews, done, status= env.step(a, _, rews)
			acts.append(a)
			probs.append(act_prob)
			if status == 'Allocation_Success':
				finished_episode_len = _ + 1
			if done:
				break
			##############logs
			c_util = ''
			m_util= ''
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.cpus_left >= 0:
						c_util += str(machine.total_cpus - machine.cpus_left) + ','
					else:
						c_util += str(machine.total_cpus) + ','
						suffer.append(abs(machine.cpus_left))
				else:
					c_util += str(0) + ','
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.mems_left >= 0:
						m_util += str(machine.total_mems - machine.mems_left) + ','
					else:
						m_util += str(machine.total_mems) + ','
						suffer.append(abs(machine.mems_left))
				else:
					m_util += str(0) + ','
				cpu_crs_this_time = [0]*pa.num_machines
				for i in range(len(machine.running_tasks)):
					for j in range(i+1, len(machine.running_tasks)):
						task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
						if task_i != task_j:
							cpu_crs[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
							cpu_crs_this_time[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
				cpu_crs_max[k] = max(cpu_crs_max[k], cpu_crs_this_time[k])
				#################
				mem_crs_this_time = [0]*pa.num_machines
				for i in range(len(machine.running_tasks)):
					for j in range(i+1, len(machine.running_tasks)):
						task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
						if task_i != task_j:
							mem_crs[k] += pa.interference_penalty * (task_i.mem_util[-1] * task_j.mem_util[-1]) * (-1)
							mem_crs_this_time[k] += pa.interference_penalty * (task_i.mem_util[-1] * task_j.mem_util[-1]) * (-1)
				mem_crs_max[k] = max(mem_crs_max[k], mem_crs_this_time[k])
				#################
			c_utils += c_util + '|'
			m_utils += m_util + '|'
		logline += str(str(_-1)+'|'+str(c_utils) + str(finished_episode_len)) + '\n' + str(sum(rews)) + '\n' + str(sum(suffer))  +'\n'
		logline+=str(m_utils) +'\n'
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

		result.append(logline)


	pg_learners = []
	envs = []
	task_dist = Task_Dist()
	workloads = task_dist.gen_seq_workload()
	for ex in range(pa.num_test_ex):
		print("-prepare for env-", ex)
		env = Env(0, 1)
		env.workload_seq = workloads[ex + pa.num_ex]
		envs.append(env)

	for ex in range(pa.batch_size):  # last worker for updating the parameters
		print("-prepare for worker-", ex)
		pg_learner = policy_network.PGLearner(pa)
		pg_learners.append(pg_learner)

	logs = open('logs_cpu_dist', 'a')
	loglines = ''
	for it in range(2, 1840, 2):
		if(it % 10):
			print('Iteration : ',it)
		pg_resume = 'params_tasks_99_11_dist/' + str(it) + '.pkl_'
		net_handle = open(pg_resume, 'rb')
		net_params = pickle.load(net_handle)
		for ex in range(pa.batch_size):
			pg_learners[ex].set_net_params(net_params)

		ps = []  # threads
		manager = Manager()  # managing return results
		manager_result = manager.list([])
		ex_counter = 0
		loglines += str(it) + '\n'
		for ex in range(pa.num_test_ex):
			p = Process(target=ex_test,
						args=(pg_learners[ex_counter], envs[ex], pa, manager_result, ))
			ps.append(p)

			ex_counter += 1

			if ex_counter >= pa.batch_size or ex == pa.num_test_ex - 1:
				ex_counter = 0
				for p in ps:
					p.start()

				for p in ps:
					p.join()

				# convert list from shared memory
				for r in manager_result:
					loglines += r 

				ps = []
				manager_result = manager.list([])
	logs.write(loglines)


def main():
	import params
	import sys
	pa=params.Params()
	pg_resume = None
	test_only = False
	if len(sys.argv) == 2:
		pg_resume=sys.argv[1] #give the path of weights file
		test_only=True
	render=False
	launch(pa,pg_resume,render=render,repre='image',end='all_done', test_only=test_only)
#	test2(pa)

if __name__ =='__main__':
	main()



