
from parameters import Parameters
from new_env import Env1
from task_dist import Task_Dist
import csv

from multiprocessing import Process
from multiprocessing import Manager
import os,json

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

class Heuristic_Agents:
	def __init__(self):
		self.params = Parameters()

	def get_action(self, env,c):
		if self.params.scheduling_policy == 'BEST_FIT':
			return self.get_best_fit_action(env,c)
		elif self.params.scheduling_policy == 'WORST_FIT':
			return self.get_worst_fit_action(env,c)

	def get_best_fit_action(self, env,c):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		elif c <len(env.waiting_tasks):
			new_task = env.waiting_tasks[c]
		else:
			return self.params.num_machines

		allocate_to = -1
		for i, machine in enumerate(env.machines):
			if machine.cpus_left >= new_task.cpu_limit:
				if allocate_to == -1:
					allocate_to = i
				elif machine.cpus_left < env.machines[allocate_to].cpus_left:
					allocate_to = i
		if allocate_to == -1:
			return self.params.num_machines #invalid action
		else:
			return allocate_to

	def get_worst_fit_action(self, env,c):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		elif c <len(env.waiting_tasks):
			new_task = env.waiting_tasks[c]
		else:
			return self.params.num_machines
		
		
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			if machine.cpus_left >= new_task.cpu_limit:
				if allocate_to == -1:
					allocate_to = i
				elif machine.cpus_left > env.machines[allocate_to].cpus_left:
					allocate_to = i
		if allocate_to == -1:
			return self.params.num_machines #invalid action
		else:
			return allocate_to

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
	plt.pause(0.1)

def test(pa):
	def ex_test(pg_learner, env, pa, result):
		env.reset()
		env.generate_workload()
		ob = env.observe()
		acts = []
		probs = []
		crs = [0]*pa.num_machines
		crs_max = [0]*pa.num_machines
		rews = []
		utils = ''
		suffer = []
		finished_episode_len = 0
		logline = ''
		for _ in range(pa.episode_max_length):
			a = pg_learner.get_action(env)
			ob, rews, done, status= env.step(a, _, rews)
			acts.append(a)
			if status == 'Allocation_Success':
				finished_episode_len = _ + 1
			if done:
				break
			##############logs
			util = ''
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.cpus_left >= 0:
						util += str(machine.total_cpus - machine.ac_cpus_left) + ','
					else:
						util += str(machine.total_cpus)  + ','
					if machine.ac_cpus_left < 0:	
						suffer.append(abs(machine.ac_cpus_left))
				else:
					util += str(0) + ','
				crs_this_time = [0]*pa.num_machines
				for i in range(len(machine.running_tasks)):
					for j in range(i+1, len(machine.running_tasks)):
						task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
						if task_i != task_j:
							crs[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
							crs_this_time[k] += pa.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
				crs_max[k] = max(crs_max[k], crs_this_time[k])
				#################
			utils += util + '|'
		logline += str(str(_-1)+'|'+str(utils) + str(finished_episode_len)) + '\n' + str(sum(rews)) + '\n' + str(sum(suffer))  +'\n'
		for i in range(len(env.machines)):
			logline += str(crs[i]) + ','
		logline = logline[:-1] + '\n'
		for i in range(len(env.machines)):
			logline += str(crs_max[i]) + ','
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
		pg_learner = Heuristic_Agents()
		pg_learners.append(pg_learner)

	logs = open('/tmp/logs3', 'a')
	loglines = ''
	for it in range(2, pa.num_epochs+1, 2):
		if(it % 10):
			print('Iteration : ',it)

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

def launch():
	agent = Heuristic_Agents()

	env = Env(0, 1)
	task_dist = Task_Dist()
	workloads = task_dist.gen_seq_workload()

	for ex in range(agent.params.num_ex):
		env.reset()
		env.workload_seq = workloads[ex]
		for i in range(len(workloads[ex])):
			env.generate_workload()
			env.seq_id += 1
		print('Testing : ', env.workload_seq)

		ob = env.observe()
		rews = []
		acts = []
		for _ in range(agent.params.episode_max_length):
			a = agent.get_action(env)
			plt1 = visualize_state(ob, agent.params, '/tmp/trajs/episode_%d' % int(_))
			ob, rews, done, status= env.step(a, _, rews)
			if status == 'Allocation_Success':
				finished_episode_len = _ + 1
			if done:
				break
		print('Test Actions: ', acts[:finished_episode_len])
		print('Reward : ', rews)
		print('Reward : ', sum(rews))

if __name__ =='__main__':
	pa = Params()
	launch()
#	test(pa)
