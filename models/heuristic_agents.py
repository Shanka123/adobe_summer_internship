from params import Params
from env import Env
from task_dist import Task_Dist
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

class Heuristic_Agents:
	def __init__(self):
		self.params = Params()

	def get_action(self, env):
		if self.params.scheduling_policy == 'BEST_FIT_PEAK':
			return self.get_best_fit_action_peak(env)
		elif self.params.scheduling_policy == 'WORST_FIT_PEAK':#dominant resource
			return self.get_worst_fit_action_peak(env)
		elif self.params.scheduling_policy == 'WORST_FIT_ONLY':#dominant resource
			return self.get_worst_fit_action1(env)
		elif self.params.scheduling_policy == 'BEST_FIT_ONLY':#dominant resource
			return self.get_best_fit_action1(env)
		elif self.params.scheduling_policy == 'RANDOM':
			return self.get_random_action(env)
		elif self.params.scheduling_policy == 'ANTI_AFFINITY':
			return self.get_anti_affinity_action(env)
		elif self.params.scheduling_policy == 'TETRIS_PACKER':
			return self.get_tetris_packer_action(env)
		elif self.params.scheduling_policy == 'TETRIS_DIST':
			return self.get_tetris_dist_action(env)


	def get_worst_fit_action_peak2(self, env):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		
		new_task = env.waiting_tasks[0]
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			sum_cpu=0
			sum_mem =0
			for c in machine.running_tasks:
				sum_cpu+=c.cpu_limit
				sum_mem+=c.mem_limit
			if (new_task.cpu_limit+sum_cpu <= machine.total_cpus) and (new_task.mem_limit+sum_mem <= machine.total_mems):
				if allocate_to == -1:
					allocate_to = i

				else:
					sum1=0
					sum2=0
					for h in env.machines[allocate_to].running_tasks:
						sum1+=h.cpu_limit 
						sum2+=h.mem_limit 

					if (machine.total_cpus-(sum_cpu) >= env.machines[allocate_to].total_cpus-sum1) and (machine.total_mems-(sum_mem) >= env.machines[allocate_to].total_mems-sum2):
						allocate_to = i
		#	print(i,machine.cpus_left)
		if allocate_to == -1:
			print("*****************")
			return self.params.num_machines #invalid action
		else:
			return allocate_to


	def get_anti_affinity_action(self, env):
		## No waiting tasks
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		
		new_task = env.waiting_tasks[0]
		allocate_to = -1
		scores = [-2]*len(env.machines)
		LAMBDA = 0.5
		for i, machine in enumerate(env.machines):
			total = 2
			if (new_task.cpu_limit <= machine.cpus_left) and (new_task.mem_limit <= machine.mems_left):
				# cpu_factor = (machine.cpus_left-new_task.cpu_limit)
				# mem_factor = (machine.mems_left-new_task.mem_limit)
				## COMPUTES CPU USAGE OF CPU DOMINANT TASKS AND MEM DOMINANT TASKS
				## SIMILARLY THE MEM USAGE
				cpu_cpu_dominant = 0
				cpu_mem_dominant = 0
				mem_cpu_dominant = 0
				mem_mem_dominant = 0
				for task in machine.running_tasks:
					if len(task.cpu_util) != 0:
						cpu_prop = task.cpu_limit/self.params.machine_res_cap[0]
						mem_prop = task.mem_limit/self.params.machine_res_cap[1]
						## CHECK WHICH IS DOMINANT
						if cpu_prop > mem_prop:
							cpu_cpu_dominant += task.cpu_util[-1]
							mem_cpu_dominant += task.mem_util[-1]
						else:
							cpu_mem_dominant += task.cpu_util[-1]
							mem_mem_dominant += task.mem_util[-1]

				## GOT TOTAL COMPUTATION DONE
				assert(cpu_cpu_dominant + cpu_mem_dominant + machine.cpus_left == self.params.machine_res_cap[0])
				## CPU_CPU_DOMINANT + CPU_MEM_DOMINANT + machine.CPUS_LEFT = machine_res_cap[0]
				cpu_score = (cpu_cpu_dominant + LAMBDA*cpu_mem_dominant)/(self.params.machine_res_cap[0]**2)
				mem_score = (mem_mem_dominant + LAMBDA*mem_cpu_dominant)/(self.params.machine_res_cap[1]**2)
				total = ((self.params.machine_res_cap[0] - machine.cpus_left + new_task.cpu_limit)*cpu_score + 
					(self.params.machine_res_cap[1] - machine.mems_left + new_task.mem_limit)*mem_score)
			scores[i] = -total
		index = np.array(scores).argmax()
		if scores[index] == -2:
			return self.params.num_machines
		else:
			# print(scores)
			return index










	def get_random_action(self,env):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		
		new_task = env.waiting_tasks[0]
		return np.random.randint(0,self.params.num_machines)

	# def get_best_fit_action_peak(self, env):
	# 	if len(env.waiting_tasks) == 0:
	# 		return self.params.num_machines #invalid action
		
	# 	new_task = env.waiting_tasks[0]
	# 	allocate_to = -1
	# 	for i, machine in enumerate(env.machines):
	# 		sum_cpu=0
	# 		sum_mem =0
	# 		for c in machine.running_tasks:
	# 			sum_cpu+=c.cpu_limit
	# 			sum_mem+=c.mem_limit
	# 		if (new_task.cpu_limit+sum_cpu <= machine.total_cpus) and (new_task.mem_limit+sum_mem <= machine.total_mems):
	# 			if allocate_to == -1:
	# 				allocate_to = i

	# 			else:
	# 				sum1=0
	# 				sum2=0
	# 				for h in env.machines[allocate_to].running_tasks:
	# 					sum1+=h.cpu_limit 
	# 					sum2+=h.mem_limit 

	# 				if (machine.total_cpus-(sum_cpu) <= env.machines[allocate_to].total_cpus-sum1) and (machine.total_mems-(sum_mem) <= env.machines[allocate_to].total_mems-sum2):
	# 					allocate_to = i
	# 	#	print(i,machine.cpus_left)
	# 	if allocate_to == -1:
	# 		print("*****************")
	# 		return self.params.num_machines #invalid action
	# 	else:
	# 		return allocate_to


	def get_tetris_packer_action(self,env):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines,None,None #invalid action
		for x in env.waiting_tasks:
			print("tasks waiting>>>",x.service)
		new_tasks = env.waiting_tasks
		eligible_tasks=[]
		scores=[]
		allocate_to = -1
		flag=0
		used_machines=[]
		used_machines_idx=[]
		for i, machine in enumerate(env.machines):
			if len(machine.running_tasks)>0:
				used_machines.append(machine)
				used_machines_idx.append(i)
		if len(used_machines)==0:
			# print('First machine added>>')
			used_machines_idx.append(0)
			used_machines.append(env.machines[0])
		if len(used_machines)>0:

			for i, machine in enumerate(used_machines):
				temp1=[]
				temp2=[]
				sum_cpu=0
				sum_mem =0
				for c in machine.running_tasks:
					sum_cpu+=c.cpu_limit
					sum_mem+=c.mem_limit
				for idx,task in enumerate(new_tasks):	
					if (task.cpu_limit+sum_cpu <= machine.total_cpus) and (task.mem_limit+sum_mem <= machine.total_mems):
						temp1.append(idx)
						temp2.append((machine.total_cpus-sum_cpu)*task.cpu_limit + (machine.total_mems-sum_mem)*task.mem_limit)
				eligible_tasks.append(temp1)
				scores.append(temp2)

			for check in eligible_tasks:
				if len(check)>0:
					flag=1
			if flag==1:
				print('No new machine required.Proceed further>>>')
			else:
				# print('New machine alloted. Enjoy>>>')
				used_machines_idx.append(i+1)
				used_machines.append(env.machines[i+1])
				eligible_tasks=[]
				scores=[]
				for i, machine in enumerate(used_machines):
					temp1=[]
					temp2=[]
					sum_cpu=0
					sum_mem =0
					for c in machine.running_tasks:
						sum_cpu+=c.cpu_limit
						sum_mem+=c.mem_limit
					for idx,task in enumerate(new_tasks):	
						if (task.cpu_limit+sum_cpu <= machine.total_cpus) and (task.mem_limit+sum_mem <= machine.total_mems):
							temp1.append(idx)
							temp2.append((machine.total_cpus-sum_cpu)*task.cpu_limit + (machine.total_mems-sum_mem)*task.mem_limit)
					eligible_tasks.append(temp1)
					scores.append(temp2)

		top_task_each_machine=[]
		top_scores_each_machine=[]
		for x,y in enumerate(scores):
			if len(y)>0:
				top_task_each_machine.append(eligible_tasks[x][y.index(max(y))])
				top_scores_each_machine.append(max(y))
			else:
				top_task_each_machine.append(100)
				top_scores_each_machine.append(0)

		print(top_task_each_machine)
		print(top_scores_each_machine)
		# no=top_task_each_machine.count(top_task_each_machine[top_scores_each_machine.index(max(top_scores_each_machine))])
		
			
		
		# print(len(set(top_task_each_machine)))
		if len(set(top_task_each_machine))==1 and top_task_each_machine[0]==100:
			return self.params.num_machines,None,None
		else:
			allocate_to=used_machines_idx[top_scores_each_machine.index(max(top_scores_each_machine))]
			task_scheduled=env.waiting_tasks[top_task_each_machine[top_scores_each_machine.index(max(top_scores_each_machine))]]
			task_id=top_task_each_machine[top_scores_each_machine.index(max(top_scores_each_machine))]
			print('allocated task is ',env.waiting_tasks[top_task_each_machine[top_scores_each_machine.index(max(top_scores_each_machine))]].service)
			# del env.waiting_tasks[top_task_each_machine[top_scores_each_machine.index(max(top_scores_each_machine))]]
			# for x in env.waiting_tasks:
			# 	print("tasks left after scheduling>>>",x.service)
			return allocate_to,task_scheduled,task_id
		#	print(i,machine.cpus_left)
		# if allocate_to == -1:
		# 	print("*****************")
		# 	return self.params.num_machines,None #invalid action
		# else:
			

	def get_tetris_dist_action(self,env):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines,None #invalid action
		
		new_task = env.waiting_tasks[0]
		eligible_machines=[]
		scores=[]
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			sum_cpu=0
			sum_mem =0
			for c in machine.running_tasks:
				sum_cpu+=c.cpu_limit
				sum_mem+=c.mem_limit
			if (new_task.cpu_limit+sum_cpu <= machine.total_cpus) and (new_task.mem_limit+sum_mem <= machine.total_mems):
				eligible_machines.append(i)
				scores.append((machine.total_cpus-sum_cpu)*new_task.cpu_limit + (machine.total_mems-sum_mem)*new_task.mem_limit)

		if len(eligible_machines)==0:
			return self.params.num_machines,None
		if len(eligible_machines)>0:
			allocate_to=eligible_machines[scores.index(max(scores))]

		#	print(i,machine.cpus_left)
		if allocate_to == -1:
			print("*****************")
			return self.params.num_machines,None #invalid action
		else:
			return allocate_to,new_task



	def get_best_fit_action_peak(self, env):#based on dominant resource
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines,None,None #invalid action
		
		new_task = env.waiting_tasks[0]
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			sum_cpu=0
			sum_mem =0
			for c in machine.running_tasks:
				sum_cpu+=c.cpu_limit
				sum_mem+=c.mem_limit
			if (new_task.cpu_limit+sum_cpu <= machine.total_cpus) and (new_task.mem_limit+sum_mem <= machine.total_mems):
				if allocate_to == -1:
					allocate_to = i

				else:
					sum1=0
					sum2=0
					for h in env.machines[allocate_to].running_tasks:
						sum1+=h.cpu_limit 
						sum2+=h.mem_limit 
					if new_task.cpu_limit>new_task.mem_limit:
						if (machine.total_cpus-(sum_cpu) <= env.machines[allocate_to].total_cpus-sum1):
							allocate_to = i
					elif new_task.mem_limit>new_task.cpu_limit:
						if (machine.total_mems-(sum_mem) <= env.machines[allocate_to].total_mems-sum2):
							allocate_to=i
		#	print(i,machine.cpus_left)
		if allocate_to == -1:
			print("*****************")
			return self.params.num_machines,None,None #invalid action
		else:
			return allocate_to,new_task,0



	def get_worst_fit_action_peak(self, env):#based on dominant resource
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines,None #invalid action
		
		new_task = env.waiting_tasks[0]
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			sum_cpu=0
			sum_mem =0
			for c in machine.running_tasks:
				sum_cpu+=c.cpu_limit
				sum_mem+=c.mem_limit
			if (new_task.cpu_limit+sum_cpu <= machine.total_cpus) and (new_task.mem_limit+sum_mem <= machine.total_mems):
				if allocate_to == -1:
					allocate_to = i

				else:
					sum1=0
					sum2=0
					for h in env.machines[allocate_to].running_tasks:
						sum1+=h.cpu_limit 
						sum2+=h.mem_limit 
					if new_task.cpu_limit>new_task.mem_limit:
						if (machine.total_cpus-(sum_cpu) >= env.machines[allocate_to].total_cpus-sum1):
							allocate_to = i
					elif new_task.mem_limit>new_task.cpu_limit:
						if (machine.total_mems-(sum_mem) >= env.machines[allocate_to].total_mems-sum2):
							allocate_to=i
		#	print(i,machine.cpus_left)
		if allocate_to == -1:
			print("*****************")
			return self.params.num_machines,None #invalid action
		else:
			return allocate_to,new_task

	def get_worst_fit_action1(self, env):#based on dominant resource
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines,None #invalid action
		
		new_task = env.waiting_tasks[0]
		flag1=0
		flag2=0
		maximum1=-10000
		maximum2=-10000
		minimum3=10000
		allocate_to = -1
		for i, machine in enumerate(env.machines):
				if (machine.cpus_left>0) and (machine.mems_left>0):
					if new_task.cpu_limit>new_task.mem_limit:
						flag1=1
						if (machine.cpus_left >= maximum1) and len(machine.running_tasks)<=minimum3:

							allocate_to = i
							maximum1=machine.cpus_left
							minimum3=len(machine.running_tasks)
					elif new_task.mem_limit>new_task.cpu_limit:
						flag2=1
						if (machine.mems_left >= maximum2) and len(machine.running_tasks)<=minimum3:

							allocate_to = i
							maximum2=machine.mems_left
							minimum3=len(machine.running_tasks)

					#maximum2=machine.mems_left
	#	if flag1==1 and flag2==0:
	#		print('cpu dominant')
	#	else:
	#		print('mem dominant')
		if allocate_to == -1:
			return self.params.num_machines,None #invalid action
		else:
			return allocate_to,new_task






	def get_best_fit_action(self, env):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		
		new_task = env.waiting_tasks[0]
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			if new_task.cpu_limit <= machine.cpus_left:
				if allocate_to == -1:
					allocate_to = i
				elif machine.cpus_left < env.machines[allocate_to].cpus_left:
					allocate_to = i
		if allocate_to == -1:
			return self.params.num_machines #invalid action
		else:
			return allocate_to




	def get_best_fit_action1(self, env):#based on dominant resource
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		
		new_task = env.waiting_tasks[0]
		flag1=0
		flag2=0
		minimum1=10000
		minimum2=10000
		maximum3=-10000
		allocate_to = -1
		for i, machine in enumerate(env.machines):
				if (machine.cpus_left>0) and (machine.mems_left>0):
					if new_task.cpu_limit>new_task.mem_limit:
						flag1=1
						if (machine.cpus_left <= minimum1) and len(machine.running_tasks)>=maximum3:

							allocate_to = i
							minimum1=machine.cpus_left
							maximum3=len(machine.running_tasks)
					elif new_task.mem_limit>new_task.cpu_limit:
						flag2=1
						if (machine.mems_left <= minimum2) and len(machine.running_tasks)>=maximum3:

							allocate_to = i
							minimum2=machine.mems_left
							maximum3=len(machine.running_tasks)

					#maximum2=machine.mems_left
	#	if flag1==1 and flag2==0:
	#		print('cpu dominant')
	#	else:
	#		print('mem dominant')
		if allocate_to == -1:
			return self.params.num_machines #invalid action
		else:
			return allocate_to

	def get_worst_fit_action(self, env):
		if len(env.waiting_tasks) == 0:
			return self.params.num_machines #invalid action
		
		new_task = env.waiting_tasks[0]
		allocate_to = -1
		for i, machine in enumerate(env.machines):
			if (new_task.cpu_limit <= machine.cpus_left) and (new_task.mem_limit <= machine.mems_left) :
				if allocate_to == -1:
					allocate_to = i
				elif (machine.cpus_left > env.machines[allocate_to].cpus_left) and (machine.mems_left > env.machines[allocate_to].mems_left):
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
	for i in range(pa.num_machines*2):
		mat[:,pa.machine_res_cap[0]*(i+1)+i]=1
	

	new_mat = np.ones((mat.shape[0]+2, mat.shape[1]+2))
	new_mat[1:-1, 1:-1] = mat

#	plt.figure(figsize=(20,20))
	plt.figure("STATE")
	plt.xlabel('RESOURCE')
	plt.ylabel('TIME')
	plt.imshow(new_mat, interpolation='nearest', cmap=cmap, norm=norm)
	if path is not None:
		plt.savefig(str(path+'.jpg'))
	plt.pause(0.01)

def main():
	agent = Heuristic_Agents()

	env = Env(0, 1)
	task_dist = Task_Dist()
	delay=0
	workloads = task_dist.gen_seq_workload()
	logs = open('/home/dell/logs_cpu_mem_tetris_23ex', 'a')
	no_machines=[]
	logline = str(0) + '\n'
	ex_indices=[]
	for ex in range(agent.params.num_ex):
		env.reset()
		env.workload_seq = workloads[ex]
	#	for i in range(len(workloads[ex+agent.params.num_ex])):
		env.generate_workload()
#		env.seq_id += 1
		print('Testing : ', env.workload_seq)

		ob = env.observe()
		rews = []
		acts = []
		cpu_crs = [0]*10
		cpu_crs_max=[0]*10
		mem_crs = [0]*10
		mem_crs_max=[0]*10
		c_utils = ''
		m_utils = ''
		suffer  = []
		np.random.seed(20)
		for _ in range(agent.params.episode_max_length):
			a,task,task_id = agent.get_action(env)
			acts.append(a)
			#if ex==0:
			#	plt1 = visualize_state(ob, agent.params, '/home/dell/trajs/worst_fit20/episode_%d' % int(_))
			ob, rews, done, status= env.step(a, _, rews,task_id)
			if status == 'Allocation_Success':

				finished_episode_len = _ + 1
			#	print('Example>>>',ex)
			#	print('Service Name>>',task.service)
			#	print('Entry time>>',task.enter_time)
				delay+=task.start_time-task.enter_time
			#	print('Scheduling Time>>',task.start_time)
			#	print('Delay>>>>',task.start_time - task.enter_time)
			if done:
				break



			c_util = ''
			m_util = ''
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.cpus_left >= 0:
						c_util+=str(machine.total_cpus - machine.cpus_left) +','
					else:
						c_util+=str(machine.total_cpus+abs(machine.cpus_left)) +','
						suffer.append(abs(machine.cpus_left))
				else:
					c_util += str(0) + ','
			for k, machine in enumerate(env.machines):
				if len(machine.running_tasks) > 0:
					if machine.mems_left >= 0:
						m_util+=str(machine.total_mems - machine.mems_left) +','
					else:
						m_util+=str(machine.total_mems+abs(machine.mems_left)) +','
						suffer.append(abs(machine.mems_left))
				else:
					m_util += str(0) + ','

				cpu_crs_this_time = [0]*agent.params.num_machines
				mem_crs_this_time = [0]*agent.params.num_machines

				for i in range(len(machine.running_tasks)):
					for j in range(i+1, len(machine.running_tasks)):
						task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
						if task_i != task_j and len(task_i.cpu_util)>0 and len(task_j.cpu_util)>0:
							cpu_crs[k] += agent.params.interference_penalty_cpu * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
							cpu_crs_this_time[k] += agent.params.interference_penalty_cpu * (task_i.cpu_util[-1] * task_j.cpu_util[-1]) * (-1)
						if task_i != task_j and len(task_i.mem_util)>0 and len(task_j.mem_util)>0:
							mem_crs[k] += agent.params.interference_penalty_mem * (task_i.mem_util[-1] * task_j.mem_util[-1]) * (-1)
							mem_crs_this_time[k] += agent.params.interference_penalty_mem * (task_i.mem_util[-1] * task_j.mem_util[-1]) * (-1)
				cpu_crs_max[k] = max(cpu_crs_max[k], cpu_crs_this_time[k])
				mem_crs_max[k] = max(mem_crs_max[k], mem_crs_this_time[k])
				#################
			c_utils += c_util + '|'
			m_utils += m_util + '|'
	
		logline += str(str(_ -1)+'|'+str(c_utils) + str(finished_episode_len)) + '\n' + str(sum(rews)) + '\n' + str(sum(suffer))  +'\n'
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
		print('Test Actions: ', acts[:finished_episode_len])
		print('Reward : ', rews)
	
		print('Reward : ', sum(rews))
		print('Number of machines used>>',len(set(acts[:finished_episode_len]))-1)

		no_machines.append(len(set(acts[:finished_episode_len]))-1)
		if len(set(acts[:finished_episode_len]))-1 ==8 or len(set(acts[:finished_episode_len]))-1 ==9 :
			ex_indices.append(ex)


	print('Average Delay>>>',delay/agent.params.num_test_ex)
	print('Number of examples with 6 machines>>>',no_machines.count(6))
	print('Number of examples with 7 machines>>>',no_machines.count(7))
	print('Number of examples with 8 machines>>>',no_machines.count(8))
	print('Number of examples with 9 machines>>>',no_machines.count(9))
	print('Number of examples with 10 machines>>>',no_machines.count(10))
	logs.write(logline)
	logs.flush()
	# import pickle
	# with open("workloads8_9_threecombo_68_random_indices.txt", "wb") as fp:   #Pickling
	# 	pickle.dump(ex_indices, fp)
	os.fsync(logs.fileno())

if __name__ =='__main__':
	main()
