import numpy as np
from task import Task
from params import Params
from task_dist import Task_Dist
from machine import Machine
from task_dist import Task_Dist

class Env:
	def __init__(self, cur_time, time_step):
		self.params = Params()
		self.cur_time = cur_time
		self.time_step = time_step
		self.machines = []
		self.waiting_tasks = []
		#initialize machines
		for i in range(self.params.num_machines):
			self.machines.append(Machine(i, self.params.machine_res_cap[0],self.params.machine_res_cap[1]))

		self.task_dist = Task_Dist()
		self.workload_seq = None
		self.seq_id = 0

	#Generate workload and populate self.waiting tasks after each interval
	def generate_workload(self):
		if len(self.workload_seq) <= self.seq_id:
			return

	#	print('Incoming Tasks: ',self.seq_id, self.workload_seq[self.seq_id])
		if self.workload_seq[self.seq_id]:
			for task_type in self.workload_seq[self.seq_id]:
				task_color, task_cpu_limit,task_mem_limit ,task_finish_time= self.task_dist.get_task_details(task_type)

				max_color = task_color
				for tsk in self.waiting_tasks:
					if task_type == tsk.service:
						if max_color <= tsk.color:
							max_color = tsk.color
				for mcn in self.machines:
					for tsk in mcn.running_tasks:
						if task_type == tsk.service and max_color <= tsk.color:
							max_color = tsk.color
				task_color = max_color + 0.01
				
				self.waiting_tasks.append(Task(task_type, task_color, task_cpu_limit,task_mem_limit, task_finish_time, self.cur_time))

	#return the state of environment as 2D-matrix
	def observe(self):
		img_repr = np.zeros((self.params.state_len, self.params.state_width))
		#add machines
		used_width = 0
		for res in range(self.params.num_res):
			for machine in self.machines:
			#	print(res,machine.canvas[res][0,:,:].shape)
				if res==0:
					img_repr[:, used_width: used_width+self.params.machine_res_cap[res]] = machine.canvas1[0,:,:]
				else:
					img_repr[:, used_width: used_width+self.params.machine_res_cap[res]] = machine.canvas2[0,:,:]
			#	print('image',img_repr)
				used_width += self.params.machine_res_cap[res] +1
		#add backlog queue
		if len(self.waiting_tasks) > 0:
			t = 0
			for i in range(self.params.state_len):
				for j in range(self.params.backlog_width):
					img_repr[i, used_width + j] = self.waiting_tasks[t].color
					t += 1
					if(t == len(self.waiting_tasks)):
						break
				if(t == len(self.waiting_tasks)):
					break

		used_width += self.params.backlog_width
		assert used_width == self.params.state_width

		k=-1
		for res in range(self.params.num_res):
			for machine in self.machines:
				k+=self.params.machine_res_cap[res]+1
				j=0
				
				for m in machine.running_tasks:
					img_repr[j,k]=m.color+machine.mid
					j+=1

		return img_repr

	#changes the state of system by taking the action and returns the rewards, new state due to that action
	def step(self, action, episode_time, rewards,task_no):
		status = None
		done = False
		reward = 0

		if len(self.waiting_tasks) == 0:
			status = 'Backlog_Empty'
		elif action == self.params.num_machines:
			status = 'Invalid'
		else:
			allocated = self.machines[action].allocate_task(self.waiting_tasks[0], episode_time)
			if allocated:
				status = 'Allocation_Success'
			#	print('Current Time>>',self.cur_time)
				self.waiting_tasks[task_no].start_time = self.cur_time
				self.waiting_tasks = self.waiting_tasks[0:task_no]+self.waiting_tasks[task_no+1:]
			else:
				status = 'Allocation_Failed'

		if (status == 'Invalid') or (status=='Allocation_Failed') or (status=='Backlog_Empty'):
			self.seq_id += 1
			# self.generate_workload()
			self.update()
			#TODO fix max no of jobs, so when all jobs complete episode ends
			unfinished = 0
			for machine in self.machines:
				if len(machine.running_tasks) != 0:
					unfinished += 1
			if unfinished == 0 and len(self.waiting_tasks) == 0:
				done = True
			if self.cur_time > self.params.episode_max_length:  # run too long, force termination
				done = True
			rewards = self.get_reward(rewards)
			self.generate_workload()
			# if status == 'Allocation_Failed':
			# 	reward += -100
			# if status == 'No_More_Jobs' and action < self.params.num_machines:
			# 	reward += -100
			# if len(self.waiting_tasks) != 0 and action == self.params.num_machines:
		
			# 	reward += -100

		else:
			unfinished = 0
			for machine in self.machines:
				if len(machine.running_tasks) != 0:
					unfinished += 1
			if unfinished == 0 and len(self.waiting_tasks) == 0:
				done = True
			if self.cur_time > self.params.episode_max_length:  # run too long, force termination
				done = True
			rewards = self.get_reward(rewards)

		ob = self.observe()
		if done:
			self.reset()
		
		return ob, rewards, done, status


	#reset the system by making all machines empty and time to 0
	def reset(self):
		self.cur_time = 0
		self.machines = []
		self.waiting_tasks = []
		for i in range(self.params.num_machines):
			self.machines.append(Machine(i, self.params.machine_res_cap[0],self.params.machine_res_cap[1]))
		self.seq_id = 0

	#update time, status of running tasks in every machine
	def update(self):
		self.cur_time += self.time_step
		for machine in self.machines:
			machine.update(self.task_dist, self.cur_time)
	
	
			
	def get_reward(self, rewards):
		rewards.append(0)
		
		#Penaly for putting a task on hold
		rewards[-1] += self.params.hold_penalty * len(self.waiting_tasks) #TODO add some penaly factor
		# print('Hold : ', self.params.hold_penalty * len(self.waiting_tasks))
		#Penalty using cross co-relation of cpu_util
		for i, machine in enumerate(self.machines):
			tasks=[]
			# print('cpus left:', machine.cpus_left)
			if len(machine.running_tasks) > 0 and machine.cpus_left > 0:
				# print(i+1, 'New machine allocated')
				rewards[-1] += (-1) * pow(machine.cpus_left ,self.params.machine_used_penalty)
			if len(machine.running_tasks) > 0 and machine.mems_left > 0:
				# print(i+1, 'New machine allocated')
				rewards[-1] += (-1) * pow(machine.mems_left ,self.params.machine_used_penalty)
		#	if len(machine.running_tasks) == 1 and len(machine.running_tasks[0].cpu_util) == 0:
		#		print(i+1, 'New machine allocated')
		#		rewards[-1] += self.params.machine_used_penalty
			for j,task in enumerate(reversed(machine.running_tasks)):
				tasks.append(task)
				if j==0:
					if machine.cpus_left < 0 and not task.already_overshoot_cpu:
						print(i+1, 'OvershootA_CPU', abs(machine.cpus_left))
						task.already_overshoot_cpu = True
						rewards[task.episode_time] += self.params.overshoot_penalty
				else:
					sum=0
					for m in tasks[0:j]:
						if len(m.cpu_util)>0:
							sum+=m.cpu_util[-1]
					if (machine.cpus_left +sum) < 0 and not task.already_overshoot_cpu:
						print(i+1, 'OvershootB_CPU', abs(machine.cpus_left+sum))
						task.already_overshoot_cpu = True
						rewards[task.episode_time] += self.params.overshoot_penalty		

			tasks=[]
			for j,task in enumerate(reversed(machine.running_tasks)):
				tasks.append(task)
				if j==0:
					if machine.mems_left < 0 and not task.already_overshoot_mem:
						print(i+1, 'OvershootA_MEM', abs(machine.mems_left))
						task.already_overshoot_mem = True
						rewards[task.episode_time] += self.params.overshoot_penalty
				else:
					sum=0
					for m in tasks[0:j]:
						if len(m.mem_util)>0:
							sum+=m.mem_util[-1]
					if (machine.mems_left +sum) < 0 and not task.already_overshoot_mem:
						print(i+1, 'OvershootB_MEM', abs(machine.mems_left+sum))
						task.already_overshoot_mem = True
						rewards[task.episode_time] += self.params.overshoot_penalty		
		
			for task in machine.running_tasks:
				for tsk in task.conf_at_scheduling:
					if tsk in machine.running_tasks:
						if len(task.cpu_util)>0 and len(tsk.cpu_util)>0:
							rewards[task.episode_time] += self.params.interference_penalty_cpu * (task.cpu_util[-1] * tsk.cpu_util[-1])
						if len(task.mem_util)>0 and len(tsk.mem_util)>0:
							rewards[task.episode_time] += self.params.interference_penalty_mem * (task.mem_util[-1] * tsk.mem_util[-1])							
						# print('Other : ', self.params.interference_penalty * (task.cpu_util[-1] * tsk.cpu_util[-1]))
			# for i in range(len(machine.running_tasks)):
			# 	for j in range(i+1, len(machine.running_tasks)):
			# 		task_i, task_j = machine.running_tasks[i], machine.running_tasks[j]
			# 		# if task_i != task_j and len(task_i.cpu_util) > self.params.hist_wind_len and len(task_j.cpu_util) > self.params.hist_wind_len:
			# 		# 	reward += self.params.interference_penalty * (np.correlate(task_i.cpu_util[-self.params.hist_wind_len:], task_j.cpu_util[-self.params.hist_wind_len:]))
			# 		if task_i != task_j:
			# 			# m = min(self.params.hist_wind_len, min(len(task_i.cpu_util), len(task_j.cpu_util)))
			# 			# reward += self.params.interference_penalty * (np.correlate(task_i.cpu_util[-m:], task_j.cpu_util[-m:]))
			# 			reward += self.params.interference_penalty * (task_i.cpu_util[-1] * task_j.cpu_util[-1])
		return rewards
			