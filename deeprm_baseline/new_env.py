import numpy as np
from task import Task
from parameters import Parameters
from task_dist import Task_Dist
from machine import Machine
from task_dist import Task_Dist

class Env1:
	def __init__(self, cur_time, time_step):
		self.params = Parameters()
		self.cur_time = cur_time
		self.time_step = time_step
		self.machines = []
		self.waiting_tasks = []
		#initialize machines
		for i in range(self.params.num_machines):
			self.machines.append(Machine(i, self.params.machine_res_cap))

		self.task_dist = Task_Dist()
		self.workload_seq = None
		self.seq_id = 0

	#Generate workload and populate self.waiting tasks after each interval
	def generate_workload(self):
		if len(self.workload_seq) <= self.seq_id:
			return

	#	print('Incoming Tasks: ',self.seq_id, self.workload_seq[self.seq_id])
		if self.workload_seq[self.seq_id] :
			for task_type in self.workload_seq[self.seq_id]:
				task_color, task_cpu_limit, task_finish_time= self.task_dist.get_task_details(task_type)

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
				
				self.waiting_tasks.append(Task(task_type, task_color, task_cpu_limit, task_finish_time, self.cur_time))

	def observe(self):
		img_repr = np.zeros((self.params.state_len, self.params.state_width))
		#add machines
		used_width = 0
		for res in range(self.params.num_res):
			for machine in self.machines:
				img_repr[:, used_width: used_width+self.params.machine_res_cap] = machine.canvas[res, :, :]
				used_width += self.params.machine_res_cap
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

		return img_repr

	def step(self, action, episode_time, rewards,c, log=False):
		status = None
		done = False
		reward = 0
		
		if len(self.waiting_tasks) == 0:
			status = 'Backlog_Empty'
			
		elif action == self.params.num_machines:
			status = 'Invalid'
		else:
			allocated = self.machines[action].allocate_task(self.waiting_tasks[c], episode_time)
			if allocated:
				status = 'Allocation_Success'
				self.waiting_tasks[c].start_time = self.cur_time
				self.waiting_tasks = self.waiting_tasks[0:c] +self.waiting_tasks[c+1:]
			else:
				status = 'Allocation_Failed'

		if status == 'Allocation_Success' or 'Invalid' or 'Allocation_Failed' or 'Backlog_Empty':
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
			rewards = self.get_reward(rewards, log)
			self.generate_workload()
			# if status == 'Allocation_Failed':
			# 	reward += -100
			# if status == 'No_More_Jobs' and action < self.params.num_machines:
			# 	reward += -100
			# if len(self.waiting_tasks) != 0 and action == self.params.num_machines:
			# 	reward += -100

		ob = self.observe()
		if done:
			self.reset()
		return ob, rewards, done, status

	def reset(self):
		self.cur_time = 0
		self.machines = []
		self.waiting_tasks = []
		for i in range(self.params.num_machines):
			self.machines.append(Machine(i, self.params.machine_res_cap))
		self.seq_id = 0
		#self.workload_seq = self.task_dist.gen_seq_workload()

	def get_suitable_machines(self, task):
		return [machine for machine in self.machines if machine.cpus_left > task.cpu_limit]

	def update(self):
		self.cur_time += self.time_step
		for machine in self.machines:
			machine.update(self.task_dist, self.cur_time)

	def schedule(self):
		unscheduled_tasks = []
		for task in self.waiting_tasks:
			# suitable_machines = self.get_suitable_machines(task)
			# if not suitable_machines:
			# 	break

			machine = self.find_best_machine(task, suitable_machines)
			if(machine is not None):
				task.start_time = self.cur_time
				machine[0].allocate_task(task)
			else:
				unscheduled_tasks.append(task)
		self.waiting_tasks = unscheduled_tasks

	def find_best_machine(self, task, suitable_machines):
		if np.random.randint(2):
			return np.random.choice(
				suitable_machines,
				1,
				p = [1/len(suitable_machines)]*len(suitable_machines)
			)
		else:
			return None
		
	def get_reward(self, rewards, log):
		rewards.append(0)
		#Penaly for putting a task on hold
		rewards[-1] += self.params.hold_penalty * len(self.waiting_tasks)/(self.params.state_len*self.params.state_width) #TODO add some penaly factor
		# print('Hold : ', self.params.hold_penalty * len(self.waiting_tasks))
		#Penalty using cross co-relation of cpu_util
		for i, machine in enumerate(self.machines):
			# print('cpus left:', machine.cpus_left)
			if len(machine.running_tasks) == 1 and len(machine.running_tasks[0].cpu_util) == 1:
				# print(i+1, 'New machine allocated')
				rewards[-1] += self.params.machine_used_penalty/(self.params.state_len*self.params.state_width)
			if machine.cpus_left < 0 and not machine.running_tasks[-1].already_overshoot:
				if log:
					print(i+1, 'Overshoot', self.params.overshoot_penalty)
				machine.running_tasks[-1].already_overshoot = True
				rewards[machine.running_tasks[-1].episode_time] += self.params.overshoot_penalty/(self.params.state_len*self.params.state_width)
			for task in machine.running_tasks:
				for tsk in task.conf_at_scheduling:
					if tsk in machine.running_tasks:
						rewards[task.episode_time] += self.params.interference_penalty * (task.cpu_util[-1] * tsk.cpu_util[-1])/(self.params.state_len*self.params.state_width)
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
			
