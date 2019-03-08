import numpy as np
from params import Params 

class Machine:
	def __init__(self, mid, cpus,mems):
		self.mid = mid + 1
		self.params = Params()
		self.total_cpus = cpus
		self.cpus_left = cpus
		self.total_mems = mems
		self.mems_left = mems  #initially eq to total cpu
		self.running_tasks = []

		# graphical representation
		self.canvas1 = np.zeros((1, self.params.hist_wind_len, self.params.machine_res_cap[0]))
		self.canvas2 = np.zeros((1, self.params.hist_wind_len, self.params.machine_res_cap[1]))
	#	self.canvas=[self.canvas1,self.canvas2]


	#update machine current utilzation, remove finished tasks 	
	def update(self, task_dist, cur_time):
		unfinished_tasks = []
		for task in self.running_tasks:
			#query model(currntly using the fixed csv having each task resource utilization) to get various resource util depending on current state and pass it
			task.update(task_dist.get_cpu_usage(task, cur_time - task.start_time),task_dist.get_mem_usage(task, cur_time - task.start_time))
			#remove the finished tasks
			if(not task.is_complete(cur_time)):
				unfinished_tasks.append(task)
		self.running_tasks = unfinished_tasks
		#update representation
		self.canvas1[:, :-1, :] = self.canvas1[:, 1:, :]
		self.canvas1[:, -1, :] = 0
		self.canvas2[:, :-1, :] = self.canvas2[:, 1:, :]
		self.canvas2[:, -1, :] = 0
		#update current utilization of machine
		self.cpus_left = self.total_cpus
		self.mems_left = self.total_mems
		for tasks in self.running_tasks:
				self.cpus_left -= tasks.cpu_util[-1]
				self.mems_left -= tasks.mem_util[-1]
		#for res in range(self.params.num_res):
		used_res1 = 0
		used_res2 =0
		for task in self.running_tasks:
		#	print('color',task.color)
		#	print(int(task.cpu_util[-1]))
			self.canvas1[0, -1, used_res1:used_res1 + int(task.cpu_util[-1])] = self.mid + task.color
			self.canvas2[0, -1, used_res2:used_res2 + int(task.mem_util[-1])] = self.mid + task.color
			used_res1 += int(task.cpu_util[-1])
			used_res2 += int(task.mem_util[-1])
	#	self.canvas=[self.canvas1,self.canvas2]
	#return true if it is possible to allocate a task to this machine based on cpus_left
	def allocate_task(self, task, episode_time):
		if self.cpus_left > 0 and self.mems_left>0:
			# print('Scheduled')
			task.episode_time = episode_time
			task.conf_at_scheduling = self.running_tasks.copy()
			self.running_tasks.append(task)
			return True
		else:
			return False
