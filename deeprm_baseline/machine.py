import numpy as np
from parameters import Parameters
class Machine:
	def __init__(self, mid, cpus):
		self.mid = mid + 1
		self.params = Parameters()
		self.total_cpus = cpus
		self.cpus_left = cpus #initially eq to total cpu
		self.running_tasks = []

		# graphical representation
		self.canvas = np.zeros((self.params.num_res, self.params.hist_wind_len, self.params.machine_res_cap))

	def update(self, task_dist, cur_time):
		unfinished_tasks = []
		for task in self.running_tasks:
			#query model to get various resource util depending on current state and pass it
			task.update(task_dist.get_cpu_usage(task, cur_time - task.start_time))
			if(not task.is_complete(cur_time)):
				unfinished_tasks.append(task)
			else:
				self.cpus_left += task.cpu_limit
		self.running_tasks = unfinished_tasks
		#update representation
		self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
		self.canvas[:, -1, :] = 0
		self.cpus_left = self.total_cpus
		for tasks in self.running_tasks:
			self.cpus_left -= tasks.cpu_util[-1]
		for res in range(self.params.num_res):
			used_res = 0
			for task in self.running_tasks:
				self.canvas[res, -1, used_res:used_res + int(task.cpu_util[-1])] = self.mid + task.color
				used_res += int(task.cpu_util[-1])

	def allocate_task(self, task, episode_time):
		if self.cpus_left > 0:
			# print('Scheduled')
			task.episode_time = episode_time
			task.conf_at_scheduling = self.running_tasks.copy()
			self.running_tasks.append(task)
			return True
		else:
			return False
