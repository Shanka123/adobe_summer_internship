class Task:
	def __init__(self, service, color, cpu_limit, cpu_req, enter_time):
		self.service = service
		self.color = color
		self.cpu_limit = cpu_limit
		self.enter_time = enter_time
		self.start_time = -1 #assigned when task will be scheduled
		self.cpu_req = cpu_req #cpu required to complete
		self.cpu_util = [] #cpu util from start at regular interval
		self.episode_time = -1
		self.conf_at_scheduling = None
		self.already_overshoot = False

	def update(self, cpu_util):
		if cpu_util != -1:
			self.cpu_util.append(int(cpu_util)) #find utilzation based on data/model

	def is_complete(self, cur_time):
		#cpu_time_spent = sum(self.cpu_util)*len(self.cpu_util)
		return (cur_time - self.start_time) > self.cpu_req
