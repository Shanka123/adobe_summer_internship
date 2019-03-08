class Params:
	def __init__(self):
		self.num_res = 2 #types of resources, currently only 1
		self.num_machines = 10 #no of machines
		self.machine_res_cap = [8,8]#capacity of machine in terms of a particular resource
		self.hist_wind_len = 20 #intervals for which we need to keep track of task 
		self.backlog_len = 20 #waiting queue for tasks
		self.backlog_width = self.backlog_len // self.hist_wind_len
		self.num_distinct_tasks = 30 # distinct types of tasks
		self.state_width = (self.num_machines * self.machine_res_cap[0]+self.num_machines * self.machine_res_cap[1]) +20+ self.backlog_width # machines' resources + backlog 
		self.state_len = self.hist_wind_len #height of state
		self.episode_max_length = 200 # max time for which a apisode last
		self.hold_penalty = -50 #penalty for holding a job for delaying it a timestep
		self.interference_penalty_cpu = -0.1
		self.interference_penalty_mem = -0.1
		
		self.machine_used_penalty = 3.5
		self.overshoot_penalty = -30000 # every time more resources are used than available in machine
		self.num_actions=self.num_machines
		self.num_frames = 1
		self.lr_rate = 0.001          # learning rate
		self.rms_rho = 0.9            # for rms prop
		self.rms_eps = 1e-9           # for rms prop
		self.num_epochs= 2000
		self.num_ex= 23 # number of training examples
		self.num_test_ex =18 #number of test examples
		self.num_seq_per_batch= 20 #number of trajectories
		self.discount= 1
		self.batch_size = 1#nunber examples to run in parallel
		self.scheduling_policy = 'TETRIS_PACKER'
		self.json_logs = '/tmp/'
		self.detailed_logs = '/tmp/'
		self.task_details_file = '~/simulation_time_series_new.csv'
		self.debug = False
		self.render = False
		self.pg_resume = None #path from where weights to be loaded
		self.render = False
