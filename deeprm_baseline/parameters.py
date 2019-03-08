import numpy as np
import math

# import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = '/home/ubuntu/deeprm_multimac_pack/'

        self.num_epochs = 1000         # number of training epochs
        self.simu_len = 50             # length of the busy cycle that repeats itself
        self.num_ex = 99                # number of sequences
        self.num_test_ex=33
        self.output_freq = 10          # interval for output and store parameters
        self.num_machines = 5 #no of machines
        self.machine_res_cap = 24 #capacity of machine in terms of a particular resource
        self.hist_wind_len = 10 #intervals for which we need to keep track of task 
        self.backlog_len = 10 #waiting queue for tasks
        self.backlog_width = self.backlog_len // self.hist_wind_len
        self.num_distinct_tasks = 30 # distinct types of tasks
        self.state_width = (self.num_machines * self.machine_res_cap) + self.backlog_width # machines' resources + backlog 
        self.state_len = 10
        
        self.hold_penalty = -50
        self.interference_penalty = -0.1
        self.machine_used_penalty = -4000
        self.overshoot_penalty = -8000
        self.num_actions=self.num_machines
        self.num_frames = 1
        





        self.num_seq_per_batch = 20    # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_res = 1               # number of resources in the system
        self.num_nw = 5                # maximum allowed number of work in the queue

        self.time_horizon = 80         # number of time steps in the graph
        self.max_job_len = 60          # maximum duration of new jobs
        self.res_slot = 120            # maximum number of available resource slots
        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 80         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40          # maximum number of distinct colors in current work graph

    #    self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor
        self.scheduling_policy='BEST_FIT'
        # distribution for new job arrival
    #    self.dist = job_distribution.Dist(self.num_res, self.max_job_size, self.max_job_len)

        # graphical representation
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = int(math.ceil(self.backlog_size / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full
        self.utilization_penalty=-1
        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 8
        self.evaluate_policy_name = "SJF"

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
            (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

