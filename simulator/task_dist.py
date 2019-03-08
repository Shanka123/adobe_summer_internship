import pandas as pd
import numpy as np
from params import Params 
seeds=np.arange(2,20000,10)
import math
import random
import pickle
class Task_Dist:
	def __init__(self):
		self.params = Params()
		self.task_details_file = self.params.task_details_file
		self.extract_info()

	def extract_info(self, seed= 40):
		np.random.seed(seed)
		self.task_details = pd.read_csv(self.task_details_file)
		colormap = np.arange(1/float(len(self.task_details['type'].unique().tolist())), 1, 1/float(len(self.task_details['type'].unique().tolist())+1))
		np.random.shuffle(colormap)
		colormap2=[]
		for c in colormap:
			colormap2.append(c)
			colormap2.append(c)
		print('Colormap',np.unique(np.array(colormap2)).shape)
		self.task_details.insert(loc=1, column='color', value= pd.Series(colormap2))

	def get_task_details(self, task):
		task_info = self.task_details[self.task_details['type'] == task]
		return task_info['color'].values[0], task_info['cpu_limit'].values[0],task_info['mem_limit'].values[0],task_info['finish_time'].values[0]

	#return task cpu_usage at given timestamp
	def get_cpu_usage(self, task, time):
		task_info = self.task_details[self.task_details['type'] == task.service]
		if(task_info['finish_time'].values[0] < time):
			return -1
		return task_info[str(time - 1)].values[0]

	def get_mem_usage(self, task, time):
		task_info = self.task_details[self.task_details['type'] == task.service]
		if(task_info['finish_time'].values[0] < time):
			return -1
		return task_info[str(time - 1)].values[1]

	#generate workload sequences based on their periodcity and number of instance coming at one time
	# def gen_seq_workload(self, seed= 20):
	# 	services = list(self.task_details['type'].tolist())
	# 	instances = list(self.task_details['instances'].tolist())
	# 	periods = list(self.task_details['period'].tolist())
	# 	np.random.seed(seed)
	# 	workloads = []
	# 	k = 0
	# 	#total num of workloads sequences = num_ex + num_test_ex
	# 	while(k < (self.params.num_ex + self.params.num_test_ex)):
	# 		seq = []
	# 		max_len = 17 #max number tasks generated in a sequence 
	# 		total_tasks = 0
	# 		for time_step in range(0,20): # max no of timesteps
	# 			tmp = []
	# 			for j in range(len((services))):
	# 				# find which tasks come at this time based on the period
	# 				if(time_step%periods[j] == 0):
	# 					# multiply it by the number instances based on task
	# 					if instances[j] == 1:
	# 						tmp += [services[j]]*instances[j]
	# 					else:
	# 						tmp += [services[j]]*np.random.randint(1, instances[j]+1)
	# 			# tmp contains all the tasks that comes at this timestep, shuffle the array randomly for this
	# 			np.random.shuffle(tmp)
	# 		#     if len(tmp):
	# 		#         tmp += (np.random.randint(0, len(tmp))) * [None]
	# 			total_tasks += len(tmp)
	# 			if total_tasks > max_len:
	# 				tmp = tmp[:len(tmp) - (total_tasks-max_len)]
	# 				seq.append(tmp.copy())
	# 				break
	# 			seq.append(tmp.copy())
	# 		# to prevent repeated sequences
	# 		if seq not in workloads:
	# 			workloads.append(seq)
	# 			k += 1

	# 	return workloads  


	def create_subsequences(self):
		subsequences=[]
		l=['s_1','s_2','s_3','s_4']
		u=['l_1','l_2','l_3','l_4']
		applications=l+u
	#	random.seed(20)
	#	random.shuffle(applications)
	#	probs=[9/100]*10 +[1/100] *10
		#probs=[0.05]*20
		import numpy as np
		np.random.seed(20)
		for i in range(100):
		#    subsequences.append(list(map(lambda _: random.choice(applications,p=probs), range(4))))
			
			subsequences.append(list(np.random.choice(applications,4)))
			

		final_sequences=[]
		for c in subsequences:
			no_u=0
			no_l=0
			for a in c:
				if a in l:
					no_l+=1
				elif a in u:
					no_u+=1
			if no_u ==2 and no_l ==2:
				final_sequences.append(c)
			if len(final_sequences)==20:
				break
		return final_sequences

	def create_train_sequence(self,no_tasks):
		x=[]
		final_sequences=self.create_subsequences()
		no_to_pick=math.ceil(no_tasks/4.)
	#    print(no_to_pick)
		for i in range(1000):
			seq=[]
			np.random.seed(seeds[i])
			l=np.random.randint(0,20,no_to_pick)
			for j in list(l):
				seq+=final_sequences[j]
			seq=seq[0:no_tasks]
		#    print(i,seq)
			x.append(seq)
		return x  
	def create_test_sequence(self,no_tasks):
		x=[]
		final_sequences=self.create_subsequences()
		no_to_pick=math.ceil(no_tasks/4.)
		#print(no_to_pick)
		for i in range(1000):
			seq=[]
			np.random.seed(seeds[i+1000])
			l=np.random.randint(0,20,no_to_pick)
			for j in list(l):
				seq+=final_sequences[j]
			seq=seq[0:no_tasks]
			x.append(seq)
		return x
		#    print(i,seq)
			  
	def gen_seq_workload(self):
		train=self.create_train_sequence(10)
		test=self.create_test_sequence(10)
		cpu=['l_2','l_3','s_2','s_3']
		s=train+test
		ex_indices=[]
		for i in range(2000):
			m=0
			n=0
			for c in s[i]:
				if c in cpu:
					m+=1
				else:
					n+=1
		#    print(i,max(m,n))
			if max(m,n) ==5:
				ex_indices.append(i)
		train_seq=[]
		test_seq=[]
		for idx,c in enumerate(ex_indices[0:self.params.num_ex]):

			train_seq.append(s[c])
		for idx,c in enumerate(ex_indices[self.params.num_ex:self.params.num_ex+self.params.num_test_ex]):

			test_seq.append(s[c])
			
	
		s1= train_seq+test_seq
		print(len(s1))
		np.random.seed(20)
		arrival=[]
		for c in range(20):
	
			arrival.append(np.random.poisson(2))
		workloads=[]
		for i in range(self.params.num_ex+self.params.num_test_ex):
			x=s1[i]
			seq=[]
			k=0
			no_tasks=0
			for j in arrival:
				seq.append(x[k:k+j])
				no_tasks+=len(x[k:k+j])
				k=k+j
				if no_tasks>=10:
					break
			workloads.append(seq)
		workloads=[]
		# with open('/home/dell/threecomb_random_107_workloads.txt', "rb") as f:
		# 	full_workloads = pickle.load(f)
		# with open('/home/dell/workloads8_9_threecombo_68_random_indices.txt', "rb") as f:
		# 	indices = pickle.load(f)
		# # indices=[2,6,8,10,12,15,16,17,22]
		# for c in indices:
		# 	workloads.append(full_workloads[c])
		with open('/home/dell/threecomb_random_workloads.txt', "rb") as f:
			full_workloads = pickle.load(f)
		with open('/home/dell/workloads8_9_threecombo_random_indices.txt', "rb") as f:
			indices = pickle.load(f)
		# indices=[2,6,8,10,12,15,16,17,22]
		for c in indices:
			workloads.append(full_workloads[c])
		# with open('/home/dell/workloads9_indices.txt', "rb") as f:
		# 	indices = pickle.load(f)
		# for c in indices:
		# 	workloads.append(full_workloads[c])

		# workloads=[[['l_4', 's_3', 'l_3'], ['s_4', 'l_4', 's_2'], ['l_4', 's_2'], ['l_1', 's_1']],
		# [['l_1', 's_1', 's_2'], ['l_1', 's_4', 'l_1'], ['l_4', 's_2'], ['l_3', 's_3']],
		# [['l_1', 's_1', 'l_3'], ['s_3', 's_4', 'l_1'], ['l_4', 's_3'], ['s_2', 'l_1']],
		# [['l_4', 's_3', 'l_4'], ['s_2', 'l_3', 's_3'], ['s_4', 'l_1'], ['s_1', 'l_4']],
		# [['s_4', 'l_1', 's_2'], ['l_1', 's_1', 'l_4'], ['s_3', 'l_1'], ['l_3', 's_3']],
		# [['l_4', 's_1', 'l_1'], ['s_2', 'l_4', 's_2'], ['l_3', 's_3'], ['s_4', 'l_1']],
		# [['l_4', 's_3', 'l_4'], ['s_2', 's_1', 'l_3'], ['l_4', 's_3'], ['s_4', 'l_1']],
		# [['l_4', 's_2', 'l_1'], ['s_1', 'l_4', 's_3'], ['s_4', 'l_3'], ['l_1', 's_2']],
		# [['l_3', 's_4', 's_2'], ['l_1', 's_1', 'l_4'], ['s_3', 'l_1'], ['s_2', 'l_1']]]

		# workloads=[[['l_1', 's_3', 'l_2'], ['s_2', 's_4'], ['s_1', 's_2'], ['s_4', 'l_4']],
		# [['l_1', 'l_2','s_3'], ['s_2', 's_4'], ['s_2', 's_1'], ['s_4', 'l_4']]]
		#workloads=[[['l_3', 'l_2', 's_3'], ['s_3', 's_2'], ['s_4', 's_1'], ['s_1', 'l_1']]]
		#workloads=[[['l_1','s_3','l_2'],['s_2','s_4'],['l_4','s_2'],['s_3','s_4','l_4']]]
		#workloads=[[['l_3', 'l_3', 's_2'], ['l_1', 's_4'], ['s_4', 's_2'], ['s_4', 's_1']]]
		# workloads=[[['l_1', 's_3', 'l_2'], ['s_2', 's_4'], ['s_1', 's_2'], ['s_4', 'l_4']],
  # [['l_3', 's_1', 'l_2'], ['s_4', 's_3'], ['s_2', 's_3'], ['s_1', 'l_4']],
  # [['l_3', 'l_2', 's_3'], ['s_1', 's_2'], ['s_3', 's_1'], ['s_4', 'l_1']],
  # [['s_1', 'l_1', 'l_3'], ['s_2', 's_4'], ['s_3', 's_4'], ['l_1', 's_2']]]
 # [['l_3', 's_2', 's_1'], ['l_1', 's_3'], ['l_1', 's_3'], ['s_2', 's_3']],
 # [['s_2', 'l_4', 's_3'], ['s_3', 's_4'], ['s_4', 'l_1'], ['s_3', 'l_4']],
 # [['s_4', 's_2', 's_2'], ['l_4', 's_3'], ['l_1', 's_1'], ['l_1', 's_4']],
 # [['s_1', 's_3', 's_4'], ['s_3', 'l_4'], ['s_1', 'l_4'], ['s_3', 'l_2']],
 # [['s_3', 's_3', 's_3'], ['l_4', 's_4'], ['s_1', 'l_1'], ['s_3', 'l_2']]]
		return workloads
