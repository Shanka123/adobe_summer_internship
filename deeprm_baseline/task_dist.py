
import pandas as pd
import numpy as np
from parameters import Parameters

class Task_Dist:
	def __init__(self, task_details_file='~/ethos_time_series.csv'):
	   self.task_details_file = task_details_file
	   self.params = Parameters()
	   self.extract_info()

	def extract_info(self, seed= 40):
	   np.random.seed(seed)
	   self.task_details = pd.read_csv(self.task_details_file)
	   colormap = np.arange(1/float(len(self.task_details['type'].unique().tolist())), 1, 1/float(len(self.task_details['type'].unique().tolist())+1))
	   np.random.shuffle(colormap)
	   self.task_details.insert(loc=1, column='color', value= pd.Series(colormap))
	   #print(self.task_details[['type', 'color']])
			# self.tasks = {}
			# i = 0
			# for t in task_details['type'].unique():
			#     self.tasks[t] = {
			#         'color': colormap[i],
			#         'finish_time': task_details['finish_time'].iloc[i],
			#         'cpu_usage': list(task_details.iloc[i, 2:task_details['finish_time'].iloc[i]])
			#     }
			#     i += 1

	def get_task_details(self, task):
	   task_info = self.task_details[self.task_details['type'] == task]
	  # print(task,task_info)
	   return task_info['color'].values[0], task_info['cpu_limit'].values[0], task_info['finish_time'].values[0]

	def get_cpu_usage(self, task, time):
	   task_info = self.task_details[self.task_details['type'] == task.service]
	   if(task_info['finish_time'].values[0] < time):
		   return -1
	   return task_info[str(time - 1)].values[0]


	def gen_seq_workload(self, seed= 20):
	# np.random.seed(seed)
	# sim_len = 7
	# seq_workload = np.random.choice(
	#     self.task_details['type'].unique().tolist() + ['None'],
	#     sim_len
	# )
		services = list(self.task_details['type'].tolist())
		instances = list(self.task_details['instances'].tolist())
		periods = list(self.task_details['period'].tolist())
		np.random.seed(seed)
		print(services, instances, periods)
		workloads = []
		for i in range(self.params.num_ex + self.params.num_test_ex):
		   seq = []
		   max_len = 11
		   total_tasks = 0
		   for i in range(0,9):
			   tmp = []
			   for j in range(len((services))):
				   if(i%periods[j] == 0):
					   if instances[j] == 1:
						   tmp += [services[j]]*instances[j]
					   else:
						   tmp += [services[j]]*np.random.randint(1, instances[j]+1)
			   np.random.shuffle(tmp)
		   #     if len(tmp):
		   #         tmp += (np.random.randint(0, len(tmp))) * [None]
			   total_tasks += len(tmp)
			   if total_tasks > max_len:
				   tmp = tmp[:len(tmp) - (total_tasks-max_len)]
				   seq.append(tmp.copy())
				   break
			   seq.append(tmp.copy())

		   workloads.append(seq)
    #    workloads=[[]]
		return workloads

	 # def gen_seq_workload(self, seed= 20):
  #            # np.random.seed(seed)
  #            # sim_len = 7
  #            # seq_workload = np.random.choice(
  #            #     self.task_details['type'].unique().tolist() + ['None'],
  #            #     sim_len
  #            # )
  #           a,b,c,d = self.task_details['type'].unique().tolist()
  #           np.random.seed(seed)
  #       #    arr=[a ,b ,c ,d ,e, b ,a ,d ,c, e ,a ,b, c, d ,e ,a, c, d, e ,b ]
  #           # nones=np.random.choice(30, 10,replace=False).tolist()
  #           # for i in sorted(nones):
  #           #     arr.insert(i,None)
  #           # arr = [
  #           #         [a, d, b, e, c],
  #           #         [b, d, a, c, e],
  #           #         [a, b, d, e, c],
  #           #         [d, b, a, e, c]
  #           #       ]
  #           arr=[]
  #           from itertools import permutations as perm
  #           #for x in list(set(perm([a,b,c,d,e]))):
  #           #    for y in list(set(perm([a,b,'None', 'None']))):
  #           #        for z in  list(set(perm([e,'None',c,d]))):
  #           #            arr.append(list(x)+list(y)+list(z))
  #           #arr = list(set(list(perm([a,a,b,b,d]))))
  #           #arr=[[a,d,b,d,b]]
  #           #arr=[[b,a,d,b,d]]
  #           #arr=[[b,b,d,d,a]]
  #           #arr=[[a]+[None]*50+[b,b]+[None]*200+[c]+[None]*50+[d,d,d]+[None]*10+[a]+[None]*50+[b,b]+[None]*200+[c]+[None]*50+[d,d,d]]
  #           #arr=[[a,b,b,c,d,a,b,a,b,b,c,d,d,c],[a,b,c,b,a,d,b,a,b,d,c,b,c,d],[b,a,b,c,d,a,b,b,a,d,b,c,c,d],[b,b,a,d,c,c,a,d,b,c,a,d,a,b]]
  #           #arr=[[c, d, a, a, d, e, b, d]]
  #           #arr = [[a, b, c, b, d, a, b, b, a, c, d, d, c, b]]
  #           arr=[[a,c,c,d,d,c,b,d,d]]
  #           return arr


	# def gen_seq_workload(self, seed= 20):
	# 	 # np.random.seed(seed)
	# 	 # sim_len = 7
	# 	 # seq_workload = np.random.choice(
	# 	 #     self.task_details['type'].unique().tolist() + ['None'],
	# 	 #     sim_len
	# 	 # )
	# 	a,b,c,d ,e= self.task_details['type'].unique().tolist()
	# 	np.random.seed(seed)
	# #    arr=[a ,b ,c ,d ,e, b ,a ,d ,c, e ,a ,b, c, d ,e ,a, c, d, e ,b ]
	# 	# nones=np.random.choice(30, 10,replace=False).tolist()
	# 	# for i in sorted(nones):
	# 	#     arr.insert(i,None)
	# 	# arr = [
	# 	#         [a, d, b, e, c],
	# 	#         [b, d, a, c, e],
	# 	#         [a, b, d, e, c],
	# 	#         [d, b, a, e, c]
	# 	#       ]
	# 	arr=[]
	# 	from itertools import permutations as perm
	# 	#for x in list(set(perm([a,b,c,d,e]))):
	# 	#    for y in list(set(perm([a,b,'None', 'None']))):
	# 	#        for z in  list(set(perm([e,'None',c,d]))):
	# 	#            arr.append(list(x)+list(y)+list(z))
	# 	#arr = list(set(list(perm([a,a,b,b,d]))))
	# 	#arr=[[a,d,b,d,b]]
	# 	#arr=[[b,a,d,b,d]]
	# 	#arr=[[b,b,d,d,a]]
	# 	#arr=[[a]+[None]*50+[b,b]+[None]*200+[c]+[None]*50+[d,d,d]+[None]*10+[a]+[None]*50+[b,b]+[None]*200+[c]+[None]*50+[d,d,d]]
	# 	#arr=[[a,b,b,c,d,a,b,a,b,b,c,d,d,c],[a,b,c,b,a,d,b,a,b,d,c,b,c,d],[b,a,b,c,d,a,b,b,a,d,b,c,c,d],[b,b,a,d,c,c,a,d,b,c,a,d,a,b]]
	# 	#arr=[[c, d, a, a, d, e, b, d]]
	# 	#arr = [[a, b, c, b, d, a, b, b, a, c, d, d, c, b]]
	# 	arr=[[[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [], [], [],[d],[e]]]
	# # 	# arr=[[[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[c],[],[c],[], [], [], [], [], [], [], [], [],[d],[e]]
	# # 	# ,[[a],[],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[c],[],[c],[], [], [], [], [], [], [], [], [],[d],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [], [], [],[d],[],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[],[],[c],[c],[], [], [], [], [], [], [], [], [],[],[],[d],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[],[],[c],[c],[], [], [], [], [], [], [], [], [],[d],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [], [], [],[],[],[d],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [], [], [],[d],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [],[d],[e]],
	# # 	# [[a],[b],[], [], [], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [],[d],[e]],
	# # 	# [[a],[],[b],[], [], [], [], [], [], [], [], [], [], [], [], [],[c],[c],[], [], [], [], [], [], [], [], [],[],[d],[],[e]]]
	# # 	arr=[[[a],[b],[], [], [], [], [], [], [], [], [], [], [], [],[],[],[],[c],[c],[], [], [], [], [], [], [],[d],[],[e]]]
	# 	return arr
