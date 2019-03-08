from env import Env
from machine import Machine

START_TIME = 0
TIME_STEP = 1
N_MACHINES = 5
MACHINE_RESOURCE = 50
SIM_LEN = 10

def show_state(environ, CUR_SIM_TIME):
	print('WAITING TASK : ', [t.service for t in environ.waiting_tasks])
	i = 0
	for m in environ.machines:
		print('M'+str(i)+' : ',m.cpus_left , [t.service for t in m.running_tasks])
		i+=1
	print('-'*20)

def start_simulator():
	#set-up environ
	environ = Env(START_TIME, TIME_STEP)
	for i in range(N_MACHINES):
		environ.machines.append(Machine(i, MACHINE_RESOURCE))

	environ.generate_workload()
	environ.step(0)
	environ.step(5)
	print(environ.observe())
	environ.generate_workload()
	environ.step(5)
	print(environ.observe())
	print(environ.machines[0].running_tasks[0].cpu_util)
	#CUR_SIM_TIME = 0
		
	#while(CUR_SIM_TIME < SIM_LEN):
		#first update changes during recentl period
		#################################
		# a = []
		# for m in environ.machines:
		#     a += [t.service for t in m.running_tasks]
		##################################

	 #   environ.update()

		#################################
		# a2 = []
		# for m in environ.machines:
		#     a2 += [t.service for t in m.running_tasks]
		# print('Finished Tasks : ', a2)
		##################################
		#Now generate workload
	  #  environ.generate_workload()
		#################################
		# a3 = []
		# for m in environ.machines:
		#     a3 += [t.service for t in m.running_tasks]
		# print('New Tasks : ', a3[len(a2)-1:])
		##################################
	   # print('TIME : ', CUR_SIM_TIME)
		#print('Before Scheduling : ')
		#show_state(environ, CUR_SIM_TIME)
		#Scheduing
		#environ.schedule()
		#print('After Scheduling : ')
		#show_state(environ, CUR_SIM_TIME)
		#CUR_SIM_TIME += 1
	
start_simulator()
