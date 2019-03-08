# schedule.ai
First, specify the file that contains the task details in the csv format with columns 
```
type : task type/class/name
cpu_limit: max cpu requirement
finish_time: time to complete
period: period of arrival
instances: no of instances come at one time
0: cpu utilization of the task at t=0
1: cpu utilization of the task at t=0
.
.
.
finish_line: cpu utilization of the task at t=finsih_time
```
Based on above data, `gen_seq_workload` method in `task_dist.py` generate a sequence of tasks. Total number of such sequences will be `num_ex + num_test_ex` to be set in `params.py`.

To install the requirments, run `bash requirements.txt`

To run the code, run `python3 models/policy_eval.py`

#### Code workflow
1. [Workload Generation](https://git.corp.adobe.com/AdobeResearchIndia/ai_systems_internship/blob/master/simulator/task_dist.py#L30)

    A workload seq is generated based on arrival characteristics of the tasks defined in the above csv like period, instances.
    Multiple distinct such sequences are generated initially before start of training
    
    **Algorithm**
    ```
    workload = empty sequence
    max_len = MAX_SEQ_LEN
    for time=0 to T_MAX:
      for each task_type:
        if (time % period(task_type)) == 0:
	        cur_workload = task_type * random(1, num_instances(task_type))
	        shuffle(cur_workload)
	        workload = workload + cur_workload
        if len(workload) > max_len:
	        return workload[:max_len]
    
    functions defnition:
    period(task_type) : return period of given task type
    num_instances(task_type) : return how many instances of the given task_type come at a time 
    shuffle(sequence): shuffles the sequence

    ```
    Each seq is a list of list which contains the tasks coming at that time. E.g
    
    `[ [task_A,task_B,task_C], [], [task_C], [task_B, task_A] [task_C] ]`
    
    At t=0: task_A,task_B,task_C arrives
    
    At t=1: no task comes
    
    At t=2: task_C comes
    
    At t=3: task_B,task_A comes
2. [Environment](https://git.corp.adobe.com/AdobeResearchIndia/ai_systems_internship/blob/master/simulator/env.py) Creation
    
    A workload sequence is assigned to the environment
3. Simulation
    ```
    Time starts at t=0.
    while(t<max_simulation_len)
      Workload is genearated i.e based on time tasks are put on waiting queue
      Agent decides to put the first task in the waiting queue
      Environment is updated by putting task in the machine(agent action) if possible and utilization of machines are updated
    ```

#### Heuristic Agents
There are 6 heuristic policies avilable:
1. Best Fit Peak
2. Worst Fit Peak
3. Tetris Packer
4. Random
5. Best fit current
6. Worst fit current
To run any of above, change the [policy](https://git.corp.adobe.com/AdobeResearchIndia/ai_systems_internship/blob/master/params.py#L28) in param file and run
    
`python3 models/heuristic_agents.py`
