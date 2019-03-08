# Demo
For each policy create a json in current folder, the code to write the file is present in `policy_eval.py`
Format of json
 ```
 env = [
 		[
     //machine0
 			[ //tasks running at this time with their utilization
 				{"name":"scid","util": [0,0,0,0,0,0,0,0,0,2]},
        {"name":"acti","util": [0,0,0,0,0,0,0,0,2, 3]}
      ],
      //machine1
      [
 			  {"name":"tps","util": [0,0,0,0,0,0,0,0,0,5]},
        {"name":"lridxsvc","util": [0,0,0,0,0,0,0,0,0,4]}
      ],
       ['lridxsvc'], //next incoming task
       [20,30,10,40,10], //probability of each action
       "0" //next action
 		],//next time_stamp
 		[
 			[
 				{"name":"scid","util": [0,0,0,0,0,0,0,0,2, 3]},
        			{"name":"acti","util": [0,0,0,0,0,0,0,2, 3 ,5]}
        		],
        		[
 				{"name":"tps","util": [0,0,0,0,0,0,0,0,5, 1]},
        			{"name":"lridxsvc","util": [0,0,0,0,0,0,0,0,4, 2]}
        		],
        		['scid'],
        		[50,10,10,30,20],
        		"1"
 		],next time_stamp
 		[
 			[
 				{"name":"scid","util": [0,0,0,0,0,0,0,2, 3, 4]},
        			{"name":"acti","util": [0,0,0,0,0,0,2, 3 ,5, 6]}
        		],
        		[
 				{"name":"tps","util": [0,0,0,0,0,0,0,5, 1, 4]},
        			{"name":"lridxsvc","util": [0,0,0,0,0,0,0,4, 2 ,3]}
        		],
        		[],
        		"2"
 		]
 	]
  ```
`main.js` constains the part related how each task is classified and scheduled and put into machine.
`chart.js` contains information about each graph, classifier and scheduler.
Code Flow: 
Current code support 5 machines, 5 type of tasks.
To run the demo, run `python2 -m SimpleHTTPServer PORT_NUMBER`
