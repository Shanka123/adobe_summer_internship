function bar_next_step(step_number) {
	div_step = document.getElementById("st" + step_number.toString());
	div_step.setAttribute("class", "col-xs-3 bs-wizard-step complete");
}

function bar_start_step(step_number) {
	div_step = document.getElementById("st" + step_number.toString());
	div_step.setAttribute("class", "col-xs-3 bs-wizard-step active");
}

function reset(){
	for(step_number=1; step_number<=4; step_number++){
		div_step = document.getElementById("st" + step_number.toString());
		div_step.setAttribute("class", "col-xs-3 bs-wizard-step disabled");
		// div_step = document.getElementById("st" + step_number.toString());
		// div_step.setAttribute("class", "col-xs-3 bs-wizard-step disabled");
	}
}
var envs = {};
var env;
var policy = 0;
//load the json for different policies
function load() {
		loadJSON("/utilization.json", function(response) {
			envs['utilization'] = JSON.parse(response);
			loadJSON("/experience.json", function(response) {
				envs['experience'] = JSON.parse(response);
				start();
			});
		});
}
load();
function start(){
	$(document).ready(function(){
		//scheduler circular chart
		init_scheduler();

		var num_machines = 5
		var machine_width = 180, machine_height = 100;
		var task_height = 20;
		var win_len = 5;
		time = 0; //current time
		var step = 1;
		var next_step;
		var timeout;
		var finish;

		function intialize(){
			//empty all the graph and machines
			reset_util_graph(machine1_util);
			reset_util_graph(machine2_util);
			reset_util_graph(machine3_util);
			reset_util_graph(machine4_util);

			$('#action_value').hide();
			$('#machines').empty();
			//select the json to be used based on policy
			if(document.getElementById("policy").selectedIndex == 0){
				env = envs['utilization'];
				policy = 0;
				num_machines = 5;
				reset_util_graph(machine5_util);
				$('#machine5_util_show').show();
				$('#5_machine_numbers').show();
				$('#4_machine_numbers').hide();
				scheduler_chart.option("values", [20,20,20,20,20]);
			}else{
				env = envs['experience'];
				policy = 1;
				num_machines = 4;
				machine5_util.dataProvider = [];machine5_util.validateData();
				$('#machine5_util_show').hide();
				$('#5_machine_numbers').hide();
				$('#4_machine_numbers').show();
				scheduler_chart.option("values", [25,25,25,25]);
			}
			finish = env.length-1;
			time = 0;
			document.getElementById('clock').innerText = '0000';
			var machines = document.getElementById('machines');
			//create the machine element
			for(mac_id= 0; mac_id< num_machines; mac_id++){
				var machine = document.createElement('div')
				machine.id = "machine"+mac_id;
				machine.setAttribute('class', 'machine');
				machine.style.width = machine_height + 'px';
				machine.style.height = machine_width + 'px';
				machine.style.left = '1250px';
				machine.style.top = (40 + mac_id*(machine_width + 10)) + 'px';
				machines.appendChild(machine);
				var machine_name = document.createElement('div')
				machine_name.id = "machine_name"+mac_id;
				machine_name.setAttribute('class', 'machine_name');
				machine_name.style.height = machine_width + 'px';
				machine_name.style.left = '1285px';
				machine_name.style.top = (70 + mac_id*(machine_width + 10)) + 'px';
				machine_name.innerText = mac_id+1;
				machine_name.style.opacity = 0.5;
				machines.appendChild(machine_name);
			}
		}

		function job_comes(){
			$('#action_value').hide();
			$('#task_util').hide();
			//if there is a incoming task at this time, create a element coreesponding to that
			if (env[time][num_machines].length){
				bar_start_step(1);
				bar_next_step(1);
				var to_be_scheduled_task = document.createElement('div');
				to_be_scheduled_task.id = 'to_be_scheduled_task';
				to_be_scheduled_task.setAttribute('class', 'new_task');
				to_be_scheduled_task.style.top = '140px';
				to_be_scheduled_task.style.left = '575px'
				to_be_scheduled_task.style.height = '40px';
				to_be_scheduled_task.style.width = '40px';
				to_be_scheduled_task.style.opacity = 0.6;
				document.body.appendChild(to_be_scheduled_task);
				$('#to_be_scheduled_task').css('border', '3px black solid');
				$("#to_be_scheduled_task").show();
				//move the task to classifier within 500ms
				$("#to_be_scheduled_task").animate({top: '400px'}, 500);
				setTimeout(cluster_step, 500);
				//
			}
		}

		function cluster_step(){
			var type = env[time][num_machines][0];
			var clus = [[50, 20, 5, 10, 15],[5,70,10,10,5], [5,15,50,15,25], [15,20,10,50,5], [15,25,15,10,45]];
			bar_start_step(2);
			//change the probabilities according the task class
			if(type=='stkidxs'){
 				change_probs(cluster_bar, clus[0]);
			}else if( type == 'cclibraries'){
				change_probs(cluster_bar, clus[1]);
			}else if(type=='activity'){
				change_probs(cluster_bar, clus[2]);
			}else if(type=='lridxsvc'){
				change_probs(cluster_bar, clus[3]);
			}else{
				change_probs(cluster_bar, clus[4]);
			}
			//wait for 1000ms and then move the task to scheduler within 500ms
			setTimeout(function(){
				bar_next_step(2);
				$('#to_be_scheduled_task').attr("class", type);
				$("#to_be_scheduled_task").animate({left: '980px'}, 500);
				setTimeout(scheduling_decision_step, 500);
				// $('#task_util').show();
			}, 1000);

			//
			// reset_util_graph(task_util);
	  //   	data = {
		 //    	'stkidxs': [9,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,8,9,9,9,9,9,8,9,9,9,9,9,9,9,8,9,9,9,9,9,9,8,9,9,9,8,8,9,9,9,6,9,9,9],
			// 	'lridxsvc': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,6,8,11,11,11,8,6,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,6,8,11,11,11,8,6,3,2],
			// 	'cclibraries': [1,1,1,2,4,5,5,4,2,1,1,1,2,4,5,5,4,2,1,1],
			// 	'activity': [4,6,8,7,5,7,8,5,2,1],
			// 	'enterorgser': [7,4,1,6,12,8,1,1,4,4]
			// }
			// console.log(data[type]);
		 //    for(i=0; i<data[type].length; i++){
		 //    	data2={'time':i}
		 //    	data2[type] = (data[type][i]*12*100)/180;
		 //    	add_value_util(task_util, data2);
		 //    }
		}

		
		function scheduling_decision_step(){
			bar_start_step(3);
			var action = parseInt(env[time][num_machines+2]);
			//change scheduler probabilities according to action
			scheduler_chart.option("values", env[time][num_machines+1]);
			setTimeout(function(){
				bar_next_step(3);
				//move the task to corresponding machine
				$("#to_be_scheduled_task").animate({left: '1250px', top: (20 + action*(machine_width + 10))+'px'}, 500); //adjust left with new UI
				setTimeout(scheduled, 500);
				setTimeout(function(){$('#action_value').html(action+1);$('#action_value').show();}, 100);
			}, 1500);
		}

		function scheduled(){
			var action = parseInt(env[time][num_machines+2]);
			bar_start_step(4);
			step();
			function step(){
				time++;
				$('.machine').empty();
				$("#to_be_scheduled_task").remove();
				if(time>finish)
					return
				var machines = document.getElementById('machines');
				//put the task and update each machine utilization
				for(mac_id= 0; mac_id< num_machines; mac_id++){
					var used_perc = 0;
					var machine = document.getElementById('machine' + mac_id);
					consumed = Array(win_len).fill(0);
					data = {"time": time-1};
					for(task_id= 0; task_id< env[time][mac_id].length; task_id++){
						for(t= 0; t<1; t++){
							var task = document.createElement('div');
							task.id = t+"_"+"task_"+mac_id+"_"+task_id;
							task.setAttribute('class', env[time][mac_id][task_id]['name']);
							task.style.opacity = 0.75;
							task.style.top = (machine_width-6-consumed[win_len-1-t]-env[time][mac_id][task_id]['util'][win_len-1-t]*12) + 'px';
							task.style.left = (task_height*t) + 'px';
							task.style.height = env[time][mac_id][task_id]['util'][win_len-1-t]*12 + 'px';
							consumed[win_len-1-t] += env[time][mac_id][task_id]['util'][win_len-1-t]*12;
							machine.appendChild(task);

							var mapping = {
								'stkidxs': 'Job A',
								'cclibraries': 'Job B',
								'activity': 'Job C',
								'lridxsvc': 'Job D',
								'enterorgser': 'Job E'
							}
							for(i=0; i<5; i++){
								if (!((mapping[env[time][mac_id][task_id]['name']]+(i+1)) in data)){
									used_perc += (env[time][mac_id][task_id]['util'][win_len-1-t]*12*100)/180;
									data[mapping[env[time][mac_id][task_id]['name']] + (i+1)] = (env[time][mac_id][task_id]['util'][win_len-1-t]*12*100)/180;
									break;
								}
							}
					        
						}
					}
					//update the utilization graph for each machine
					used_perc = parseInt(used_perc);
					if(mac_id==0){
						$('#machine1_util_show').html(used_perc+'%');
						add_value_util(machine1_util, data)
					}else if(mac_id==1){
						$('#machine2_util_show').html(used_perc+'%');
						add_value_util(machine2_util, data)
					}else if(mac_id==2){
						$('#machine3_util_show').html(used_perc+'%');
						add_value_util(machine3_util, data)
					}else if(mac_id==3){
						$('#machine4_util_show').html(used_perc+'%');
						add_value_util(machine4_util, data)
					}else if(mac_id==4){
						$('#machine5_util_show').html(used_perc+'%');
						add_value_util(machine5_util, data)
					}
					
				}
				//reset();

				var timeString;
				if(parseInt(time/10))
					timeString = '00' + time.toString();
				else if (parseInt(time/100))
					timeString = '0' + time.toString();
				else
					timeString = '000' + time.toString();
				if(env[time][num_machines].length){
					setTimeout(reset, 1000);
					//timeout = setTimeout(job_comes, 1500);
					next_step = 1;
					document.getElementById('clock').innerText = timeString;
				}else{
					setTimeout(reset, 100);
					//timeout = setTimeout(step, 100);
					next_step = 4
					document.getElementById('clock').innerText = timeString;
				}						
			}
		}

		intialize();
		$("#scheduler").hover(
			function(){
		  //   reset_util_graph(task_util);

		  //   data = {
		  //   	'stkidxs': [9,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,8,9,9,9,9,9,8,9,9,9,9,9,9,9,8,9,9,9,9,9,9,8,9,9,9,8,8,9,9,9,6,9,9,9],
				// 'lridxsvc': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,6,8,11,11,11,8,6,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,3,6,8,11,11,11,8,6,3,2],
				// 'cclibraries': [1,1,1,2,4,5,5,4,2,1,1,1,2,4,5,5,4,2,1,1],
				// 'activity': [4,6,8,7,5,7,8,5,2,1],
				// 'enterorgser': [7,4,1,6,12,8,1,1,4,4]
				// }
				// var c = data[$('#to_be_scheduled_task').attr('class')];
				// console.log(c[0]);
		  //   for(i=0; i<c.length; i++){
		  //   	data={'time':i, 'Job A1': c[i]};
		  //   	add_value_util(task_util, data);
		  //   }
		    // $('#task_util').show();
			},
			function(){
				// $('#task_util').hide();
			}
		);
		$("#policy").change(function(){
			intialize();
		});
		document.getElementById("start").addEventListener("click", function(){
			intialize();
			job_comes();
		});
		document.getElementById("stop").addEventListener("click", function(){
			clearTimeout(timeout);
		});
		$(document).keydown(function(){
			if(next_step == 1){
				next_step = 0;
				job_comes();
			}else if(next_step == 4){
				next_step = 0;
				scheduled();
			}
		});

	});
}
