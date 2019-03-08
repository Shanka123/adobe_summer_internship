
var scheduler_chart;
function init_scheduler(){
  scheduler_chart = $("#scheduler").dxBarGauge({
       geometry: {
           startAngle: 90,
           endAngle: 200
       },
       startValue: 0,
       endValue: 100,
       relativeInnerRadius: 0.40,
       barSpacing: 3,
       values: [47.27, 65.32, 84.59, 71.86, 80],
       palette:['#444444', '#444455','#444466','#444477','#444488'],
       label: {
           indent: 10,
           font: {
               size: 15
           },
           format: {
               type: "fixedPoint",
               precision: 1
           },
           customizeText: function (arg) {
               return arg.valueText + " %";
           }
       },
       "export": {
           enabled: false
       },
   }).dxBarGauge("instance");

}

function add_value_util(chart, data){
    chart['dataProvider'].push(data)
    chart.validateData()
}
function reset_util_graph(chart){
    chart.dataProvider = [{"time": -1, 'init':80}]
    chart.validateData() 
}
var machine1_util = AmCharts.makeChart("machine1_util", {
        "hideCredits": true,
        "type": "serial",
        "theme": "light",
        "marginRight":20,
        // "legend": {
        //     "equalWidths": false,
        //     "position": "top",
        //     "valueAlign": "left",
        //     "valueWidth": 100
        // },
        "dataProvider": [],
        "valueAxes": [{
            "stackType": "regular",
            "gridAlpha": 0.07,
            "position": "left",
            "title": "Utilization"
        }],
        "graphs": [
        {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "init",
            "lineColor": "white",
            "valueField": "init"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job A",
            "lineColor": "yellow",
            "valueField": "Job A1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job B",
            "lineColor": "red",
            "valueField": "Job B1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C2"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job D",
            "lineColor": "blue",
            "valueField": "Job D1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job E",
            "lineColor": "#8B008B",
            "valueField": "Job E1"
        }],
        "plotAreaBorderAlpha": 0,
        "marginTop": 4,
        "marginLeft": 0,
        "marginBottom": 0,
        "categoryField": "time",
        // "categoryAxis": {
        //     "startOnAxis": true,
        //     "axisColor": "#000000",
        //     "gridAlpha": 0.07,
        //     "title": "time",
        // },
        "export": {
            "enabled": false
         }
    });
var machine2_util = AmCharts.makeChart("machine2_util", {
        "hideCredits": true,
        "type": "serial",
        "theme": "light",
        "marginRight":20,
        // "legend": {
        //     "equalWidths": false,
        //     "position": "top",
        //     "valueAlign": "left",
        //     "valueWidth": 100
        // },
        "dataProvider": [],
        "valueAxes": [{
            "stackType": "regular",
            "gridAlpha": 0.07,
            "position": "left",
            "title": "Utilization"
        }],
        "graphs": [
        {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "init",
            "lineColor": "white",
            "valueField": "init"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job A",
            "lineColor": "yellow",
            "valueField": "Job A1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job B",
            "lineColor": "red",
            "valueField": "Job B1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job B",
            "lineColor": "red",
            "valueField": "Job B2"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C2"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job D",
            "lineColor": "blue",
            "valueField": "Job D1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job E",
            "lineColor": "#8B008B",
            "valueField": "Job E1"
        }],
        "plotAreaBorderAlpha": 0,
        "marginTop": 4,
        "marginLeft": 0,
        "marginBottom": 0,
        "categoryField": "time",
        // "categoryAxis": {
        //     "startOnAxis": true,
        //     "axisColor": "#000000",
        //     "gridAlpha": 0.07,
        //     "title": "time",
        // },
        "export": {
            "enabled": false
         }
    });var machine3_util = AmCharts.makeChart("machine3_util", {
        "hideCredits": true,
        "type": "serial",
        "theme": "light",
        "marginRight":20,
        // "legend": {
        //     "equalWidths": false,
        //     "position": "top",
        //     "valueAlign": "left",
        //     "valueWidth": 100
        // },
        "dataProvider": [],
        "valueAxes": [{
            "stackType": "regular",
            "gridAlpha": 0.07,
            "position": "left",
            "title": "Utilization"
        }],
        "graphs": [
        {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "init",
            "lineColor": "white",
            "valueField": "init"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "lineColor": "yellow",
            "title": "Job A",
            "valueField": "Job A1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job B",
            "lineColor": "red",
            "valueField": "Job B1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C2"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job D",
            "lineColor": "blue",
            "valueField": "Job D1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job E",
            "lineColor": "#8B008B",
            "valueField": "Job E1"
        }],
        "plotAreaBorderAlpha": 0,
        "marginTop": 4,
        "marginLeft": 0,
        "marginBottom": 0,
        "categoryField": "time",
        // "categoryAxis": {
        //     "startOnAxis": true,
        //     "axisColor": "#000000",
        //     "gridAlpha": 0.07,
        //     "title": "time",
        // },
        "export": {
            "enabled": false
         }
    });var machine4_util = AmCharts.makeChart("machine4_util", {
        "hideCredits": true,
        "type": "serial",
        "theme": "light",
        "marginRight":20,
        // "legend": {
        //     "equalWidths": false,
        //     "position": "top",
        //     "valueAlign": "left",
        //     "valueWidth": 100
        // },
        "dataProvider": [],
        "valueAxes": [{
            "stackType": "regular",
            "gridAlpha": 0.07,
            "position": "left",
            "title": "Utilization"
        }],
        "graphs": [
        {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "init",
            "lineColor": "white",
            "valueField": "init"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "lineColor": "yellow",
            "title": "Job A",
            "valueField": "Job A1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job B",
            "lineColor": "red",
            "valueField": "Job B1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C2"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job D",
            "lineColor": "blue",
            "valueField": "Job D1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job E",
            "lineColor": "#8B008B",
            "valueField": "Job E1"
        }],
        "plotAreaBorderAlpha": 0,
        "marginTop": 4,
        "marginLeft": 0,
        "marginBottom": 0,
        "categoryField": "time",
        // "categoryAxis": {
        //     "startOnAxis": true,
        //     "axisColor": "#000000",
        //     "gridAlpha": 0.07,
        //     "title": "time",
        // },
        "export": {
            "enabled": false
         }
    });var machine5_util = AmCharts.makeChart("machine5_util", {
        "hideCredits": true,
        "type": "serial",
        "theme": "light",
        "marginRight":20,
        // "legend": {
        //     "equalWidths": false,
        //     "position": "top",
        //     "valueAlign": "left",
        //     "valueWidth": 100
        // },
        "dataProvider": [],
        "valueAxes": [{
            "stackType": "regular",
            "gridAlpha": 0.07,
            "position": "left",
            "title": "Utilization"
        }],
        "graphs": [
        {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "init",
            "lineColor": "white",
            "valueField": "init"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job A",
            "lineColor": "yellow",
            "valueField": "Job A1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job B",
            "lineColor": "red",
            "valueField": "Job B1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "Job C",
            "lineColor": "green",
            "valueField": "Job C1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "lineColor": "green",
            "title": "Job C",
            "valueField": "Job C2"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "lineColor": "blue",
            "title": "Job D",
            "valueField": "Job D1"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "lineColor": "#8B008B",
            "title": "Job E",
            "valueField": "Job E1"
        }],
        "plotAreaBorderAlpha": 0,
        "marginTop": 4,
        "marginLeft": 0,
        "marginBottom": 0,
        "categoryField": "time",
        // "categoryAxis": {
        //     "startOnAxis": true,
        //     "axisColor": "#000000",
        //     "gridAlpha": 0.07,
        //     "title": "time",
        // },
        "export": {
            "enabled": false
         }
    });
//cluster
AmCharts.addInitHandler(function(chart) {
 // check if there are graphs with autoColor: true set
 for(var i = 0; i < chart.graphs.length; i++) {
   var graph = chart.graphs[i];
   if (graph.autoColor !== true)
     continue;
   var colorKey = "autoColor-"+i;
   graph.lineColorField = colorKey;
   graph.fillColorsField = colorKey;
   for(var x = 0; x < chart.dataProvider.length; x++) {
     var color = chart.colors[x]
     chart.dataProvider[x][colorKey] = color;
   }
 }
 
}, ["serial"]);

var cluster_bar = AmCharts.makeChart("cluster_bar", {
      "hideCredits" : "true",
      "type": "serial",
      "theme": "light",
      "colors": ["yellow","red","green","blue","#8B008B"],
      "dataProvider": [ {
        "toj": "",
        "prob": 60,
      }, {
        "toj": "",
        "prob": 10
      }, {
        "toj": "",
        "prob": 15
      }, {
        "toj": "",
        "prob": 5
      }, {
        "toj": "",
        "prob": 10
      }],
      "valueAxes": [ {
        "gridColor": "#FFFFFF",
        "gridAlpha": 0.2,
        "dashLength": 0
      } ],
      "startDuration": 1,
      "graphs": [ {
        "fillAlphas": 0.8,
        // "lineAlpha": 0,
        "type": "column",
        "valueField": "prob",
        "autoColor": true
      } ],
      "categoryField": "toj",
      "categoryAxis": {
        "gridPosition": "start",
        "gridAlpha": 0,
        "tickPosition": "start",
        "tickLength": 0,
        "labelRotation": 90,
        "fontSize": 15
      },
      "export": {
        "enabled": false
      }
    });

var task_util = AmCharts.makeChart("task_util", {
        "hideCredits": true,
        "type": "serial",
        "theme": "light",
        "marginRight":20,
        // "legend": {
        //     "equalWidths": false,
        //     "position": "top",
        //     "valueAlign": "left",
        //     "valueWidth": 100
        // },
        "dataProvider": [],
        "valueAxes": [{
            "stackType": "regular",
            "gridAlpha": 0.07,
            "position": "left",
            "title": "Utilization"
        }],
        "graphs": [
        {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "init",
            "lineColor": "white",
            "valueField": "init"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "stkidxs",
            "lineColor": "yellow",
            "valueField": "stkidxs"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "cclibraries",
            "lineColor": "red",
            "valueField": "cclibraries"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "activity",
            "lineColor": "green",
            "valueField": "activity"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "lridxsvc",
            "lineColor": "blue",
            "valueField": "lridxsvc"
        }, {
            "fillAlphas": 0.6,
            "lineAlpha": 0.4,
            "title": "enterorgser",
            "lineColor": "#8B008B",
            "valueField": "enterorgser"
        }],
        "plotAreaBorderAlpha": 0,
        "marginTop": 4,
        "marginLeft": 0,
        "marginBottom": 0,
        "categoryField": "time",
        // "categoryAxis": {
        //     "startOnAxis": true,
        //     "axisColor": "#000000",
        //     "gridAlpha": 0.07,
        //     "title": "time",
        // },
        "export": {
            "enabled": false
         }
    });

function change_probs(chart, probs){
    for (var i = 0; i < probs.length; i++) {
        chart.dataProvider[i].prob = probs[i];
    }
    chart.validateData();
    chart.startEffect = "elastic";
   chart.startDuration = "1";
   chart.animateAgain();
}
