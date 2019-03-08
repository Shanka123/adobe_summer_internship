package org.processing.examples;

import org.processing.protobuf.MetricOuterClass;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.File;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Modified by harvines for reading protobuf files from S3 bucket
 * ReadMetricFromFile class takes in input and output file params,
 * parses the input files into a list of metrics
 * and writes the results to output file
 * Created by sbalakri on 5/28/17.
 * ComputeMetricRegionCount class takes in input and output file params
 * and parses the input files into a list of metrics
 * and computes the count of metrics per region
 * and writes this count to the output.
 */

public class ReadMetricFromFile {
	private static final int HEADER_LENGTH = 2;

	// for files in the standard format, read first two bytes for version, parse using parseDelimitedFrom to obtain metrics
	protected static List<MetricOuterClass.Metric> getDelimitedMetrics(BufferedInputStream inputStream) throws Exception {
		byte[] versionBuffer = new byte[HEADER_LENGTH];
		// version based processing here if number of versions > 1
		inputStream.read(versionBuffer);

		ArrayList<MetricOuterClass.Metric> metricList = new ArrayList<>();
		while (inputStream.available() > 0) {
			MetricOuterClass.Metric metric = MetricOuterClass.Metric.parseDelimitedFrom(inputStream);
			if (metric == null)
				break;
			metricList.add(metric);
		}
		return metricList;
	}
	
	public static void main(String[] args) throws Exception {
		/*xvx
		 * Input params -
		 * 1. input foldername
		 * 2. output filename
		 * Example -
		 * "/home/ubuntu/2018/06/01/" "/home/ubuntu/2018_06_01.csv"
		 */
		try {
			// set input file name
			String inputFolderName = args[0];
			String outputFileName = args[1];
			//br = new BufferedReader(new FileReader(FILENAME));
			int MIB_TO_BYTES_FACTOR = 1048576;
			double BYTES_TO_MIB_FACTOR = 9.53674316e-7;
			int INFLATED_MEM = 32*MIB_TO_BYTES_FACTOR;
			double INFLATED_CPU = 0.1;
			PrintWriter writer = new PrintWriter(new File(outputFileName));
			StringBuilder sb0 = new StringBuilder();
            sb0.append("mesos_task_id");
            sb0.append(',');
            sb0.append("mesos_task_total_cpu_time");
            sb0.append(',');
            sb0.append("mesos_task_cpu_limit");
            sb0.append(',');
            sb0.append("mesos_task_mem_rss");
            sb0.append(',');
            sb0.append("mesos_task_mem_limit");
            sb0.append(',');
            sb0.append("mesos_node_id");
            sb0.append(',');
            sb0.append("mesos_node_timestamp");
            sb0.append(',');
            sb0.append("mesos_node_mem_utilization");
            sb0.append(',');
            sb0.append("mesos_node_cpu_utilization");
            sb0.append('\n');
            writer.write(sb0.toString());
            
            File folder = new File(inputFolderName);
            File[] listOfFiles = folder.listFiles();
            Arrays.sort(listOfFiles);
            //int k = 0;
           
            for (int j = 0; j < listOfFiles.length; j++) {
            	if (!listOfFiles[j].isFile()) continue;
				String inputFileName = listOfFiles[j].getName();
				
				try {
					InputStream is = new FileInputStream(inputFolderName+inputFileName);
					InputStream gis = new GZIPInputStream(is);
					BufferedInputStream bis = new BufferedInputStream(gis);
					List<MetricOuterClass.Metric> metricList = getDelimitedMetrics(bis);
				
					if(j%1000 == 0) System.out.println(j);
					for(int i=0; i<metricList.size(); i++) {
						try {
							String service = metricList.get(i).getMesosTask().getId().split("\\.")[0];
							if(service.matches("\\w+-\\w+-\\w{10}")) {//for user-facing jobs
		            	   StringBuilder sb = new StringBuilder();          	   
		            	   sb.append(String.valueOf(metricList.get(i).getMesosTask().getId()));
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosTask().getCpusUserTimeSecs() + metricList.get(i).getMesosTask().getCpusSystemTimeSecs()));
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosTask().getCpusLimit() - INFLATED_CPU));
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosTask().getMemRssBytes()));
		                		 
		                   
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosTask().getMemLimitBytes()));
		                   sb.append(',');
		                   
		                   sb.append(String.valueOf(metricList.get(i).getMesosNode().getId()));
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosNode().getTimestamp()));
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosNode().getUsedMem() / metricList.get(i).getMesosNode().getTotalMem()));
		                   sb.append(',');
		                   sb.append(String.valueOf(metricList.get(i).getMesosNode().getUsedCpus() / metricList.get(i).getMesosNode().getTotalCpus()));
		                   sb.append('\n');
		                   writer.write(sb.toString());
		              }
						}catch(Exception e) {
							System.out.println(e);
							continue;
						}
					}
				}catch(Exception e) {
					continue;
				}
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
			throw e;
		}
	}
}
