import os, sys, rospy, time
import numpy as np
from scipy.spatial.distance import cdist
os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/time_of_flight")
sys.path.append(os.getcwd())
import SCR_TOF_client as tof
os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/lights")	
sys.path.append(os.getcwd())
import SCR_OctaLight_client as lights_func 										# lights ros functions
os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/color_sensors")
sys.path.append(os.getcwd())
import SCR_COS_client as csensors_func											# color sensors ros functions

os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/tk/sensor_output")		# save sensor data location


fid1 = "_3Dto2D.txt";		# extension for saving files : 


# get location of lights
lightsidx = np.asarray( lights_func.get_lights() );

numtrials = 5000;
tof_data = [];
cct_all_vals = [];
csensor_all_vals =[]; # 57 sensors, 6 outputs
light_response_all_vals = []; # 10 lights, 8 outputs
time_all_vals =[];
set_color_channels_all_vals = []; 
timestr = time.strftime("%Y%m%d-%H%M%S");



for i in range( 0, numtrials ):
	# set output of 
	# if i == 0:
	if i == 0:
		# save start time in mm/dd/yr & time format
		fid = "init_time" +  timestr + fid1;  
		np.savetxt( fid, time.localtime());

		#  save start time as an integer
		init_time = time.time();

		# set cct, intensity and color channels of lights
		cct  = 0;
		intensity = 0;
	else:
		# set cct value
		cct = np.random.randint(1800, 10000); 
		intensity = 100;
		
	lights_func.cct_all( cct, intensity); # change light output : cct
	# set_color_channel = intensity * np.random.randint(2, size=8);
	# lights_func.sources_all(set_color_channel[0], set_color_channel[1], set_color_channel[2], set_color_channel[3], set_color_channel[4], set_color_channel[5], set_color_channel[6], set_color_channel[7] )

	# check output of light channels
	for numlight in range( 0, 10, 1):
		if numlight == 0:
			lightresponse = np.asarray(lights_func.get_sources( lightsidx[numlight,0], lightsidx[numlight,1] ) );
		else:
			templightresponse = np.asarray(lights_func.get_sources( lightsidx[numlight,0], lightsidx[numlight,1] ) );
			lightresponse = np.append( lightresponse, templightresponse, axis=0 )

	# get output of tof sensors
	temp =  np.asarray(tof.get_distances());
	tof_data.append( np.ndarray.flatten( temp) );

	# get response of color sensors
	tempRGB = np.asarray(csensors_func.read_all());
	tempRGB = np.ndarray.flatten( tempRGB )


	# append current sensor data to previous sensor data
	cct_all_vals.append( cct );
	csensor_all_vals.append( tempRGB );
	time_all_vals.append(time.time() - init_time );
	# set_color_channels_all_vals.append(set_color_channel);
	light_response_all_vals.append(lightresponse);

	print(i)

fid = "tof_data_"+ timestr + fid1;
np.savetxt( fid, tof_data );

fid = "time_all_vals_" + timestr + fid1;
np.savetxt( fid, time_all_vals);

fid = "csensor_all_vals_" + timestr + fid1;
np.savetxt( fid, csensor_all_vals);

fid = "cct_all_vals_" + timestr + fid1;
np.savetxt( fid, cct_all_vals);

fid = "light_response_" + timestr + fid1;
np.savetxt( fid, light_response_all_vals);

fid = "end_time_" + timestr + fid1;
np.savetxt( fid, time.localtime());


