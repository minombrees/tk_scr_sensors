import sys, os, time, math, rospy
import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats
import scipy.ndimage
from scipy.ndimage.filters import maximum_filter
from scipy.misc import toimage
from skimage.measure import label, regionprops
from skimage import morphology
import matplotlib.pyplot as plt

os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/time_of_flight")
sys.path.append(os.getcwd())
import SCR_TOF_client as tof
os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/lights")	
sys.path.append(os.getcwd())
import SCR_OctaLight_client as lights_func 										# lights ros functions
os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/color_sensors")
sys.path.append(os.getcwd())
import SCR_COS_client as csensors_func											# color sensors ros functions

os.chdir("/home/lesa/catkin_ws/src/scr_control/scripts/tk")

# fid = 'C:\\Users\\tk_wo\\OneDrive\\Fall_\\csensors_tof_locs.txt';
# csensors_tof_locs = np.loadtxt(fid, delimiter = ',');
# print( csensors_tof_locs )

fidOcc = "/home/lesa/catkin_ws/src/scr_control/scripts/Toufiq/Occupancy_info.txt";

# load needed files
fid = "cos_comp_array.txt";
cos_comp_array = np.loadtxt( fid, delimiter =',' );
fid = "freq_comp_array.txt";
freq_comp_array = np.loadtxt( fid, delimiter =',' );
fid = "sensor2Exy_dims.txt";
sensor2Exy_dims = np.loadtxt( fid, delimiter =',' );
fid = "bxy.txt";
Bxy = np.loadtxt( fid, delimiter =',' );
fid = "east_window_tof_locations.txt";
east_win_locs = np.loadtxt( fid, delimiter =',' );
fid = "north_window_tof_locations.txt";
north_win_locs = np.loadtxt( fid, delimiter =',' );
# fid = "csensors_tof_locs.txt";
# csensors_tof_locs = np.loadtxt( fid, delimiter =',' );

# declare variables
counting = 1;
hgt_SCR = 2820; #mm
light_thresh = 1000;
win_counter = np.zeros((2,1)); # 1. north, 2. east
win_counter_thresh = 100;
win_counter_bool = np.zeros((2,1));

def tofHgtComp( Exy, freq_comp_array, sensor2Exy_dims ):

	Exy = hgt_SCR - Exy;
	Exy[ Exy < 100 ] = 0;
	Exy[ Exy > 2300 ] = 0;

	# compensate diff sections Exy = Exy * freqHgt;
	for i in range(0,18,1):
		idx = sensor2Exy_dims[i,:];
		idx1 = int(idx[1])-1; idx2 = int(idx[2]); 
		idx3 = int(idx[3])-1; idx4 = int(idx[4]);
		# print( idx1, idx2, idx3, idx4 )
		temp = Exy[ idx1:idx2, idx3:idx4  ];
		# print(np.shape(temp))
		temp = np.polyval( freq_comp_array[i,:], temp);
		Exy[ idx1:idx2, idx3:idx4  ] = temp;

	Exy = 0.001 * Exy; #m
	return Exy

# Occupancy Mapping Function
def getOMaps( Fxy ):
	bw = Fxy > 0.1
	bw = morphology.remove_small_objects( bw, 8 )
	Fxy[ (bw==0) ] = 0;
	Fxy = scipy.signal.medfilt2d(Fxy, kernel_size=3)

	# Fxy = np.delete( Fxy, [26,27,28,29,48,49,50], 0)
	Fxy = np.delete( Fxy, [26,27,28,29,48,49,50], 0)
	Fxy = np.delete( Fxy, [20,21], 1)

	H = 0.2 * np.matrix(' 0 1 0; 1 1 1; 0 1 0 ')
	tFxy = scipy.ndimage.correlate(Fxy, H, mode='constant')

	rng =  np.linspace(0, 3.0, num=100)

	alpha = scipy.stats.norm( 0.94, 0.17).pdf(rng)
	alpha = np.max(alpha)
	
	sit = scipy.stats.norm( 0.94, 0.17).pdf(tFxy)
	sit = sit / alpha

	beta = scipy.stats.norm( 1.6, 0.2).pdf(rng)
	beta = np.max(beta)

	stand = scipy.stats.norm( 1.6, 0.2).pdf(tFxy)
	stand = stand / beta

	tFxy[ (sit<0.5) & (stand<0.5)] = 0;
	bw = tFxy > 0.1
	bw = morphology.remove_small_objects( bw, 8 )
	tFxy[ (bw==0) ] = 0;

	labels = label(tFxy, neighbors = 8)
	regions = regionprops(labels, coordinates='rc')

	num = len( regions )

	occlocs = []; occarea = []; occMAL = []; occhgt = [];

	if num > 0:
		I = toimage(tFxy)
		strel = morphology.disk( 1 )
		Ie = morphology.erosion( I, strel)
		Id = morphology.dilation( I, strel)

		Iobr = morphology.reconstruction( Ie, Id )
		Fgm = morphology.local_maxima( Iobr )
		# print( "Fgm = ", Fgm)
		Iobr[ (Fgm ==0 ) ] = 0

		labels = label(Fgm)	
		regions = regionprops(labels, tFxy, coordinates='rc')

		num = len( regions )
		if num > 0:
			for props in regions:
				occlocs.append( props.centroid)
				# occarea.append( props.area )
				# occMAL.append( props.major_axis_length )
				occhgt.append( props.max_intensity)

	return num, occlocs, occhgt

# tau_Mag = 0;
# def getRGBresponse_nearOcc( occlocs, allRGB ):
	
# 	occsRGB = []; occsMag = [];
# 	for i in range( 0, np.shape(occlocs)[0]):
# 		loc = np.asarray(occlocs[i])
# 		loc = loc[np.newaxis,:] ###
# 		tempdist = cdist( loc, csensors_tof_locs );
# 		# occRGB = np.zeros((1,3)); occMag = 0;
# 		idx = np.argmin( tempdist ); # nearest color sensor
# 		# tempRGB = np.asarray(csensors_func.read_all()); # read all color sensors

# 		if allRGB[idx,3] > tau_Mag: # use clear channel to determine if likely occupant
# 			occRGB = allRGB[idx,0:2];
# 			occMag = max(occRGB); # maximum of r,g,b channels
# 			occRGB = occRGB / occMag; # set rgb values between 0 and 1
# 		else: #check 2nd closest sensor response
# 			tempdist[0,idx] = 1000;
# 			idx = np.argmin( tempdist ); # nearest color sensor
# 			if allRGB[idx,3] > tau_Mag: # use clear channel to determine if likely occupant					
# 				occRGB = allRGB[idx,0:2];
# 				occMag = max(occRGB); # maximum of r,g,b channels
# 				occRGB = occRGB / occMag; # set rgb values between 0 and 1
# 			else:
# 				occRGB = np.zeros((1,3));
# 				occMag = 0;

# 		# print('RGB', np.shape(occRGB))
# 		occsRGB.append(occRGB); occsMag.append(occMag);

# 	return occsRGB, occsMag

def determineOccStatus( currOccs, prevOccs ):

	numCurr = 0; numPrev = 0; whgt = 0.3; ROI = 40; tau_door = 5; doorLoc = [60,150];
	tau_Occ = 5;

	if currOccs:
		currLoc = np.array( currOccs["Location"] ); 
		currHgt = np.array( currOccs["Height"] ); 
		# currMag = np.array( currOccs["Magnitude"] ); 
		# currRGB = np.array( currOccs["RGB"] ); 
		currPose = np.array( currOccs["Pose"] )
		numCurr = np.shape(currLoc)[0];
	
	if prevOccs:
		prevLoc = np.array( prevOccs["Location"] ); 
		prevHgt = np.array( prevOccs["Height"] ); 
		# prevMag = np.array( prevOccs["Magnitude"] ); 
		# prevRGB = np.array( prevOccs["RGB"] ); 
		prevCount = np.array( prevOccs["Count"] ); 
		prevMotion = np.array( prevOccs["Motion"]);
		prevStatus = prevOccs["Status"];
		prevLabel = np.array( prevOccs["Label"] )
		prevPose = np.array( prevOccs["Pose"] )
		numPrev = np.shape(prevLoc)[0];

	correlStates = np.zeros((numCurr, numPrev));

	# print('currHgt', np.shape(currHgt) )
	# print('prevHgt', np.shape(prevHgt) )

	for numC in range( 0, numCurr ): # cycle through current detections

		# wcol = currMag[numC];
		# if wcol < tau_Mag:
		# 	wcol = 0; 
		# else:
		# 	 wcol = 1;

		for numP in range( 0, numPrev ): # check against previous detections
			correlStates[ numC, numP ] = math.sqrt(sum( (currLoc[numC,0:1] - prevLoc[numP,0:1])**2 + \
										whgt * ( currHgt[numC] - prevHgt[numP] )**2 )); #+ \
										# wcol * ( currRGB[numC,0:2] - prevRGB[numP,0:2])^2 ));


	# print( 'correlStates: \n ', correlStates )
	idx_prev = []; idx_curr = []; 
	trackedOccs = {"Location": [], "Height": [], "Count": [], "Label": [], "Status":[], "Motion": [], "Pose": [] };

	# print("currPose", currPose);
	# print("prevPose", prevPose);

	for numC in range(0,numCurr):
		print('numP', numPrev)
		print('numC', numCurr)
		print( 'correlStates', correlStates )

		idx_min = np.argmin(correlStates[numC,:])  # most similar occupant between frames
		val_min = np.amin(correlStates[numC,:])
		
		if (val_min == np.amin( correlStates[:, idx_min] ) and val_min <= ROI) : # check if least similar for this occupant as well as all occupants
			trackedOccs["Location"].append( currLoc[numC, :] );
			trackedOccs["Height"].append(currHgt[numC]);			
			trackedOccs["Count"].append(prevCount[idx_min] + 1);
			trackedOccs["Label"].append(prevLabel[idx_min]);
			trackedOccs["Status"].append({'tracked'});
			trackedOccs["Motion"].append(currLoc[numC,:] - prevLoc[idx_min,:]);
			trackedOccs["Pose"].append(round( 0.5 * (currPose[numC] + prevPose[idx_min]) ));
			correlStates[ :, idx_min ] = 10000;
			idx_prev.append(idx_min);
			idx_curr.append(numC);

	# if not 'trackedOccs' in locals():
			# trackedOccs = {'Label': 0, 'Status': '', 'Location': [0,0], 'Motion': [0,0], 'Height': 0.0, 'Pose': 10, 'RGB': [0,0,0], 'Magnitude': 0};
		# trackedOccs = { 'Label', 'DetectionCount', 'MissingCount', 'Status', 'Location', 'Motion', 'Height', 'Pose', 'RGB', 'Magnitude', 'ClosestOcc' };
	# print(idx_prev, idx_curr)
	for numC in range(0, numCurr): #cycle through current detections
		# print('numC:', numC, 'numCurr:', numCurr)
		if numC not in idx_curr: #not tracked
			print(111)

			if math.sqrt( sum( (currLoc[numP,:] - doorLoc)**2 ) ) > tau_door: #outside door region
				for numP in range(0, numPrev): # check if previous occupant
					if numP not in idx_prev:
						if prevStatus[numP] == 'missed' and math.sqrt( sum( (prevLoc[numP,:] - currLoc[numC,:])**2 ) ) < 2 * ROI:
							trackedOccs["Label"].append(prevLabel[numP]);					
							trackedOccs["Status"].append({'prev_missed'});
							trackedOccs["Count"].append(prevCount[numP] + 1);
							trackedOccs["Motion"].append(prevMotion[numP,:]);
							# idx_prev.append(numP);


						# else: # missed initial detection inside?
						# # check proximity to other occupants
						# 	tempLoc1 = currLoc[numC,:]; tempLoc2 = currLoc;
						# 	del tempLoc2[numC,:];
						# 	if np.amin( cdist( tempLoc1, tempLoc2 ) ) > tau_Occ:
						# 		alllabels = trackedOccs["Label"];

						# 		for i in range(1,200):
						# 			if i not in alllabels:
						# 				idx = i;
						# 				break;

						# 		trackedOccs["Label"].append(idx);	
						# 		trackedOccs["Status"].append({'possible'});
						# 		trackedOccs["Count"].append(1);
						# 		trackedOccs["Motion"].append([0,0]);
							# idx_prev.append(numP);


			elif math.sqrt( sum( (currLoc[numP,:] - doorLoc)**2 ) ) < tau_door: # new occupant
				alllabels = trackedOccs["Label"];

				for i in range(1,200):
					if i not in alllabels:
						idx = i;
						break;
				# difflabels = np.diff( alllabels );
				# print('difflabels:', difflabels)

				# try:
				# 	idx = difflabels.index(~1)
				# except ValueError:
				# 	idx = np.amax(alllabels) + 1;

				trackedOccs["Label"].append(idx);
				trackedOccs["Status"].append({'new'});
				trackedOccs["Count"].append(1);
				trackedOccs["Motion"].append([0,0]);
	
	for numP in range(0, numPrev):

		if numP not in idx_prev: # previous occupant not located
			trackedOccs["Location"].append( prevLoc[numP, :] );
			trackedOccs["Height"].append(prevHgt[numP]);			
			trackedOccs["Count"].append(prevCount[numP] - 1);
			trackedOccs["Label"].append(prevLabel[numP]);					
			trackedOccs["Motion"].append(prevMotion[numP,:]);
			trackedOccs["Pose"].append(prevPose[numP]);
				
			if math.sqrt( sum( (prevLoc[numP,:] - doorLoc)**2 ) ) < tau_door:
				trackedOccs["Status"].append({'exit'});
			else:
				trackedOccs["Status"].append({'missing'});
	

	return trackedOccs

def respondOccActivities( num, occlocs, numSit, numStand):
	occsActs = np.zeros((4,1));

	if num != 0:
		occsActs[0] = num; # total occupants
		occsActs[1] = numStand;
		occsActs[2] = numSit;

		occsActs[3] = 0; # fall detection

		for i in range( 0, np.shape(occlocs)[0]):
			idx_north = cdist(occlocs[i,:], north_win_locs);
			if np.amin(idx_north) < 1: #empty matrix
				win_counter[0] = win_counter[0] + 1;
			else:
				win_counter[0] = 0;

			idx_east = cdist(occlocs, east_win_locs);
			if np.amin(idx_east) < 1: #empty matrix
				win_counter[1] =  win_counter[1] + 1;
			else:
				win_counter[1] = 0;

		for i in range(0,2):
			if win_counter[i] > win_counter_thresh:
				win_counter_bool[i] = 1;
			else:
				win_counter_bool[i] = 0;

		# print('Counter', win_counter);
		# print('Bool', win_counter_bool);
	np.savetxt( fidOcc, occsActs );


currOccs = {}; prevOccs = {};
posethresh = 1.2; #meters

## main function
# begin detection
while( counting ):
	Exy = np.asarray(tof.get_distances());
	Exy = Exy * cos_comp_array;
	Exy = tofHgtComp( Exy, freq_comp_array, sensor2Exy_dims );

	Fxy = Exy - Bxy;
	Fxy[ Fxy < 0.1 ] = 0;
	Fxy[ Fxy > 2.3 ] = 0;

	# get location, height of 'occupants'
	num, occlocs, occhgt = getOMaps( Fxy )

	
	# # get response of color sensors
	# allRGB = np.asarray(csensors_func.read_all());	
	# occRGB, occMag = getRGBresponse_nearOcc( occlocs, allRGB );

	# # create table of current detections
	currOccs = {"Location": [], "Height": [], "Count": [], "Label": [], "Status":[], "Motion": [], "Pose": [] };
	
	currOccs["Location"] = occlocs;	currOccs["Height"] = occhgt;
	# currOccs["RGB"] = occRGB; currOccs["Magnitude"] = occMag;
	for i in range(0,np.shape(occlocs)[0]):
		currOccs["Label"].append(i);
		currOccs["Count"].append(1);
		currOccs["Motion"].append([0,0]);
		currOccs["Status"].append({'possible'});
		# tempLocs = currOccs["Location"];
		# tempLocs[i] = [];
		# currOccs["Closest_Neighbor"] = min( cdist(currOccs["Location"][i], tempLocs ) );
		if currOccs["Height"][i] <= posethresh: # sitting
			currOccs["Pose"].append(0);
		else: #standing
			currOccs["Pose"].append(1);

	
	print( 'Frame:', counting )
	print('currOccs:', currOccs)
	print('prevOccs:', prevOccs)



	## perform tracking


	if not prevOccs:
		if currOccs["Location"]: #initialization
			prevOccs = currOccs.copy();

	if prevOccs:
		# print(np.shape(prevOccs))
		prevOccs = determineOccStatus( currOccs, prevOccs );

		## determine 'true' occupant count, locations, etc.
		tempStatus = prevOccs.copy(); tempStatus = tempStatus["Status"];	
		numOcc = 0; numSit = 0; numStand = 0; locsOcc = [];
		for i in range(0,np.shape(tempStatus)[0]):
			if tempStatus[i] == 'missing' and prevOccs["Count"][i] > -500:
				numOcc = numOcc + 1;
				if prevOccs["Pose"][i] == 0:
					numSit = numSit + 1;
				elif prevOccs["Pose"][i] == 1:
					numStand = numStand + 1;

				locsOcc.append(prevOccs["Location"][i,:])
				# determine which ToF sensor region

			if tempStatus[i] == 'tracked' and prevOccs["Count"][i] > -500:
				numOcc = numOcc + 1;
				if prevOccs["Pose"][i] == 0:
					numSit = numSit + 1;
				elif prevOccs["Pose"][i] == 1:
					numStand = numStand + 1;

				locsOcc.append(prevOccs["Location"][i,:])

		print('numOcc:', numOcc)
		print('numSit:', numSit)
		print('numStand:', numStand)

		if numOcc != 0:
			respondOccActivities( numOcc, locsOcc, numSit, numStand );


		# currOccs = getRGBresponse_nearOcc( locsOcc );


	counting = counting + 1;
	# print( 'Tracked', (prevOccs["Location"]) )
