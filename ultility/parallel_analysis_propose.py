#######################################
# Create by NguyenTT
# This tool help to
# 1. Generate platform file
# Input: parrallel_analysis.py network_file
#		Set parameter in script (TODO: to be argument)
#		network_file: file descript the DNN with format
#			Firstline is the input layer (W1xH1xC1)
#			From second line, DNN's layer (Activation, Weight, Estimated Computation (FLops))
# Ouput: 
#####################################

import re
import string
import sys
import math
import pprint
import random
import Queue

GOAL_COST = 1
GOAL_PERFORMANCE = 2
GOAL_MEM = 3
##--------MAIN FUNCTION-----------
def main():
	if len(sys.argv) < 2:
		print 'Syntax error! Lack of arguments.'
		print NOTICE
		return
	networkFileName = str(sys.argv[1])
	# save_network(networkFileName)
	# return
	#1. SET PARAMETER
	MAX_MINIBATCH = 256.0 
	TOTAL_SAMPLE = 1280000 #1280000 #For imageNet. 1280000
	MAX_CHANNEL = 64 #Use in hModel For ResNet and VGG
	#if (MAX_COLUMN == 0) or (MAX_COLUMN is None):
		# MAX_COLUMN = TOTAL_SAMPLE/MAX_MINIBATCH #5000
	
	MEM_PER_NODE = 4e9 #16 GB for Nvidia V100
	NODE_SPEED = 9.3e12 # flop for NVIDIA Tesla P100
	LINK_BW = 12.5E9 # Bype per second
	BW_FACTOR = 1/LINK_BW
	LATENCY_FACTOR = 500E-9 #second. 5 hops of 100ns switch latency
	
	SEGMENT = 2
	GOAL = GOAL_PERFORMANCE
	BYTE_PER_ITEM = 4.0
	ITEM_PER_NODE = MEM_PER_NODE/BYTE_PER_ITEM
	#2. Load DNN
	network = load_network(networkFileName)
	print_network(network)
	totalWeight = 0;
	totalComp = 0;
	totalActivation = 0;
	maxActivation = 0;
	for i in range(0,len(network['lay'])):
		layer = network['lay'][i]
		if maxActivation < layer[0]:
			maxActivation = layer[0]
		totalActivation = totalActivation + layer[0]
		totalWeight = totalWeight + layer[1]
		totalComp = totalComp + 2*layer[2]  #include FW, BW and TODO: Weight update?
	print "Total activation: " + str(totalActivation) + "items"
	print "Total weight: " + str(totalWeight) + "items"
	print "Total comp: " + str(totalComp) + "items"
	print "Max Activation" + str(maxActivation) + "items"
	
	#3.4 Analysis iModel
	results=[]
	print "==========iMODEL PARALLELISM=========="
	L = len(network['lay'])
	MAX_COLUMN = L
	if (MAX_COLUMN > MAX_MINIBATCH):
		MAX_COLUMN = MAX_MINIBATCH
	MAX_ROW = MAX_CHANNEL
	
	maxColIdx = int(math.log(MAX_COLUMN,2)) #Number of composite layer (or node)
	maxRowIdx = int(math.log(MAX_ROW,2))
	for colIdx in range(1,maxColIdx+1):
		colNumber = int(math.pow(2,colIdx))
		for rowIdx in range(1,maxRowIdx+1):
			rowNumber = int(math.pow(2,rowIdx))
			print "Considering colNumber: " + str(colNumber) + ",rowNumber: " + str(rowNumber)
			# Partition the layer
			approxComp = totalComp/colNumber
			firstLayerIdx = []
			lastLayerIdx = []
			nodeTotalWeight = []
			nodeTotalActivation = []
			nodeTotalComp = []
			nodeTotalWeight2 = []
			nodeTotalActivation2 = []
			nodeTotalComp2 = []
			totalComp2 =0
			for i in range(0,colNumber):
				lastLayerIdx.append(-1)
				firstLayerIdx.append(-1)
				nodeTotalWeight.append(0)
				nodeTotalActivation.append(0)
				nodeTotalComp.append(0)
				nodeTotalWeight2.append(0)
				nodeTotalActivation2.append(0)
				nodeTotalComp2.append(0)
			firstLayerIdx[0] = 0	
			lastLayerIdx[colNumber-1] = L-1;
			
			i = 0
			j = 0
			currentComp = 0
			lastComp = currentComp
			while (j < L and i < colNumber-1):
				currentComp = currentComp + 2*network['lay'][j][2]
				if currentComp > approxComp:
					if (currentComp - approxComp) > (approxComp - lastComp):
						lastLayerIdx[i] = j - 1
						firstLayerIdx[i+1] = j
						#totalComp = totalComp - lastComp
						currentComp = 2*network['lay'][j][2]
					else:
						lastLayerIdx[i] = j
						firstLayerIdx[i+1] = j+1
						currentComp = 0
					i = i+1
				lastComp = currentComp
				j = j+1
			
			if firstLayerIdx[colNumber-1] == -1:
				print "Cannot split model"
				break
			
			#Find microBatch
			microBatch = MAX_MINIBATCH
			maxBatchPerNode = MAX_MINIBATCH
			nodeInput = network['in']
			maxMem = -1
			for i in range(0,colNumber):
				if i > 0:
					nodeInput = network['lay'][lastLayerIdx[i-1]][0]
				for j in range(firstLayerIdx[i],lastLayerIdx[i]):
					nodeTotalWeight[i] = nodeTotalWeight[i] + network['lay'][j][1]
					nodeTotalActivation[i] = nodeTotalActivation[i] + network['lay'][j][0]
					nodeTotalComp[i] = nodeTotalComp[i] + 2*network['lay'][j][2]
				# Last layer in columns
				nodeTotalWeight2[i] = nodeTotalWeight[i] + network['lay'][lastLayerIdx[i]][1]/rowNumber
				nodeTotalActivation2[i] = nodeTotalActivation[i] + network['lay'][j][0]/rowNumber
				nodeTotalComp2[i] = nodeTotalComp[i] + 2*network['lay'][j][2]/rowNumber
				totalComp2 = totalComp2 + nodeTotalComp2[i]
				
				nodeTotalWeight[i] = nodeTotalWeight[i] + network['lay'][lastLayerIdx[i]][1]
				nodeTotalActivation[i] = nodeTotalActivation[i] + network['lay'][j][0]
				nodeTotalComp[i] = nodeTotalComp[i] + 2*network['lay'][j][2]

				#print "Node " + str(i) + " keeps layer " + str(firstLayerIdx[i]) + "-" + str(lastLayerIdx[i]) + "[" + str(nodeTotalActivation[i]) + "," + str(nodeTotalWeight[i]) + "," + str(nodeTotalComp[i]) + "]"
				if (i ==0):
					maxBatch = (ITEM_PER_NODE - 2*nodeTotalWeight[i])/(2*nodeInput + 2*nodeTotalActivation[i]) 
				else:
					maxBatch = (ITEM_PER_NODE - 2*nodeTotalWeight[i])/(nodeInput + 2*nodeTotalActivation[i]) 
				if maxBatch < maxBatchPerNode:
					maxBatchPerNode = maxBatch

			print "maxBatchPerNode: " + str(maxBatchPerNode)
			maxIndex = int(math.log(maxBatchPerNode,2))
			MAX_SEGMENT = colNumber
			if (MAX_MINIBATCH/colNumber > maxBatchPerNode):
				microBatch = math.pow(2,maxIndex)
				if MAX_SEGMENT > MAX_MINIBATCH/microBatch:
					MAX_SEGMENT = MAX_MINIBATCH/microBatch
			else:
				microBatch = MAX_MINIBATCH/colNumber
				
			
			for seg in range(int(math.log(MAX_SEGMENT,2)),int(math.log(MAX_SEGMENT,2))+1):
				Tcomp = 0
				Tcomm = 0
		
				SEGMENT = int(math.pow(2,seg))
				miniBatch = microBatch*SEGMENT
				
				maxMem = 0
				lastActivation = 0
				for i in range(0,colNumber):
					if (i ==0):
						nodeMem = BYTE_PER_ITEM*((microBatch)*(nodeInput + 2*nodeTotalActivation[i]) + 2*nodeTotalWeight2[i]) #bytes
					else:
						nodeMem = BYTE_PER_ITEM*((microBatch)*(2*nodeInput + 2*nodeTotalActivation[i]) + 2*nodeTotalWeight2[i]) #bytes
					if (maxMem < nodeMem):
						maxMem = nodeMem
					if i < colNumber - 1:	
						lastActivation = lastActivation + network['lay'][lastLayerIdx[i]][0]
				lastActivation2 = lastActivation+  network['lay'][lastLayerIdx[colNumber-1]][0]
				print "lastActivation: " + str(lastActivation)
				Tcomp = (TOTAL_SAMPLE/SEGMENT)*(2*totalComp2/NODE_SPEED)
				print totalComp, totalComp2
				Tcomp = Tcomp + ((SEGMENT-1)*TOTAL_SAMPLE/SEGMENT)*(nodeTotalComp2[0]+nodeTotalComp2[colNumber-1])/(2*NODE_SPEED)		
				
				Tcomm = (TOTAL_SAMPLE/miniBatch)*2*((colNumber-1)*LATENCY_FACTOR + lastActivation*miniBatch*BYTE_PER_ITEM*BW_FACTOR/SEGMENT)
				Tcomm = Tcomm + (3*TOTAL_SAMPLE/miniBatch)*(colNumber*math.log(rowNumber,2)*LATENCY_FACTOR + (rowNumber-1)*lastActivation2*miniBatch*BYTE_PER_ITEM*BW_FACTOR/(rowNumber*SEGMENT))
				print (3*TOTAL_SAMPLE/miniBatch)*(colNumber*math.log(rowNumber,2)*LATENCY_FACTOR)
				print (3*TOTAL_SAMPLE/miniBatch)*((rowNumber-1)*lastActivation2*miniBatch*BYTE_PER_ITEM*BW_FACTOR/(rowNumber*SEGMENT))
				result = {'name':'imodel'+str(SEGMENT),'B':str(miniBatch) + "(" + str(int(microBatch)) +")",'p':rowNumber*colNumber,'mMem':maxMem,'Tcomp':Tcomp,'Tcomm':Tcomm,'p2': str(rowNumber) + "x" + str(colNumber)}
				#str(miniBatch) + "(" + str(int(microBatch)) +")"
				results.append(result)
				
	#3.5 Analysis iModel
	#print results
	#4. Summary
	line = "name\tB\tp\tMem (bytes)\tTcomp (s)\tTcomm(s)\tTime(s)\tCost(min*Node)"
	print line
	for i in range(0,len(results)):
		result = results[i]
		Tdata = result['Tcomp'] + result['Tcomm']
		cost = result['p'] * math.ceil(Tdata /60) # points = (node mins)
		line = result['name']
		line = line + "\t" + str(result['B'])
		line = line + "\t" + str(result['p']) + "(" + str(result['p2']) +")"
		line = line + "\t" + str(result['mMem'])
		line = line + "\t" + str(result['Tcomp'])
		line = line + "\t" + str(result['Tcomm'])
		line = line + "\t" + str(Tdata)
		line = line + "\t" + str(cost)
		print line
		
		
def load_network(networkFileName):
	nw = {'in':0,'lay':[]}
	
	print 'Read dnn from' + networkFileName
	f = open(networkFileName, 'r')
	lineIdx = 1
	for line in f:
		splitLine = line.replace('\r','')
		splitLine = splitLine.replace('\n','')
		splitLine = splitLine.split('\t')
		
		if len(splitLine) >= 3:
			nw['lay'].append([float(splitLine[0]),float(splitLine[1]),float(splitLine[2])])
		elif len(splitLine) == 1:
			nw['in'] = float(splitLine[0])
		else:
			print '[WARNING] Invalid format at line ' + str(lineIdx)			
	f.close()
	return nw	

def print_network(nw):
	print "INPUT: " + str(nw['in']) 
	print "LAYER:	(Activation, Weight, Operations)"
	for i in range(0,len(nw['lay'])):
		layer = nw['lay'][i]
		line = "Layer " + str(i) + ":\t"  + str(layer[0]) + "\t\t" + str(layer[1])+ "\t\t" + str(layer[2]) #+ "\n"
		print line
	
def save_network(networkFileName):
	#VGG16
	nw = {'in':244*244*3,'lay':[
		[224*224*64 ,3*3*3*64 ,224*224*64*3*3*3],
		[224*224*64 ,3*3*64*64 ,224*224*64*3*3*64],
		[112*112*64 ,0 ,112*112*64*2*2*64],
		[112*112*128 ,3*3*64*128 ,112*112*128*3*3*64],
		[112*112*128 ,3*3*128*128,112*112*128*3*3*128],
		[56*56*128 ,0,56*56*128*2*2*128],
		[56*56*256 ,3*3*128*256,56*56*256*3*3*128],
		[56*56*256 ,3*3*256*256,56*56*256*3*3*256],
		[56*56*256 ,3*3*256*256,56*56*256*3*3*256],
		[28*28*256 ,0,28*28*256*2*2*256],
		[28*28*512 ,3*3*256*512,28*28*512*3*3*256],
		[28*28*512 ,3*3*512*512,28*28*512*3*3*512],
		[28*28*512 ,3*3*512*512,28*28*512*3*3*512],
		[14*14*512 ,0,14*14*512*2*2*512],
		[14*14*512 ,3*3*512*512,14*14*512*3*3*512],
		[14*14*512 ,3*3*512*512,14*14*512*3*3*512],
		[14*14*512 ,3*3*512*512,14*14*512*3*3*512],
		[7*7*512,0,7*7*512*2*2*512],
		[1*1*4096 ,7*7*512*4096,7*7*512*4096],
		[1*1*4096 ,4096*4096,4096*4096],
		[1*1*1000 ,4096*1000,4096*1000]
	]}

	#ResNet50
	# nw = {'in':244*244*3,'lay':[
		# [112*112*64,7*7*3*64,112*112*64*3*7*7],
		# [56*56*64,0,56*56*64*3*3*64],
		# [56*56*64,1*1*64*64,56*56*64*1*1*64],
		# [56*56*64,3*3*64*64,56*56*64*3*3*64],
		# [56*56*256,1*1*64*256,56*56*256*1*1*64],
		# [56*56*64,1*1*64*64,56*56*64*1*1*256],
		# [56*56*64,3*3*64*64,56*56*64*3*3*64],
		# [56*56*256,1*1*64*256,56*56*256*1*1*64],
		# [56*56*64,1*1*64*64,56*56*64*1*1*256],
		# [56*56*64,3*3*64*64,56*56*64*3*3*64],
		# [56*56*256,1*1*64*256,56*56*256*1*1*64],
		# [28*28*128,1*1*256*128,28*28*128*1*1*256],
		# [28*28*128,3*3*128*128,28*28*128*3*3*128],
		# [28*28*512,1*1*128*512,28*28*512*1*1*128],
		# [28*28*128,1*1*512*128,28*28*128*1*1*512],
		# [28*28*128,3*3*128*128,28*28*128*3*3*128],
		# [28*28*512,1*1*128*512,28*28*512*1*1*128],
		# [28*28*128,1*1*512*128,28*28*128*1*1*512],
		# [28*28*128,3*3*128*128,28*28*128*3*3*128],
		# [28*28*512,1*1*128*512,28*28*512*1*1*128],
		# [28*28*128,1*1*512*128,28*28*128*1*1*512],
		# [28*28*128,3*3*128*128,28*28*128*3*3*128],
		# [28*28*512,1*1*128*512,28*28*512*1*1*128],
		# [14*14*256,1*1*512*256,14*14*256*1*1*512],
		# [14*14*256,3*3*256*256,14*14*256*3*3*256],
		# [14*14*1024,1*1*256*1024,14*14*1024*1*1*256],
		# [14*14*256,1*1*1024*256,14*14*256*1*1*1024],
		# [14*14*256,3*3*256*256,14*14*256*3*3*256],
		# [14*14*1024,1*1*256*1024,14*14*1024*1*1*256],
		# [14*14*256,1*1*1024*256,14*14*256*1*1*1024],
		# [14*14*256,3*3*256*256,14*14*256*3*3*256],
		# [14*14*1024,1*1*256*1024,14*14*1024*1*1*256],
		# [14*14*256,1*1*1024*256,14*14*256*1*1*1024],
		# [14*14*256,3*3*256*256,14*14*256*3*3*256],
		# [14*14*1024,1*1*256*1024,14*14*1024*1*1*256],
		# [14*14*256,1*1*1024*256,14*14*256*1*1*1024],
		# [14*14*256,3*3*256*256,14*14*256*3*3*256],
		# [14*14*1024,1*1*256*1024,14*14*1024*1*1*256],
		# [14*14*256,1*1*1024*256,14*14*256*1*1*1024],
		# [14*14*256,3*3*256*256,14*14*256*3*3*256],
		# [14*14*1024,1*1*256*1024,14*14*1024*1*1*256],
		# [7*7*512,1*1*1024*512,7*7*512*1*1*1024],
		# [7*7*512,3*3*512*512,7*7*512*3*3*512],
		# [7*7*2048,1*1*512*2048,7*7*2048*1*1*512],
		# [7*7*512,1*1*2048*512,7*7*512*1*1*2048],
		# [7*7*512,3*3*512*512,7*7*512*3*3*512],
		# [7*7*2048,1*1*512*2048,7*7*2048*1*1*512],
		# [7*7*512,1*1*2048*512,7*7*512*1*1*2048],
		# [7*7*512,3*3*512*512,7*7*512*3*3*512],
		# [7*7*2048,1*1*512*2048,7*7*2048*1*1*512],
		# [1*1*1000,0,1*1*1000*7*7*2048],
		# [1*1*1000,1000*1000,1000*1000]
	# ]}
	# print 'Write dnn into ' + networkFileName
	fo = open(networkFileName, "w")
	line = str(nw['in']) + "\n"
	fo.writelines(line)
	for i in range(0,len(nw['lay'])):
		layer = nw['lay'][i]
		line = str(layer[0]) + "\t" + str(layer[1])+ "\t" + str(layer[2]) + "\n"
		fo.writelines(line)
	fo.close()
	
main()