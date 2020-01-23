#######################################
# Create by NguyenTT
# This tool help to numerically calculate the communication / computation time of different distributed training DNN
# Assumption: 
# (0) All the approaches are considered with the weak-scaling. B is in propotional to the number of GPUs p.
#		In data parallelism: fix samples per GPU.
# (1) When change the parallelism approaches, fixing the mini-batch size 
# (2) there is no temporal change in type of parallelism, i.e. different layers do not have different parallel strategy or different number of PEs.
# (3) No memory reduction likes low precision or sparsity / reused technique (x of layer l is the y of layer l-1)
# (4) Use of ring-based algorithm for both Allgather and Allreduce operation
# 
# Input: parrallel_analysis.py specification_files goal contrains
#	specification_flies: files descript the (a) dataset and dnn model *.net and (b) computer system *.sys
#		First file is the dataset specification and the model specification(*.net)
#			First line: Dataset: "Name";
#			Second line: Input: [N,C,W,H, D if have] 
#			Third line: DNN: "Name of the NET";
#			From the forth line. Each line is one layer
#			each layer has format |"NAME"|x|w|y|Flopcount|
#			where x = C,W,H,D if have;   w = C,F,K,K,K if have and y = F,W,H,D if have
#							in case of MPOOL w=0,0,K,K
#		Second file is the computer system specification (*.plat) TODO: use platform file of simgrid
#			First line: MEM_PER_NODE|MB
#			Second line: NODE_SPEED|FLOPS
#			Third line: Bandwidth| Bps
#			Forth line: Latency|second
#			Network description
#	goal: Goal of the analysis 1.Performance 2. Memory  3. Cost
#		Performance: Training time per epoch
#		Memory:	??
#		Cost consider as: Cost = p x T
#	contrains
#		max-mini-batch
#		batch-per-node?
#		max-node
# 
# Ouput: Tried to generate all the cases based on the constrains. 
# 		TODO: and then sort them by the goal.
###########################################

import re
import string
import sys
import math
import pprint
import random
import Queue
import time
import argparse

##-----------CONSTANT-------------
GOAL_COST = 1	
GOAL_PERFORMANCE = 2
GOAL_MEM = 3

ALGORITHM_RING = 1

PARATYPE_ALL = "a" #all
PARATYPE_SEQENTIAL = "o" #one node
PARATYPE_SPATIAL = "s" # spatial
PARATYPE_PIPE = "p" #pipe
PARATYPE_FILTER = "f" #filter
PARATYPE_CHANNEL = "c" #channel
PARATYPE_DATA = "d" #data
PARATYPE_DATA_SPATIAL = "ds" #data + spatial
PARATYPE_DATA_FILTER = "df" # data + filter

##----------GLOBAL VALUE----------
class GlobalVal:
	GOAL = GOAL_PERFORMANCE
	MAX_MINIBATCH = 256.0
	MAX_RANK = 128  #Required from users.
	FIX_MIRCO_BATCH = None	
	NODE_SPEED = 7.8E12 # double precision flop for NVIDIA Tesla V100
	MEM_PER_NODE = 16e9
	BYTE_PER_ITEM = 4.0 #Bytes
	ITEM_PER_NODE = MEM_PER_NODE/BYTE_PER_ITEM
	BW_FACTOR = 1/12.5E9
	LATENCY_FACTOR = 500E-9
	TOTAL_SAMPLE = 0
	GPU_PER_NODE = 4
	
	def __str__(self):
		print "GOAL", str(self.GOAL)
		print "MAX_MINIBATCH", str(self.MAX_MINIBATCH)
		print "MAX_RANK", str(self.MAX_RANK)
		print "FIX_MIRCO_BATCH", str(self.FIX_MIRCO_BATCH)
		print "NODE_SPEED", str(self.NODE_SPEED)
		print "MEM_PER_NODE", str(self.MEM_PER_NODE)
		print "ITEM_PER_NODE", str(self.ITEM_PER_NODE)
		print "BW_FACTOR", str(self.BW_FACTOR)
		print "LATENCY_FACTOR", str(self.LATENCY_FACTOR)
		print "TOTAL_SAMPLE", str(self.TOTAL_SAMPLE)
		print "GPU_PER_NODE", str(self.GPU_PER_NODE)
		return ""
		

##---------PARSER-----------------
argumentparser = argparse.ArgumentParser()
argumentparser.add_argument('-net', help="filename of the dataset specification and the model specification(*.net)")
argumentparser.add_argument('-plat', help="filename of the computer system specification (*.plat)")
argumentparser.add_argument('-goal', type=int, help="Goal of the analysis 1.Performance 2. Memory  3. Cost")
argumentparser.add_argument('--cmaxB', type=int, help="Constrain: Maximum mini-Batch size")
argumentparser.add_argument('--cmaxp', type=int, help="Constrain: Maximum number of node")
argumentparser.add_argument('--cBon', type=int, help="Constrain: Micro-batch size per node")
argumentparser.add_argument('--paratype', help='''parallelism type a: all, o: sequential in one PE, s: spatial, p:pipeline, f:filter, c:channel,d:data.
Multiple type can be seperated by ",". Other than that it is hybrid.
For example, ds refers to hybrid of data + spatial but d|s refers to analysis of data and spatial''')
argumentparser.add_argument('--debug', help="Debug mode. any character mean yes")
##--------MAIN FUNCTION-----------
def main(args):
	#1. GET Argment
	netFileName = str(args.net)
	platFileName = str(args.plat)
	paratype = "a"
	if args.paratype is not None:
		paratype = str(args.paratype)
	paraTypes = paratype.split(',')
	
	debug = False
	if args.debug is not None:
		debug = True
	g = GlobalVal()
	if args.goal is not None:
		g.GOAL = int(args.goal)
	if args.cmaxB is not None:
		g.MAX_MINIBATCH = float(args.cmaxB)
	if args.cmaxp is not None:
		g.MAX_RANK = float(args.cmaxp)
	if args.cBon is not None:
		g.FIX_MIRCO_BATCH = float(args.cBon)

	print "#######################################################"	
	print "-net", netFileName, "-plat",platFileName ,"-goal",g.GOAL, "--paratype", paraTypes
	print "--cmaxB", g.MAX_MINIBATCH, "--cmaxp", g.MAX_RANK, "--cBon", g.FIX_MIRCO_BATCH	
	
	# ###This code to test the communication
	# for i in range(1,9):
		# nodeNumber = math.pow(2,i)
		# bandwidth, latency = get_network_factor("ABCI",nodeNumber,ALGORITHM_RING)
		# alpha = latency
		# beta = 1/float(bandwidth)
		# for j in range(1,8):
			# messageSize = math.pow(2,j) * 1E6
			# Tcomm = 2*(nodeNumber-1)*(alpha + messageSize*beta/nodeNumber)
			# print nodeNumber,"\t",messageSize,"\t", Tcomm
		
		
	# return
	#2. Load Platform, Dataset and DNN
	platform = load_platform(platFileName)
	g.MAX_RANK = min(g.MAX_RANK,platform['max_node'])
	g.NODE_SPEED = platform['node_speed']
	g.MEM_PER_NODE = platform['mem_per_node']
	if debug == True:	
		print_platform(platform)
	data, network = load_network(netFileName,g.NODE_SPEED)
	if debug == True:	
		print_dataset(data)
	if debug == True:		
		print_network_all(network)
	metaData = summary_network2(network, debug)
	
	#3. Other Parameter
	g.BYTE_PER_ITEM = 4.0 #Bytes
	g.ITEM_PER_NODE = g.MEM_PER_NODE/g.BYTE_PER_ITEM
	g.BW_FACTOR = 1/platform['bandwidth']
	g.LATENCY_FACTOR = platform['latency']
	g.TOTAL_SAMPLE = data['size']
	g.GPU_PER_NODE = int(platform['gpu_per_node'])
	if debug == True:		
		print "****"
		print "global value g", g
	# 3.2. Preprocess
	minW = network['lays'][0]['x'][1]
	minFilter = network['lays'][0]['y'][0]
	minChannel = network['lays'][0]['x'][0]
	for i in range(0,len(network['lays'])):
		layer = network['lays'][i]
		minW = min(minW,layer['x'][1])
		minW = min(minW,layer['y'][1])
		minFilter = min(minFilter,layer['y'][0])
		minChannel = min(minChannel,layer['x'][0])
	metaData["minW"] = minW
	metaData["minFilter"] = minFilter
	metaData["minChannel"] = minChannel
	if debug == True:
		print metaData
		
	#4. Analysis
	results=[]
	# 4.1. Sequential implementation
	if ("a" in paraTypes) or ("o" in paraTypes):
		analysis_sequential(network,g,metaData,results)

	# 4.2. Data Paralllelism
	if ("a" in paraTypes) or ("d" in paraTypes):
		analysis_data(network, platform, g, metaData, results)

	# 4.3. Spatial Parallelism
	if ("a" in paraTypes) or ("s" in paraTypes):
		analysis_spatial(network, platform, g, metaData, results)
		
	# 4.4. Filter Paralllelism
	if ("a" in paraTypes) or ("f" in paraTypes):
		analysis_filter(network, platform, g, metaData, results)

	# 4.5. Channel Paralllelism		
	if ("a" in paraTypes) or ("c" in paraTypes):
		analysis_channel(network, platform, g, metaData, results)
	
	# 4.6 Pipeline Paralllelism and vModel
	if ("a" in paraTypes) or ("p" in paraTypes):
		analysis_pipeline(network, platform, g, metaData, results)
	
	# 4.7 Filter + Data Parallelism
	if ("a" in paraTypes) or ("df" in paraTypes):
		analysis_hybrid_df(network, platform, g, metaData, results)
		
	# 4.8 Spatial + Data Parallelism
	if ("a" in paraTypes) or ("ds" in paraTypes):
		analysis_hybrid_ds(network, platform, g, metaData, results)

	print_result(results)
	return


def analysis_sequential(network, g, metaData, results):
	print "==========SINGLE-NODE==============="
	totalIn = metaData["totalIn"]
	totalOut = metaData["totalOut"]
	totalWeight = metaData["totalWeight"]
	totalComp = metaData["totalComp"]
	
	maxSamplePerNode = float(g.ITEM_PER_NODE/2 - totalWeight)/(totalIn + totalOut)
	if maxSamplePerNode < 1:
		print "Not enough memory to store model and 1 sample"
		return results
		
	print "maxSamplePerNode: " + str(maxSamplePerNode)
	# mem4Sample = 2*(1*(totalIn + totalOut) + totalWeight)*BYTE_PER_ITEM
	# print "mem4Sample: " + str(mem4Sample)
	
	miniBatch = math.pow(2,int(math.log(maxSamplePerNode,2)))
	if g.FIX_MIRCO_BATCH is not None:
		print "Use FIX_MIRCO_BATCH set by user for miniBatch: ", g.FIX_MIRCO_BATCH
		miniBatch = g.FIX_MIRCO_BATCH
	
	memPerNode = 2*g.BYTE_PER_ITEM*(miniBatch*(totalOut + totalIn) + totalWeight)
	Tcomp = g.TOTAL_SAMPLE*totalComp
	Tcomm = 0
	nodeNumber = 1
	result = {'name':'single','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
	results.append(result)
	return results

def analysis_data(network, platform, g, metaData, results):
	#+ For data parallelism, find the biggest micro-batch size (samples per GPU), e.g., 512 in case of ALEXNET, 16 in case of VGG. Then I tried all the case whenever (p <= maxP) and (miniBatch <= maxB ). maxP and maxB are parameter passed by users. For example set maxP = 2048 and maxB =4096, respectively
	print "==========DATA PARALLELISM=========="
	totalIn = metaData["totalIn"]
	totalOut = metaData["totalOut"]
	totalWeight = metaData["totalWeight"]
	totalComp = metaData["totalComp"]

	maxSamplePerNode = float(g.ITEM_PER_NODE/2 - totalWeight)/(totalIn + totalOut)
	if maxSamplePerNode < 1:
		print "Not enough memory to store model and 1 sample"
		return results

	print "maxSamplePerNode: " + str(maxSamplePerNode)
	microBatch = math.pow(2,int(math.log(maxSamplePerNode,2)))
	if g.FIX_MIRCO_BATCH == None:
		g.FIX_MIRCO_BATCH = microBatch
	else:
		print "Use FIX_MIRCO_BATCH set by user, ", g.FIX_MIRCO_BATCH
		microBatch = g.FIX_MIRCO_BATCH
		
	maxIdx = int(math.log(g.MAX_RANK,2))
	for i in range(1,maxIdx+1):
		nodeNumber = math.pow(2,i)

		miniBatch = microBatch * nodeNumber
		if (miniBatch <= g.MAX_MINIBATCH):
			print microBatch, (totalIn + totalOut), totalWeight, g.BYTE_PER_ITEM
			memPerNode = 2*(microBatch*(totalIn + totalOut) + totalWeight)*g.BYTE_PER_ITEM
			Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
			bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
			alpha = latency
			beta = 1/float(bandwidth)
			#print alpha,beta,nodeNumber, g.BYTE_PER_ITEM*totalWeight, math.ceil(g.TOTAL_SAMPLE/miniBatch)
			#print (nodeNumber-1)*(alpha + totalWeight*g.BYTE_PER_ITEM*beta/nodeNumber)
			Tcomm = 2 * math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha + totalWeight*g.BYTE_PER_ITEM*beta/nodeNumber)
			#print Tcomm
			#print "---------------"
			result = {'name':'data','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
			results.append(result)		
	return results

def analysis_filter(network, platform, g, metaData, results):
	# + For Filter and Channel, there are 3 cases
	# Case 1: for given a number of GPU p, I use the same miniBatch size as they are in data parallelism with the same p. In that case, filter and channel always impossible because out-of-memory at each GPU
	# Case 2: for a given a number of GPU p, I tried to find the biggest (power-of-2) miniBatch size  that the total memory can fit to the memory of GPU. I change p from 2 to the biggest number based on the constraints in Table II of our paper in overleaf.
	# Case 3: Fix the mini BatchSize as set by user as the maxSampleperNode, increasing p...
	print "==========FILTER PARALLELISM=========="
	totalIn = metaData["totalIn"]
	totalOut = metaData["totalOut"]
	totalWeight = metaData["totalWeight"]
	totalComp = metaData["totalComp"]
	minFilter = metaData["minFilter"]	
	
	max_rank = min(g.MAX_RANK,minFilter)	
	print "max_rank", max_rank
	maxIdx = int(math.log(max_rank,2))
	G = len(network['lays'])
	totalOutExceptLast = totalOut - network['lays'][G-1]['out']
	
	#Case 1: Also fix batch per node = maxSamplePerNode of data.
	for i in range(1,maxIdx+1):
		nodeNumber = math.pow(2,i)
		miniBatch = g.FIX_MIRCO_BATCH * nodeNumber
		if (miniBatch <= g.MAX_MINIBATCH):
			memPerNode = 2*(miniBatch*(totalIn + totalOut) + float(totalWeight)/nodeNumber)*g.BYTE_PER_ITEM
			#print memPerNode, nodeNumber,miniBatch
			if memPerNode > g.MEM_PER_NODE:
				print "Not enough memory to store model and",miniBatch ,"samples in Filter-Parallelism with",nodeNumber,"GPUs"
				#break
			else:
				Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
				bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
				alpha = latency
				beta = 1/float(bandwidth)
				
				Tcomm = 3*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
				print "Get result from Case 1 with miniBatch", miniBatch, "nodeNumber",nodeNumber
				result = {'name':'filter1','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
				results.append(result)	
	
	# #Case 2: For each network size, use the biggest batch size
	for i in range(1,maxIdx+1):
		nodeNumber = math.pow(2,i)	
		maxMiniBatch = float(g.ITEM_PER_NODE/2 - float(totalWeight)/nodeNumber)/(totalIn + totalOut)
		maxMiniBatch = min(maxMiniBatch,g.MAX_MINIBATCH)
		miniBatch = math.pow(2,int(math.log(maxMiniBatch,2)))
		memPerNode = 2*(miniBatch*(totalIn + totalOut) + float(totalWeight)/nodeNumber)*g.BYTE_PER_ITEM
		Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
		bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
		alpha = latency
		beta = 1/float(bandwidth)
		Tcomm = 3*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
		print "Get result from Case 2 with miniBatch", miniBatch, "nodeNumber",nodeNumber
		result = {'name':'filter2','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
		results.append(result)

	#Case 3: Also fix batch per node = maxSamplePerNode or by user setting	
	maxSamplePerNode = float(g.ITEM_PER_NODE/2 - float(totalWeight)/nodeNumber)/(totalIn + totalOut)
	if maxSamplePerNode < 1:
		print "Not enough memory to store model and 1 sample"
		return results

	print "maxSamplePerNode: " + str(maxSamplePerNode)
	miniBatch = math.pow(2,int(math.log(maxSamplePerNode,2)))
	if g.FIX_MIRCO_BATCH is not  None:
		print "Use FIX_MIRCO_BATCH set by user, ", g.FIX_MIRCO_BATCH
		miniBatch = g.FIX_MIRCO_BATCH
		
	if (miniBatch <= g.MAX_MINIBATCH):
		for i in range(1,maxIdx+1):
			nodeNumber = math.pow(2,i)		
			memPerNode = 2*(miniBatch*(totalIn + totalOut) + float(totalWeight)/nodeNumber)*g.BYTE_PER_ITEM
			if memPerNode > g.MEM_PER_NODE:
				print "Not enough memory to store model and",miniBatch ,"samples in Filter-Parallelism with",nodeNumber,"GPUs"
				#break
			else:
				Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
				bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
				alpha = latency
				beta = 1/float(bandwidth)
				
				Tcomm = 3* math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
				print "Get result from Case 3 with miniBatch", miniBatch, "nodeNumber",nodeNumber
				result = {'name':'filter3','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
				results.append(result)	

def analysis_channel(network, platform, g, metaData, results):
	# + For Filter and Channel, there are 3 cases
	# Case 1: for given a number of GPU p, I use the same miniBatch size as they are in data parallelism with the same p. In that case, filter and channel always impossible because out-of-memory at each GPU
	# Case 2: for a given a number of GPU p, I tried to find the biggest (power-of-2) miniBatch size  that the total memory can fit to the memory of GPU. I change p from 2 to the biggest number based on the constraints in Table II of our paper in overleaf.
	# Case 3: Fix the mini BatchSize as set by user as the maxSampleperNode, increasing p...
	print "==========Channel PARALLELISM=========="
	totalIn = metaData["totalIn"]
	totalOut = metaData["totalOut"]
	totalWeight = metaData["totalWeight"]
	totalComp = metaData["totalComp"]
	minChannel = metaData["minChannel"]	
	
	max_rank = min(g.MAX_RANK,minChannel)	
	print "max_rank", max_rank
	maxIdx = int(math.log(max_rank,2))
	print maxIdx
	G = len(network['lays'])
	totalOutExceptLast = totalOut - network['lays'][G-1]['out']
	
	nodeRange = []
	for i in range(1,maxIdx+1):
		nodeRange.append(math.pow(2,i))
	if max_rank not in nodeRange:
		nodeRange.append(max_rank)
	#Case 1: Also fix batch per node = maxSamplePerNode of data.
	for nodeNumber in nodeRange:
		#nodeNumber = math.pow(2,i)
		miniBatch = g.FIX_MIRCO_BATCH * nodeNumber
		if (miniBatch <= g.MAX_MINIBATCH):
			memPerNode = 2*(miniBatch*(totalIn + totalOut) + float(totalWeight)/nodeNumber)*g.BYTE_PER_ITEM
			#print memPerNode, nodeNumber,miniBatch
			if memPerNode > g.MEM_PER_NODE:
				print "Not enough memory to store model and",miniBatch ,"samples in Channel-Parallelism with",nodeNumber,"GPUs"
				#break
			else:
				Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
				bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
				alpha = latency
				beta = 1/float(bandwidth)
				Tcomm = 3*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
				print "Get result from Case 1 with miniBatch", miniBatch, "nodeNumber",nodeNumber
				result = {'name':'chan1','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
				results.append(result)
		
	#Case 2: For each network size, use the biggest batch size
	for nodeNumber in nodeRange:
		#nodeNumber = math.pow(2,i)	
		maxMiniBatch = float(g.ITEM_PER_NODE/2 - float(totalWeight)/nodeNumber)/(totalIn + totalOut)
		maxMiniBatch = min(maxMiniBatch,g.MAX_MINIBATCH)
		miniBatch = math.pow(2,int(math.log(maxMiniBatch,2)))
		memPerNode = 2*(miniBatch*(totalIn + totalOut) + float(totalWeight)/nodeNumber)*g.BYTE_PER_ITEM
		Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
		bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
		alpha = latency
		beta = 1/float(bandwidth)
		Tcomm = 3*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
		print "Get result from Case 2 with miniBatch", miniBatch, "nodeNumber",nodeNumber
		result = {'name':'chan2','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
		results.append(result)

	#Case 3: Also fix batch per node = maxSamplePerNode or by user setting	
	maxSamplePerNode = float(g.ITEM_PER_NODE/2 - float(totalWeight)/nodeNumber)/(totalIn + totalOut)
	if maxSamplePerNode < 1:
		print "Not enough memory to store model and 1 sample"
		return results

	print "maxSamplePerNode: " + str(maxSamplePerNode)
	miniBatch = math.pow(2,int(math.log(maxSamplePerNode,2)))
	if g.FIX_MIRCO_BATCH is not  None:
		print "Use FIX_MIRCO_BATCH set by user, ", g.FIX_MIRCO_BATCH
		miniBatch = g.FIX_MIRCO_BATCH
	
	if (miniBatch <= g.MAX_MINIBATCH):
		for nodeNumber in nodeRange:
			#nodeNumber = math.pow(2,i)	
			memPerNode = 2*(miniBatch*(totalIn + totalOut) + float(totalWeight)/nodeNumber)*g.BYTE_PER_ITEM
			#print memPerNode, nodeNumber,miniBatch
			if memPerNode > g.MEM_PER_NODE:
				print "Not enough memory to store model and",miniBatch ,"samples in Channel-Parallelism with",nodeNumber,"GPUs"
				#break
			else:
				Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
				bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
				alpha = latency
				beta = 1/float(bandwidth)
				Tcomm = 3*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
				print "Get result from Case 3 with miniBatch", miniBatch, "nodeNumber",nodeNumber
				result = {'name':'chan3','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
				results.append(result)

def analysis_spatial(network, platform, g, metaData, results):
	# For weak scaling,  use the same miniBatch size as they are set by user
	# Only seperate the |W| dimension
	# To increase the scaling, split of some begining layers. 
	# The following layer work as sequential. They agregated value by perform an Allgather.
	print "==========SPATIAL PARALLELISM=========="
	splitLayerIdx = 0  # 1st layer in 1st parition. other in the 2nd partition
	##TODO: detect the best spliting
	if "AlexNet" in network['name']:
		splitLayerIdx = 12
	if "VGG16" in network['name']:
		splitLayerIdx = 30
	if "ResNet50" in network['name']:
		splitLayerIdx = 164
	if "CosmoFlow" in network['name']:
		splitLayerIdx = 5		
	
	totalWeight = metaData["totalWeight"]		
	minW = network['lays'][0]['x'][1]
	for i in range(0,splitLayerIdx + 1):
		layer = network['lays'][i]
		minW = min(minW,layer['x'][1])
		minW = min(minW,layer['y'][1])
	max_rank = min(g.MAX_RANK,minW)	
	print "max_rank", max_rank
	
	maxIdx = int(math.log(max_rank,2))
	for i in range(1,maxIdx+1):
		nodeNumber = math.pow(2,i)
		totalIn = 0
		totalOut = 0
		totalComp = 0
		for i in range(0,splitLayerIdx + 1):
			totalOut = totalOut + math.ceil(float(layer['out']) / nodeNumber)	#only divide by p for some begining layer
			totalIn = totalIn + math.ceil(float(layer['in']) / nodeNumber)		
			totalComp = totalComp + float(layer['comp']) / nodeNumber
		
		for i in range(splitLayerIdx+1,len(network['lays'])):
			totalOut = totalOut + layer['out']
			totalIn = totalIn + layer['in']
			totalComp = totalComp + layer['comp']
			
		maxSamplePerNode = float(g.ITEM_PER_NODE/2 - totalWeight)/(totalIn + totalOut)
		if maxSamplePerNode < 1:
			print "Not enough memory to store model and 1 sample when split into", nodeNumber, "at layer",splitLayerIdx
			continue
		else:
			print "maxSamplePerNode: " + str(maxSamplePerNode)
			miniBatch = math.pow(2,int(math.log(maxSamplePerNode,2)))
			if g.FIX_MIRCO_BATCH is not None:
				print "Use FIX_MIRCO_BATCH set by user, ", g.FIX_MIRCO_BATCH
				miniBatch = g.FIX_MIRCO_BATCH
	

			if (miniBatch <= g.MAX_MINIBATCH):
				memPerNode = 2*(float(miniBatch)*(totalIn + totalOut) + totalWeight)*g.BYTE_PER_ITEM  #(x + y)/ p but already divided by p 
				print nodeNumber, totalComp
				Tcomp = math.ceil(g.TOTAL_SAMPLE)*totalComp
				
				bandwidth, latency = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
				alpha = latency
				beta = 1/float(bandwidth)
				#Communication the halo exchange. Only apply for CONV and POOLING (which has kernel size > 1)
				# A GPU only need to communicate with its preceding and next GPUs along the ring except the first GPU and the last GPU (near the border of sample). Communication will be performed in 2 round in 2 directions clockwise and counter-clockwise along the ring so that no network conflict appear!!! 
				# Secondly, we group the communication of B sample at the same time to reduce the latency. 
				# The communication time can be estimated as the maximum communication of 1 GPU pair at each layer. 
				# The size of each end-2-end is (K-1)/2|W|
				Tcomm = 0
				for i in range(0,splitLayerIdx + 1):
					kernelSize = network['lays'][i]['w'][2] 
					if (kernelSize > 1):
						# Halo exchange the activation: halo(y)
						haloSize = network['lays'][i]['y'][1] * ((kernelSize -1)/2) * g.BYTE_PER_ITEM
						Thalo = 2*(alpha + haloSize*beta)
						
						# Halo exchange the input gradient: halo(dL/dx)
						haloSize = network['lays'][i]['x'][1] * ((kernelSize -1)/2) * g.BYTE_PER_ITEM
						Thalo = Thalo  + 2*(alpha + haloSize*beta)
					Tcomm = Tcomm + Thalo
				#Algather communication |y| at the begining of the 2nd partition (in the forward pass)
				#Scatter communication the input gradient  ==> factor of 2
				totalOutatSplit = float(network['lays'][splitLayerIdx]['out'])
				Tcomm = Tcomm + 2*(nodeNumber-1)*(alpha + (miniBatch*totalOutatSplit*g.BYTE_PER_ITEM*beta)/nodeNumber)
				Tcomm = Tcomm * math.ceil(g.TOTAL_SAMPLE/miniBatch)
				
				#Communication the weight at the end of each iteration
				Tcomm = Tcomm  + 2*(g.TOTAL_SAMPLE/miniBatch)*(nodeNumber-1)*(alpha + totalWeight*g.BYTE_PER_ITEM*beta/nodeNumber)
				result = {'name':'spatial','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
				results.append(result)

def analysis_pipeline(network, platform, g, metaData, results):
	return
	### TODO: Not yet finished revising
	# Case 1: Fix S = p then scale like parallelism. Fix batch per node .... 
	# Case 2: Choose S < p MiniBatch = S*micro_batch
	print "==========VModel and Pipeline PARALLELISM=========="
	G = len(network['lays'])
	max_rank = min(MAX_RANK,G)
	print "max_rank", max_rank
	maxIdx = int(math.log(max_rank,2)) 
	for idx in range(1,maxIdx+1):
		nodeNumber = int(math.pow(2,idx))
		print "----------",nodeNumber,"--------"
		# Partition the layer by finding the nearest side that make group computation = approximate. 
		approxComp = totalComp/nodeNumber
		compositeLayers = []
		for i in range(0,nodeNumber):
			comLay = {'id':i,'in':0,'out':0,'tcomp':0,'fl':-1,'ll':-1,'tw':0,'tin':0,'tout':0}
			compositeLayers.append(comLay)
		compositeLayers[0]['fl'] = 0	
		compositeLayers[nodeNumber-1]['ll'] = G-1	
		
		i = 0
		j = 1
		currentComp = network['lays'][0]['comp']
		lastComp = currentComp
		while (j < G and i < nodeNumber-1):
			currentComp = currentComp + network['lays'][j]['comp']
			if currentComp > approxComp:
				#if (currentComp - approxComp) > (approxComp - lastComp):
				compositeLayers[i]['ll'] = j-1
				compositeLayers[i+1]['fl'] = j
				currentComp = network['lays'][j]['comp']
				# else:
					# compositeLayers[i]['ll'] = j
					# compositeLayers[i+1]['fl'] = j + 1
					# currentComp = 0
				i = i+1
			lastComp = currentComp
			j = j+1

		if compositeLayers[nodeNumber-1]['fl'] == -1:
			print "Cannot split model into", nodeNumber, "partitions using average method."
			continue
			# TODO:
			# print "Cannot split model into", nodeNumber, "partitions using average method. Use naive method instead"
			# for i in range (0, nodeNumber):
				# compositeLayers[nodeNumber-1]['fl'] = i
				# compositeLayers[nodeNumber-1]['ll'] = i
			# compositeLayers[nodeNumber-1]['ll'] = G-1
		
		maxComp = 0
		totalComp
		for i in range (0, nodeNumber):
			firstLayerIdx = compositeLayers[i]['fl']
			lastLayerIdx = compositeLayers[i]['ll']
			compositeLayers[i]['in'] = network['lays'][firstLayerIdx]['in']	
			compositeLayers[i]['out'] = network['lays'][lastLayerIdx]['out']	
			for j in range(firstLayerIdx, lastLayerIdx+1):
				compositeLayers[i]['tin'] += network['lays'][j]['in']
				compositeLayers[i]['tout'] += network['lays'][j]['out']
				compositeLayers[i]['tcomp'] += network['lays'][j]['comp']
				compositeLayers[i]['tw'] += network['lays'][j]['weight']
			maxComp = max(maxComp,compositeLayers[i]['tcomp'])
			print "compositeLayers",i,": layer ", 	compositeLayers[i]["fl"], " to ", compositeLayers[i]["ll"]  
			#print compositeLayers[i]
			
		#Case A.1: VModel Purre-Use the microBatch off data parallelism	
		miniBatch = FIX_MIRCO_BATCH * nodeNumber
		maxMem = 2*(float(miniBatch)*(compositeLayers[0]['tin'] + compositeLayers[0]['tout']) + compositeLayers[0]['tw'])*BYTE_PER_ITEM
		for i in range (1, nodeNumber):
			memPerNode = 2*(float(miniBatch)*(compositeLayers[i]['tin'] + compositeLayers[i]['tout']) + compositeLayers[i]['tw'])*BYTE_PER_ITEM
			maxMem = max(maxMem, memPerNode)
		if maxMem > MEM_PER_NODE:
			print "Not enough memory to store a part of model and",miniBatch ,"samples in VModel-Parallelism with",nodeNumber,"GPUs"
		else:
			TOTAL_SAMPLE*totalComp
			Tcomm = 0
			#Communication is also follow the ring.
			#Assume that mapping composite layer 1 to GPU0. then composite layer 4 to GPU4 and composite layer 5 to GPU0 of 2nd node.
			Tcomm = 0
			for i in range(0,nodeNumber-1):
				bandwidth, latency = get_network_factor(platform,i+2,ALGORITHM_RING)
				alpha = latency
				beta = 1/float(bandwidth)
				Tcomm_node = alpha + microBatch * compositeLayers[i]['out'] * BYTE_PER_ITEM * beta
				Tcomm = Tcomm + Tcomm_node
			Tcomm =  2*(TOTAL_SAMPLE/miniBatch) * Tcomm
			result = {'name':'vModel','B':miniBatch,'p':nodeNumber,'mMem':maxMem,'Tcomp':Tcomp,'Tcomm':Tcomm}
			results.append(result)
		#Case B.1: Pipeline-Use the microBatch of data parallelism
		miniBatch = FIX_MIRCO_BATCH * nodeNumber
		microBatch = FIX_MIRCO_BATCH
		SEGMENT = nodeNumber
		maxMem = 2*(float(miniBatch)*(compositeLayers[0]['tin'] + compositeLayers[0]['tout'])/SEGMENT + compositeLayers[0]['tw'])*BYTE_PER_ITEM
		for i in range (1, nodeNumber):
			memPerNode = 2*(float(miniBatch)*(compositeLayers[i]['tin'] + compositeLayers[i]['tout'])/SEGMENT + compositeLayers[i]['tw'])*BYTE_PER_ITEM
			maxMem = max(maxMem, memPerNode)
		Tcomp = (TOTAL_SAMPLE/SEGMENT)*(nodeNumber + SEGMENT - 1)*(maxComp)
		
		#Communication is also follow the ring.
		#Assume that mapping composite layer 1 to GPU0. then composite layer 4 to GPU4 and composite layer 5 to GPU0 of 2nd node.
		Tcomm = 0
		for i in range(0,nodeNumber-1):
			bandwidth, latency = get_network_factor(platform,i+2,ALGORITHM_RING)
			alpha = latency
			beta = 1/float(bandwidth)
			Tcomm_node = alpha + microBatch * compositeLayers[i]['out'] * BYTE_PER_ITEM * beta
			Tcomm = max(Tcomm, Tcomm_node)
			#print i, alpha, beta, Tcomm_node, compositeLayers[i]['out']
		#print Tcomm	
		Tcomm = 2*(TOTAL_SAMPLE/miniBatch)*(nodeNumber + SEGMENT - 2) * Tcomm
		result = {'name':'pipe','B':miniBatch,'p':nodeNumber,'mMem':maxMem,'Tcomp':Tcomp,'Tcomm':Tcomm}
		results.append(result)
		
		#Case B.2: Pipeline-Find the biggest microBatch
		# TODO
	print_result(results)
	return
	
def analysis_hybrid_df(network, platform, g, metaData, results):
	# In this case, Filter for inside group (1 node = 4GPUs) P2 = 4
	# Data for inter groups (scale...) So that Mini Batch = micro batch * p/4
	print "==========DATA + FILTER PARALLELISM=========="
	totalIn = metaData["totalIn"]
	totalOut = metaData["totalOut"]
	totalWeight = metaData["totalWeight"]
	totalComp = metaData["totalComp"]
	minFilter = metaData["minFilter"]	
	
	G = len(network['lays'])
	totalOutExceptLast = totalOut - network['lays'][G-1]['out']
	P2 = g.GPU_PER_NODE 
	
	maxSamplePerNode = float(g.ITEM_PER_NODE/2 - float(totalWeight)/P2)/(totalIn + totalOut)
	if maxSamplePerNode < 1:
		print "Not enough memory to store model and 1 sample"
		return results
		
	microBatch = math.pow(2,int(math.log(maxSamplePerNode,2)))
	if g.FIX_MIRCO_BATCH == None:
		g.FIX_MIRCO_BATCH = microBatch
	else:
		print "Use FIX_MIRCO_BATCH set by user, ", g.FIX_MIRCO_BATCH
		microBatch = g.FIX_MIRCO_BATCH
		
	maxIdx = int(math.log(g.MAX_RANK,2))
	#Case 1: minibatch = sample per node * #node
	for i in range(2,maxIdx+1): #from 1 nodes
		nodeNumber = math.pow(2,i)
		P1 = nodeNumber/P2
		miniBatch = microBatch * P1
		if (miniBatch <= g.MAX_MINIBATCH):
			memPerNode = 2*(microBatch*(totalIn + totalOut) + float(totalWeight)/P2)*g.BYTE_PER_ITEM
			Tcomp = math.ceil(g.TOTAL_SAMPLE/nodeNumber)*totalComp
			
			#Communication inside nodes
			bandwidth1, latency1 = get_network_factor(platform,P2,ALGORITHM_RING)
			alpha = latency1
			beta = 1/float(bandwidth1)
			Tcomm = 3*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(P2-1)*(alpha*(G-1) + (miniBatch*totalOutExceptLast*g.BYTE_PER_ITEM*beta)/nodeNumber)
			
			#Communication between nodes (4 flows at the same time may reduce the bandwidth 4 times)
			bandwidth2, latency2 = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
			alpha = latency2
			bandwidth2 = bandwidth2 /GPU_PER_NODE
			beta = 1/float(bandwidth2)
			Tcomm = Tcomm + 2*math.ceil(g.TOTAL_SAMPLE/miniBatch)*(P1-1)*(alpha + totalWeight*g.BYTE_PER_ITEM*beta/nodeNumber)

			result = {'name':'df','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
			results.append(result)

def analysis_hybrid_ds(network, platform, g, metaData, results):
	# For weak scaling,  set the sample per node as user value (or biggest number by default)
	# Only seperate the |W| dimension
	# To increase the scaling, split of some begining layers. 
	# The following layer work as sequential. They agregated value by perform an Allgather.
	# In this case, Spatial inside 1 group = 1 node = 4GPU, data for inter-node
	print "==========DATA + SPATIAL PARALLELISM=========="
	splitLayerIdx = 0  # 1st layer in 1st parition. other in the 2nd partition
	##TODO: detect the best spliting
	if "AlexNet" in network['name']:
		splitLayerIdx = 12
	if "VGG16" in network['name']:
		splitLayerIdx = 30
	if "ResNet50" in network['name']:
		splitLayerIdx = 164
	if "CosmoFlow" in network['name']:
		splitLayerIdx = 5		
	P2 = g.GPU_PER_NODE 
	totalWeight = metaData["totalWeight"]		
	minW = network['lays'][0]['x'][1]
	for i in range(0,splitLayerIdx + 1):
		layer = network['lays'][i]
		minW = min(minW,layer['x'][1])
		minW = min(minW,layer['y'][1])
	max_rank = min(g.MAX_RANK,minW)	
	print "max_rank", max_rank
	
	totalIn = 0
	totalOut = 0
	totalComp = 0
	for i in range(0,splitLayerIdx + 1):
		totalOut = totalOut + math.ceil(float(layer['out']) / P2) #only divide by p2 for some begining layer
		totalIn = totalIn + math.ceil(float(layer['in']) / P2)
		totalComp = totalComp + float(layer['comp']) / P2
	
	for i in range(splitLayerIdx+1,len(network['lays'])):
		totalOut = totalOut + layer['out']
		totalIn = totalIn + layer['in']
		totalComp = totalComp + layer['comp']

	maxSamplePerNode = float(g.ITEM_PER_NODE/2 - totalWeight)/(totalIn + totalOut)
	if maxSamplePerNode < 1:
		print "Not enough memory to store model and 1 sample when split into", nodeNumber, "at layer",splitLayerIdx
		return
	else:
		print "maxSamplePerNode: " + str(maxSamplePerNode)
		if g.FIX_MIRCO_BATCH is not None:
			print "Use FIX_MIRCO_BATCH set by user, ", g.FIX_MIRCO_BATCH
			maxSamplePerNode = g.FIX_MIRCO_BATCH
		
	max_rank = g.MAX_RANK
	maxIdx = int(math.log(max_rank,2))
	for i in range(1,maxIdx+1):
		nodeNumber = math.pow(2,i)
		P1 = nodeNumber/P2
		miniBatch = maxSamplePerNode *P1

		if (miniBatch <= g.MAX_MINIBATCH) and (nodeNumber >= P2):
			memPerNode = 2*(float(miniBatch)*(totalIn + totalOut)/P1 + totalWeight)*g.BYTE_PER_ITEM  #(x + y)/(p1*p2) but already divided by p2 
			print nodeNumber, totalComp
			Tcomp = math.ceil(g.TOTAL_SAMPLE / P1)*totalComp
			
			#Communication inside node
			bandwidth1, latency1 = get_network_factor(platform,P2,ALGORITHM_RING)
			alpha = latency1
			beta = 1/float(bandwidth1)
			#Communication the halo exchange. Only apply for CONV and POOLING (which has kernel size > 1)
			# A GPU only need to communicate with its preceding and next GPUs along the ring except the first GPU and the last GPU (near the border of sample). Communication will be performed in 2 round in 2 directions clockwise and counter-clockwise along the ring so that no network conflict appear!!! 
			# Secondly, we group the communication of B sample at the same time to reduce the latency. 
			# The communication time can be estimated as the maximum communication of 1 GPU pair at each layer. 
			# The size of each end-2-end is (K-1)/2|W|
			Tcomm = 0
			for i in range(0,splitLayerIdx + 1):
				kernelSize = network['lays'][i]['w'][2] 
				if (kernelSize > 1):
					# Halo exchange the activation: halo(y)
					haloSize = network['lays'][i]['y'][1] * ((kernelSize -1)/2) * g.BYTE_PER_ITEM
					Thalo = 2*(alpha + haloSize*beta)
					
					# Halo exchange the input gradient: halo(dL/dx)
					haloSize = network['lays'][i]['x'][1] * ((kernelSize -1)/2) * g.BYTE_PER_ITEM
					Thalo = Thalo  + 2*(alpha + haloSize*beta)
				Tcomm = Tcomm + Thalo
			#Algather communication |y| at the begining of the 2nd partition (in the forward pass)
			#Scatter communication the input gradient  ==> factor of 2
			totalOutatSplit = float(network['lays'][splitLayerIdx]['out'])
			Tcomm = Tcomm + 2*(P2-1)*(alpha + (maxSamplePerNode*totalOutatSplit*g.BYTE_PER_ITEM*beta)/P2)

			#Communication the weight at the end of each iteration
			#Perform the reduction inside node first
			Tcomm = Tcomm  + 2*(P2-1)*(alpha + totalWeight*g.BYTE_PER_ITEM*beta/P2)
			#Perform the reducetion between node 
			#(4 flows at the same time may reduce the bandwidth 4 times)
			bandwidth2, latency2 = get_network_factor(platform,nodeNumber,ALGORITHM_RING)
			alpha = latency2
			bandwidth2 = bandwidth2 /GPU_PER_NODE
			beta = 1/float(bandwidth2)
			Tcomm = Tcomm  + 2*(P1-1)*(alpha + totalWeight*g.BYTE_PER_ITEM*beta/P1)
			
			Tcomm = Tcomm * math.ceil(g.TOTAL_SAMPLE/miniBatch)
			result = {'name':'ds','B':miniBatch,'p':nodeNumber,'mMem':memPerNode,'Tcomp':Tcomp,'Tcomm':Tcomm}
			results.append(result)
		
#################################################################
# This version is only support for ABCI and ring-based algorithms
# In this example, generate the topology like ABCI. 
# 	+ 1:3 oversubscription FBB-SPINE (4 links each...)
#	+ 1:1 oversubcription LEAR-FBB (18 up, 18 down)
#	+ ~1:1 LEAF-node (18 up, 17 down)
#	+ 2 SPINE for all. Each rack has 3 FBB, 4 LEAF, 
#	+ Each 2 nodes = 1 group. 1 group has 4 IB links that connect to 4 LEAFs
#	Down link of LEAF connect to CPU then PLX and then GPU... (assume that ignore CPU)...
# SPINE switch (defaut = 2; support up to 648/(4link * 3FBB) = 54 rack; up to 1836 nodes = 7344 GPUs).
#################################################################
# # Parameter for ABCI (https://portal.abci.ai/docs/en/01/)
# - 1088 computing node; 34 computing nodes are mounted on a rack. There are 32 racks 
GPU_SPEED = 7.8E12 # double precision flop for NVIDIA Tesla V100 for NVLINK (https://www.nvidia.com/en-us/data-center/tesla-v100/)
GPU_PER_NODE = 4
INTRA_LINK_BW = 50E9 # Bype per second #NVLINK
# End to end latency of NVLINK 1 9us per hop and 10us per 2 hop, 3.5us for on node communication
# https://arxiv.org/abs/1903.04611
# https://www.microway.com/hpc-tech-tips/nvidia-tesla-p100-nvlink-16gb-gpu-accelerator-pascal-gp100-sxm2-close/
INTRA_SWITCH_LAT = 0 # second. It should be > 0 but very small
#INTRA_CALBE_LAT = 0# 9E-6# second ~ use 1 hop end-2-end latency...
INTRA_CALBE_LAT = 9E-6# second ~ use 1 hop end-2-end latency...
PLX_LAT = 110E-9 #second. PLX Switch latency
# https://arxiv.org/abs/1903.04611
PLX_CABLE_LAT = 15E-6 # 15us GPU-to-GPU ~~> GPU-to-PLX become 0.5x but let use it.
#PLX_CABLE_LAT = 110E-9 #15E-6/2 # 15us GPU-to-GPU ~~> GPU-to-PLX become 0.5x
PLX_BW = 16E9 #Byte per second

# Internode parameter: (https://www.ssken.gr.jp/MAINSITE/event/2018/20181025-sci/lecture-03/SSKEN_sci2018_TakanoRyousei_presentation.pdf)
# 3 level (SPINE - FBB - LEAF - (nodes)
INTER_LINK_BW = 12.5E9 #25E9 #12.5E9 # Bype per second  # InfiniBand EDR (12.5 GB/s)
INTER_SWITCH_LAT_LEAF = 90E-9 #second # 90ns for InfiniBand Switch using cut through switching (http://www.mellanox.com/related-docs/prod_ib_switch_systems/pb_sb7890.pdf)
INTER_SWITCH_LAT_FBB = INTER_SWITCH_LAT_LEAF
INTER_SWITCH_LAT_SPINE = 400E-9 #second # 400ns (http://www.mellanox.com/related-docs/prod_ib_switch_systems/pb_cs7500.pdf)
CABLE_LENGTH = 0 # meter. Assume no cable length in this work
CALBE_LAT = CABLE_LENGTH * 5.2E-9 #Speed of light in glass= 190000km/sec ==> 1/19.000.000.000 = 5.2E-9 s/m

def get_network_factor(plat,rank, algorithm):
	# TODO: Real implementation for generic system. 
	#################################################################
	#SUMMARY: If 
	#1. Intra-node (RANK <= GPU_PER_NODE)  { Bandwidth = INTRA_LINK_BW, latency = INTRA_SWITCH_LAT + INTRA_CALBE_LAT}
	#2. Same-LEAF switch Inside one rack, 17 nodes = 68GPUs has connection to the same LEAF switches. 
	#{Bandwidth = min (INTRA_LINK_BW,PLX_BW, INTER_LINK_BW)   Max Latency = max(INTRA_SWITCH_LAT + INTRA_CALBE_LAT,2*(PLX_CABLE_LAT+PLX_LAT) +  INTER_SWITCH_LAT_LEAF + 2*CALBE_LAT}
	#3. Same-Rack: 34 nodes per rack = 136 GPUs. 
	# {Bandwidth = min (INTRA_LINK_BW,PLX_BW, INTER_LINK_BW)   Max Latency = max(INTRA_SWITCH_LAT + INTRA_CALBE_LAT,2*(PLX_CABLE_LAT+PLX_LAT) + INTER_SWITCH_LAT_FBB + 2*INTER_SWITCH_LAT_LEAF + 4*CALBE_LAT}
	#4. Diff-Rack: > 34 nodes per rack = 136 GPUs. 
	# {Bandwidth = min (INTRA_LINK_BW,PLX_BW, INTER_LINK_BW)   Max Latency = max(INTRA_SWITCH_LAT + INTRA_CALBE_LAT,2*(PLX_CABLE_LAT+PLX_LAT) + INTER_SWITCH_LAT_SPINE + 2*INTER_SWITCH_LAT_FBB + 2*INTER_SWITCH_LAT_LEAF + 6*CALBE_LAT}	
	if (rank <= 4):
		bandwidth = INTRA_LINK_BW
		latency  = INTRA_SWITCH_LAT + INTRA_CALBE_LAT
	elif (rank <= 68):
		bandwidth = min(INTER_LINK_BW,INTRA_LINK_BW)
		CABLE_LENGTH = 5 #5 meter for intra-rack cable length
		#CALBE_LAT = CABLE_LENGTH * 5.2E-9
		# Below factor of 2 for send and receiver node. 
		# Another factor of two is due to intra-node architecture of ABCI node. It need go through PCIe link 3 times and 1 PLX switch
		latency = max(INTRA_SWITCH_LAT + INTRA_CALBE_LAT, 2*(PLX_CABLE_LAT+PLX_LAT) +  INTER_SWITCH_LAT_LEAF + 2*CALBE_LAT)
	elif (rank <=136):
		bandwidth = min(INTER_LINK_BW,INTRA_LINK_BW)
		CABLE_LENGTH = 5 #5 meter for intra-rack cable length
		#CALBE_LAT = CABLE_LENGTH * 5.2E-9
		latency = max(INTRA_SWITCH_LAT + INTRA_CALBE_LAT, 2*(PLX_CABLE_LAT+PLX_LAT) + INTER_SWITCH_LAT_FBB + 2*INTER_SWITCH_LAT_LEAF + 4*CALBE_LAT)
	else:
		bandwidth = min(INTER_LINK_BW,INTRA_LINK_BW)
		CABLE_LENGTH = 20 #100 meter for inter-rack cable length
		#CALBE_LAT = CABLE_LENGTH * 5.2E-9
		latency = max(INTRA_SWITCH_LAT + INTRA_CALBE_LAT, 2*(PLX_CABLE_LAT+PLX_LAT) + INTER_SWITCH_LAT_SPINE + 2*INTER_SWITCH_LAT_FBB + 2*INTER_SWITCH_LAT_LEAF + 6*CALBE_LAT)
	return bandwidth,latency
	
def print_result(results):
	print "==================SUMMARY=================="
	line = "name\t\tB\tp\tMem (bytes)\tTcomp (s)\tTcomm(s)\tTime(s)\tCost(min*Node)"
	print line
	for i in range(0,len(results)):
		result = results[i]
		Tdata = result['Tcomp'] + result['Tcomm']
		cost = result['p'] * math.ceil(Tdata /60) # points = (node mins)
		line = result['name']
		line = line + "\t\t" + str(result['B'])
		line = line + "\t" + str(result['p'])
		line = line + "\t" + str(result['mMem'])
		line = line + "\t" + str(result['Tcomp'])
		line = line + "\t" + str(result['Tcomm'])
		line = line + "\t" + str(Tdata)
		line = line + "\t" + str(cost)
		print line	
		
def print_dataset(dt):
	print "===================================="
	print "DATASET: ", dt['name']
	print "DATASET SIZE: ", dt['size']
	line = "SAMPLE SIZE: "
	for i in range(0,len(dt['dim'])):
		line = line + str(dt['dim'][i])
		if i < len(dt['dim'])- 1:
			line = line + "x"
	print line, dt['sample_size']
	#print "===================================="

def print_platform(plat):
	print "===================================="
	print "FILE: ", plat['name']
	print "MEM_PER_NODE: ", plat['mem_per_node']
	print "NODE_SPEED: ", plat['node_speed']
	print "BANDWIDTH: ", plat['bandwidth']
	print "LATENCY: ", plat['latency']
	print "MAX_NODE: ", plat['max_node']
	#print "===================================="
	
def load_platform(platFileName):	
	plat = {'name':platFileName,'node_speed':0,'mem_per_node':0,'bandwidth':0,'latency':0}
	print "===================================="
	print 'Read platform from ', platFileName
	f = open(platFileName, 'r')
	lineIdx = 1
	for line in f:
		splitLine = line.replace('\r','')
		splitLine = splitLine.replace('\n','')
		splitLine = splitLine.split('|')
		if (lineIdx == 1): # MEM_PER_NODE
			if len(splitLine) >= 2:
				plat['mem_per_node'] = float(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		if (lineIdx == 2): # NODE_SPEED
			if len(splitLine) >= 2:
				plat['node_speed'] = float(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		if (lineIdx == 3): # BANDWIDTH
			if len(splitLine) >= 2:
				plat['bandwidth'] = float(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		if (lineIdx == 4): # LATENCY
			if len(splitLine) >= 2:
				plat['latency'] = float(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		if (lineIdx == 5): # MAX_NODE
			if len(splitLine) >= 2:
				plat['max_node'] = float(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		if (lineIdx == 6): # GPU_PER_NODE
			if len(splitLine) >= 2:
				plat['gpu_per_node'] = float(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		lineIdx = lineIdx + 1
	f.close()
	#print "===================================="
	return plat	
	
def load_network(networkFileName, NODE_SPEED):
	dt = {'name':"", 'size':0, 'dim':[],'sample_size':0}
	nw = {'name':"",'lays':[]}
	print "===================================="
	print 'Read dataset and DNN from ', networkFileName
	f = open(networkFileName, 'r')
	lineIdx = 1
	for line in f:
		splitLine = line.replace('\r','')
		splitLine = splitLine.replace('\n','')
		splitLine = splitLine.split('|')
		if (lineIdx == 1): #Dataset Name
			if len(splitLine) >= 2:
				dt['name'] = str(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)			
		elif (lineIdx == 2): # Dataset detail
			if len(splitLine) >= 2:
				splitDim = str(splitLine[1])
				splitDim = splitDim.split('x')
				dt['size'] = float(splitDim[0])
				dt['sample_size'] = 1 
				for i in range(1,len(splitDim)):
					dt['dim'].append(int(splitDim[i]))
					dt['sample_size'] = dt['sample_size']*int(splitDim[i])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)
		elif (lineIdx == 3): # Model Name
			if len(splitLine) >= 2:
				nw['name'] = str(splitLine[1])
			else:
				print '[WARNING] Invalid format at line ' + str(lineIdx)			
		else:
			if len(splitLine) >= 4: #Layer of DNN
				layer = {'name':"",'x':[],'w':[],'y':[],'comp':0,'in':0,'out':0,'weight':0}
				layer['name'] = str(splitLine[0])
				
				splitDim = str(splitLine[1])
				splitDim = splitDim.split('x')
				layer['in'] = 1
				for i in range(0,len(splitDim)):
					layer['x'].append(int(splitDim[i]))
					layer['in'] = layer['in'] * int(splitDim[i])
					
				splitDim = str(splitLine[2])
				splitDim = splitDim.split('x')
				layer['weight'] = 1
				for i in range(0,len(splitDim)):
					layer['w'].append(int(splitDim[i]))
					layer['weight'] = layer['weight'] * int(splitDim[i])
				
				splitDim = str(splitLine[3])
				splitDim = splitDim.split('x') 
				layer['out'] = 1
				for i in range(0,len(splitDim)):
					layer['y'].append(int(splitDim[i]))
					layer['out'] = layer['out'] * int(splitDim[i])
				if len(splitLine) >= 5:
					layer['comp'] = float(splitLine[4])
				else:
					#FlopCount of FW
					if ("CONV" in layer['name']): # 2 * Wy * Hy * C * F * K * K
						layer['comp'] = layer['w'][0] * layer['w'][1]
						for i in range(2,len( layer['w'])):
							layer['comp'] = layer['comp'] * layer['w'][i] * layer['y'][i-1]
						#Factor of 2: 1 for add, 1 for mul
						layer['comp'] = 2 * layer['comp']
					elif ("MPOOL" in layer['name']): # Wy * Hy * C * F * K * K 
						layer['comp'] = layer['x'][0] * layer['y'][0]  #C *F
						for i in range(2,len( layer['w'])):
							layer['comp'] = layer['comp'] * layer['w'][i] * layer['y'][i-1]
					elif ("APOOL" in layer['name']): # Wy * Hy * C * F * K * K 
						layer['comp'] = layer['x'][0] * layer['y'][0]  #C *F
						for i in range(2,len( layer['w'])):
							layer['comp'] = layer['comp'] * layer['w'][i] * layer['y'][i-1]
					elif ("RELU" in layer['name']): # element wise: Wy * Hy * F
						layer['comp'] = 1
						for i in range(0,len( layer['y'])):
							layer['comp'] = layer['comp'] * layer['y'][i]	
					elif ("FC" in layer['name']): # C * F * K * K
						layer['comp'] = 1
						for i in range(0,len( layer['w'])):
							layer['comp'] = layer['comp'] * layer['w'][i]
					elif ("BNORM" in layer['name']): # 4* Wx * Hx * C  (this calculation per item, will multiple with Batch later)
						layer['comp'] = layer['x'][0]
						for i in range(1,len( layer['x'])):
							layer['comp'] = layer['comp'] * layer['x'][i]
						# Factor of 4 are: 1 for mini-batch mean, 1 formini-batch variance, 1 for normalize 1 for scale and shift
						layer['comp'] = 4 * layer['comp']
					elif ("ADD" in layer['name']): # Wy * Hy * F  (this calculation per item, will multiple with Batch later)
						layer['comp'] = layer['y'][0]
						for i in range(1,len( layer['y'])):
							layer['comp'] = layer['comp'] * layer['y'][i]
					#elif ("DROPOUT" in layer['name']): #
					else:
						layer['comp'] = 0 #Default
					# FlopCount of BW to calculate x, y gradient
					layer['comp'] = layer['comp'] * 2 
					# FlopCount of weight gradient
					weightGradientFlop = 1
					for i in range(0,len( layer['w'])):
						weightGradientFlop = weightGradientFlop * layer['w'][i]
					layer['comp'] = layer['comp'] + weightGradientFlop
					# Comp in total 
					layer['comp'] = float(layer['comp'])/NODE_SPEED
				nw['lays'].append(layer)
				
			else:
				if splitLine[0][0] == "#":
					# comment line
					print splitLine[0]
				else:
					print '[WARNING] Invalid format at line ' + str(lineIdx)
					print line
		lineIdx = lineIdx + 1
	f.close()
	#print "===================================="
	return dt, nw	

def summary_network(nw):
	totalWeight = 0;
	totalComp = 0;
	totalIn = 0;
	totalOut = 0;
	maxOut = 0;
	for i in range(0,len(nw['lays'])):
		layer = nw['lays'][i]
		if maxOut < layer['out']:
			maxOut = layer['out']
		totalOut = totalOut + layer['out']
		totalIn = totalIn + layer['in']
		totalWeight = totalWeight + layer['weight']
		totalComp = totalComp + layer['comp']
	print "Total |x|: " + str(totalIn) + " items"
	print "Total |y|: " + str(totalOut) + " items"
	print "Total |w|: " + str(totalWeight) + " items"
	print "Total comp: " + str(totalComp) + " items"
	print "Max |y|: " + str(maxOut) + " sec"
	return totalIn,totalWeight,totalOut,totalComp,maxOut

def summary_network2(nw, debug):
	totalWeight = 0;
	totalComp = 0;
	totalIn = 0;
	totalOut = 0;
	maxOut = 0;
	maxComp = 0;
	for i in range(0,len(nw['lays'])):
		layer = nw['lays'][i]
		if maxOut < layer['out']:
			maxOut = layer['out']
		if maxComp < layer['comp']:
			maxComp = layer['comp']
		if 	("RELU" in layer['name']) or ("BNORM" in layer['name']):
			totalOut = totalOut + 0
			totalIn = totalIn + 0
		else:
			totalOut = totalOut + layer['out']
			totalIn = totalIn + layer['in']
		totalWeight = totalWeight + layer['weight']
		totalComp = totalComp + layer['comp']
	if debug == True:
		print "Model with", str(len(nw['lays'])), "layers"
		print "Total |x|: " + str(totalIn) + " items"
		print "Total |y|: " + str(totalOut) + " items"
		print "Total |w|: " + str(totalWeight) + " items"
		print "Total comp: " + str(totalComp) + " sec" 
		print "Max comp: " + str(maxComp) + " sec ==>" + str(1/totalComp) + "samples for 100% GPU ultilization"
		print "Max |y|: " + str(maxOut) + " items"
	metaData = {"totalIn":totalIn,"totalOut":totalOut,"totalWeight":totalWeight,"totalComp":totalComp,"maxOut":maxOut}
	return metaData
	
def print_network_all(nw):
	print "===================================="
	print "Model: ", nw['name'] 
	print "LAYER\t x[C,W,H] \t w[C,F,K,K] \t y[F,W,H] \t TComp"
	for i in range(0,len(nw['lays'])):
		layer = nw['lays'][i]
		print i, layer['name'],"\t",layer['x'],"=",layer['in'],"\t",layer['w'],"=",layer['weight'],"\t",layer['y'],"=",layer['out'],"\t",layer['comp']	
	
def print_network(nw):
	print "===================================="
	print "Model: ", nw['name'] 
	print "LAYER\t x[C,W,H] \t w[C,F,K,K] \t y[F,W,H] \t TComp"
	for i in range(0,len(nw['lays'])):
		layer = nw['lays'][i]
		print layer['name'],"\t",layer['x'],"\t",layer['w'],"\t",layer['y'],"\t",layer['comp']
	
if __name__ == '__main__':
	main(argumentparser.parse_args())