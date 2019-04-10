#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <smpi/smpi.h>

int main(int argc, char *argv[])
{
	int size, rank;
	struct timeval start, end;
	char hostname[256];
	int hostname_len;

	gettimeofday(&start,NULL);
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(hostname,&hostname_len);

	/***********        Parameter    ************/
	int B = 256; //Minibatch size.
	int S = 2; //Number of Segment in pipeline mode.
	int Pr = 2; //Number of row 
	int Pc = size/Pr; // Number of column
	int Si = 2; // Number of segment in pipieline mode for iModel.
	int I =  2048/B; //1280000/B; //Number of Iteration per epoch
	int E = 1; //Number of epoch;
	/******* Neural Model Declaration ***********/
	/* VGG16 Deep Neural Network for the sample (image) size 244x244x3 
	Layer		Activation |Yi |	Weight |WTi |		#Operations (FW)
	INPUT 		244x244x3 			0 					0
	CONV3-64 	224x224x64 			3x3x3x64 			224x224x64x3x3x3
	CONV3-64 	224x224x64 			3x3x64x64 			224x224x64x3x3x64
	POOL2 		112x112x64 			0 					112x112x64x2x2x64
	CONV3-128 	112x112x128 		3x3x64x128 			112x112x128x3x3x64
	CONV3-128 	112x112x128 		3x3x128x128			112x112x128x3x3x128
	POOL2 		56x56x128 			0					56x56x128x2x2x128
	CONV3-256	56x56x256 			3x3x128x256			56x56x256x3x3x128
	CONV3-256 	56x56x256 			3x3x256x256			56x56x256x3x3x256
	CONV3-256 	56x56x256 			3x3x256x256			56x56x256x3x3x256
	POOL2 		28x28x256 			0					28x28x256x2x2x256
	CONV3-512 	28x28x512 			3x3x256x512			28x28x512x3x3x256
	CONV3-512 	28x28x512 			3x3x512x512			28x28x512x3x3x512
	CONV3-512 	28x28x512 			3x3x512x512			28x28x512x3x3x512
	POOL2 		14x14x512 			0					14x14x512x2x2x512
	CONV3-512 	14x14x512 			3x3x512x512			14x14x512x3x3x512
	CONV3-512 	14x14x512 			3x3x512x512			14x14x512x3x3x512
	CONV3-512 	14x14x512 			3x3x512x512			14x14x512x3x3x512
	POOL2 		7x7x512				0					7x7x512x2x2x512
	FC4096 		1x1x4096 			7x7x512x4096		7x7x512x4096
	FC4096 		1x1x4096 			4096x4096			4096x4096
	FC1000 		1x1x1000 			4096x1000			4096x1000
	Total		15.2M				138.3M				16.3B
	*/  
	
	int i = 0;
	double input = 244*244*3;
	double nw[21][3] = {
		{224*224*64 ,3*3*3*64 ,224*224*64*3*3*3},
		{224*224*64 ,3*3*64*64 ,224*224*64*3*3*64},
		{112*112*64 ,0 ,112*112*64*2*2*64},
		{112*112*128 ,3*3*64*128 ,112*112*128*3*3*64},
		{112*112*128 ,3*3*128*128,112*112*128*3*3*128},
		{56*56*128 ,0,56*56*128*2*2*128},
		{56*56*256 ,3*3*128*256,56*56*256*3*3*128},
		{56*56*256 ,3*3*256*256,56*56*256*3*3*256},
		{56*56*256 ,3*3*256*256,56*56*256*3*3*256},
		{28*28*256 ,0,28*28*256*2*2*256},
		{28*28*512 ,3*3*256*512,28*28*512*3*3*256},
		{28*28*512 ,3*3*512*512,28*28*512*3*3*512},
		{28*28*512 ,3*3*512*512,28*28*512*3*3*512},
		{14*14*512 ,0,14*14*512*2*2*512},
		{14*14*512 ,3*3*512*512,14*14*512*3*3*512},
		{14*14*512 ,3*3*512*512,14*14*512*3*3*512},
		{14*14*512 ,3*3*512*512,14*14*512*3*3*512},
		{7*7*512,0,7*7*512*2*2*512},
		{1*1*4096 ,7*7*512*4096,7*7*512*4096},
		{1*1*4096 ,4096*4096,4096*4096},
		{1*1*1000 ,4096*1000,4096*1000}
	};
	int L = 21; // Number of Layer
	double totalWeight = 0;
	double totalComp = 0;
	double maxActivation = 0;
	for (i = 0; i < L; i++){
		if (maxActivation < nw[i][0]){
			maxActivation = nw[i][0];
		}
		totalWeight+= nw[i][1];
		totalComp+= nw[i][2];
		if (rank == 0) {
			printf("Layer %d [%f,%f,%.1f]\n",i,nw[i][0],nw[i][1],nw[i][2]);
		}
	}
	if (rank == 0) {
		printf("Total weight %f bytes\n", totalWeight*4);
		printf("Max activation %f bytes\n", maxActivation*4);
		printf("Total computation %f flop\n", totalComp);
	}
	/******* Model partition ***********/
	double approxComp = totalComp/size;
	int firstLayerIdx[size];
	int lastLayerIdx[size];
	firstLayerIdx[0] = 0;
	lastLayerIdx[size-1] = L-1;
	i = 0;
	int j =0;
	double currentComp = 0;
	double lastComp = currentComp;
	while (j < L && i < size - 1){
		currentComp += nw[j][2];
		//if (rank == 0) {printf("approxComp %f flop\n", approxComp);}
		if (currentComp > approxComp){
			if (currentComp - approxComp > approxComp - lastComp){ //Choose the point near the approxComp: layer j-1
				lastLayerIdx[i] = j - 1;
				if (rank == 0) {printf("Node %d with layer: %d - %d (Total computation: %f)\n", i,firstLayerIdx[i],lastLayerIdx[i], lastComp);}
				firstLayerIdx[i+1] = j;
				totalComp = totalComp - lastComp;
				currentComp = nw[j][2];
				
			}
			else {//Chose layer j as the last layer
				lastLayerIdx[i] = j;
				if (rank == 0) {printf("Node %d keep layer: %d - %d (Total computation: %f)\n", i,firstLayerIdx[i],lastLayerIdx[i], currentComp);}
				firstLayerIdx[i+1] = j + 1;
				totalComp = totalComp - currentComp;
				currentComp = 0;
			}
			i++;
			//approxComp = totalComp / (size - i);
		}
		lastComp = currentComp;
		j++;
	}
	if (rank == 0) {printf("Node %d keep layer: %d - %d (Total computation: %f)\n", i,firstLayerIdx[i],lastLayerIdx[i], totalComp);}
	
	int mB = B/S; //microBatch size	
	double *local_act = malloc(sizeof(double) * maxActivation*mB);
	double *global_act = malloc(sizeof(double) * maxActivation*mB);
	MPI_Barrier(MPI_COMM_WORLD);
	
	/****** Training ***********/
 	gettimeofday(&end,NULL);
	if (rank == 0) {
		printf("Start training\t%f\n",(end.tv_sec*1000000.0 + end.tv_usec -
			start.tv_sec*1000000.0 - start.tv_usec) / 1000000.0);  
	}
		
	int iter, seg;
	for (iter = 0; iter < I*E; iter++){
		// FW phase
		seg = 0;
		for (seg = 0; seg < S; seg++){
			if (rank > 0){
				// Wait for receiving information from previous node
				//printf("[Node %d] recv %f items from %d with tag %d\n", rank, mB*nw[lastLayerIdx[rank-1]][0], rank-1, (rank-1)*10000+seg);
				MPI_Recv(global_act, mB*nw[lastLayerIdx[rank-1]][0], MPI_DOUBLE_PRECISION, rank-1, (rank-1)*10000+seg, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			// FW computation
			gettimeofday(&end,NULL);
			printf("[Node %d] Start FW\t%f\n",rank,(end.tv_sec*1000000.0 + end.tv_usec -
				start.tv_sec*1000000.0 - start.tv_usec) / 1000000.0);  
			for (i = firstLayerIdx[rank]; i <= lastLayerIdx[rank]; i++){
				SMPI_SAMPLE_FLOPS(nw[i][2]*mB) {}
			}
			gettimeofday(&end,NULL);
			printf("[Node %d] End FW\t%f\n",rank, (end.tv_sec*1000000.0 + end.tv_usec -
				start.tv_sec*1000000.0 - start.tv_usec) / 1000000.0); 
		
			//Communication in FW phase.
			if (rank < size -1){
				//printf("[Node %d] send %f items to %d with tag %d\n", rank, mB*nw[lastLayerIdx[rank]][0], rank+1, rank*10000+seg);
				MPI_Send(local_act, mB*nw[lastLayerIdx[rank]][0], MPI_DOUBLE_PRECISION, rank + 1, rank*10000+seg, MPI_COMM_WORLD);
			}

			//MPI_Barrier(MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		//BW phase
		for (seg = S-1; seg >= 0; seg--){
			if (rank < size -1 ){
				// Wait for receiving information from previous node
				//printf("[Node %d] recv %f items from %d with tag %d\n", rank, mB*nw[lastLayerIdx[rank]][0], rank+1, (rank+1)*10000+seg);
				MPI_Recv(global_act, mB*nw[lastLayerIdx[rank]][0], MPI_DOUBLE_PRECISION, rank+1, (rank+1)*10000+seg, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			// BW computation
			gettimeofday(&end,NULL);
			printf("[Node %d] Start BW\t%f\n",rank,(end.tv_sec*1000000.0 + end.tv_usec -
				start.tv_sec*1000000.0 - start.tv_usec) / 1000000.0);  
			for (i = lastLayerIdx[rank]; i>=firstLayerIdx[rank] ; i--){
				SMPI_SAMPLE_FLOPS(nw[i][2]*mB) {}
				//Gradient calculation
				SMPI_SAMPLE_FLOPS(nw[i][1]) {}
			}
			gettimeofday(&end,NULL);
			printf("[Node %d] End BW\t%f\n",rank, (end.tv_sec*1000000.0 + end.tv_usec -
				start.tv_sec*1000000.0 - start.tv_usec) / 1000000.0); 
		
			//Communication in BW phase.
			if (rank > 0){
				//printf("[Node %d] send %f items to %d with tag %d\n", rank, mB*nw[lastLayerIdx[rank-1]][0], rank-1, rank*10000+seg);
				MPI_Send(local_act, mB*nw[lastLayerIdx[rank-1]][0], MPI_DOUBLE_PRECISION, rank - 1, rank*10000+seg, MPI_COMM_WORLD);
			}

			//MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		gettimeofday(&end,NULL);
		printf("End training\t %f\n",(end.tv_sec*1000000.0 + end.tv_usec -
			start.tv_sec*1000000.0 - start.tv_usec) / 1000000.0);  
	}
	MPI_Finalize();
	return 0;
}
