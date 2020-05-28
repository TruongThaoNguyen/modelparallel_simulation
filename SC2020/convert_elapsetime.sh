#DATA_DIR="./test/no_comm/"
#DATA_DIR="../../paraDL/data/no_comm_results/"
#DATA_DIR="../../paraDL/hybrid_spatial/ranking_results/"
DATA_DIR="../../paraDL/hybrid_filter/ranking_results/"
#DATA_DIR="../../paraDL/filter/ranking_results/"
#DATA_DIR="../../paraDL/spatial/no_comm_results/"


# DATA_DIR="../../paraDL/cosmoflow/revision_results/"
# for NODE in 4 8 16 32 64 128 256 512
# do
	# FILENAME="${DATA_DIR}/${NODE}/2020*.log"
	# OUTFILE="${DATA_DIR}/${NODE}/elapsed_time.out"
	# #rm ${FILENAME}.out
	# grep '\"elapsed_time\":' ${FILENAME} |  sed  's/ \"elapsed_time\"://g' >> ${OUTFILE}
	
	# python convert_log.py -infile "${DATA_DIR}/${NODE}/forward_halo_exchange_time.log" -maxline 2
	# python convert_log.py -infile "${DATA_DIR}/${NODE}/backward_halo_exchange_time_file.log" -maxline 2
# done

for MODEL in "vgg" #"resnet" #"vgg" #"resnet"
do
	for NODE in 16 64 #8 16 32 64 #128 256 512 # #8 #16 32 64 128 #8 16 #32 64
	do
		FILENAME="${DATA_DIR}/${MODEL}/${NODE}/2020*.log"
		OUTFILE="${DATA_DIR}/${MODEL}/${NODE}/elapsed_time.out"
		rm ${FILENAME}.out
		grep '\"elapsed_time\":' ${FILENAME} |  sed  's/ \"elapsed_time\"://g' >> ${OUTFILE}
	done
done

# for MODEL in "vgg"
# do
	# for NODE in 8 #16 32 64 128 #8 16 #32 64
	# do
		# python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/pooling_halo_exchange_time.log" -maxline 5
		# python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log" -maxline 13
		# python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log" -maxline 13
	# done
# done

# for MODEL in "resnet"
# do
	# for NODE in 8 16 32 64 128 256 512 #256 #32 64 128 #8 16 #32 64
	# do
		# python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/forward_halo_exchange_time.log" -maxline 17
		# python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/backward_halo_exchange_time_file.log" -maxline 53
	# done
# done

for MODEL in "vgg" #"resnet"
do
	for NODE in 16 64 #128 256 512 #64 #32 64  #8 16 32 64 128 256 512
	do
		python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/high_level_allreduce_times.log" -maxline 13 #53
		python convert_log.py -infile "${DATA_DIR}/${MODEL}/${NODE}/high_level_allgather_times.log" -maxline 13 #53
	done
done
