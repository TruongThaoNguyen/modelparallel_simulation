#SIMGRID="/home/nguyen_truong/optelectric_simulate/SimGrid-3.21/build/bin/smpirun"
SIMGRID="/home/nguyen/ai/optelectric_simulate/SimGrid-3.21/build/bin/smpirun"
PLATFORM="../platforms/Tsubame3_64.xml"
HOSTFILE="../platforms/Tsubame3_64.txt"
LOG_DIR="./"
CONFIG="--cfg=exception/cutpath:1 --cfg=smpi/display-timing:1 --cfg=smpi/allreduce:lr"

# #Single node 
# make vgg16_o
# SIZE=1
# LOG_FILE="${LOG_DIR}/vgg16_o_${SIZE}.log"
# ${SIMGRID} -np ${SIZE} -map -platform ${PLATFORM} -hostfile ${HOSTFILE} ${CONFIG} VGG16.run >> ${LOG_FILE} 2>&1 &

# #Data parallelism
# make vgg16_d
# for SIZE in 4 8 16 32 64 128 256
# do
	# LOG_FILE="${LOG_DIR}/vgg16_d_${SIZE}.log"
	# ${SIMGRID} -np ${SIZE} -map -platform ${PLATFORM} -hostfile ${HOSTFILE} ${CONFIG} VGG16_d.run >> ${LOG_FILE} 2>&1 &
# done

hModel parallelism
make vgg16_h
for SIZE in 2 4 8 16 32 
do
	LOG_FILE="${LOG_DIR}/vgg16_h_${SIZE}.log"
	${SIMGRID} -np ${SIZE} -map -platform ${PLATFORM} -hostfile ${HOSTFILE} ${CONFIG} VGG16_h.run >> ${LOG_FILE} 2>&1 &
done

# #vModel parallelism
# make vgg16_v
# for SIZE in 4 8
# do
	# LOG_FILE="${LOG_DIR}/vgg16_v_${SIZE}.log"
	# ${SIMGRID} -np ${SIZE} -map -platform ${PLATFORM} -hostfile ${HOSTFILE} ${CONFIG} VGG16_v.run >> ${LOG_FILE} 2>&1 &
# done

#grep "Simulated time" *.log