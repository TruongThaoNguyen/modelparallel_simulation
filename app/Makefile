vgg: vgg16_o vgg16_d vgg16_vmodel vgg16_hmodel vgg16_imodel

vgg16_o: VGG16.c
	/home/nguyen_truong/optelectric_simulate/SimGrid-3.21/build/bin/smpicc VGG16.c -o VGG16.run

vgg16_d: VGG16_data.c
	/home/nguyen_truong/optelectric_simulate/SimGrid-3.21/build/bin/smpicc VGG16_data.c -o VGG16_d.run
		
clean:
	/bin/rm -f *.run
	/bin/rm -f *.log*
	/bin/rm -f smpitmp*

