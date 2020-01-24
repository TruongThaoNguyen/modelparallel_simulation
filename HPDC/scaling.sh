#python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype a --debug y >> resnet_scaling_peak.log

python parallel_analysis.py -net ./RESNET50_ImageNet_profall.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype o,d --debug y >> data_resnet_scaling_proj.log


#python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net1  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=512 --paratype a --debug y >> alexnet_scaling_proj.log

#python parallel_analysis.py -net ./VGG_ImageNet_prof.net1  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=32 --paratype a --debug y >> vgg_scaling_proj.log
