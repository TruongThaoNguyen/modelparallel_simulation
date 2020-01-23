#python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype a --debug y >> resnet_scaling_peak.log


python parallel_analysis.py -net ./RESNET50_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype a --debug y >> resnet_scaling_proj.log
