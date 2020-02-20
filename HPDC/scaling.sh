#python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype a --debug y >> resnet_scaling_peak.log

#python parallel_analysis.py -net ./RESNET50_ImageNet_profall.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype o,d --debug y >> data_resnet_scaling_proj.log

echo "===============Below is for data + spatial===============" >> data_resnet_scaling_proj.log
python parallel_analysis.py -net ./RESNET50_ImageNet_profblock.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=64 --paratype ds --debug y >> data_resnet_scaling_proj.log

python parallel_analysis.py -net ./RESNET50_ImageNet_profblock.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=256 --paratype ds --debug y >> data_resnet_scaling_proj.log

# python parallel_analysis.py -net ./ALEXNET_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 524288 --cBon=512 --paratype a --debug y >> alexnet_scaling_peak.log

# python parallel_analysis.py -net ./ALEXNET_ImageNet_profall.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 524288 --cBon=512 --paratype o,d --debug y >> alexnet_scaling_proj.log

python parallel_analysis.py -net ./VGG_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=32 --paratype a --debug y >> vgg_scaling_peak.log

python parallel_analysis.py -net ./VGG_ImageNet_profall.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=32 --paratype o,d --debug y >> vgg_scaling_proj.log


##FOR RANKING
python parallel_analysis.py -net ./VGG_ImageNet_profall.net  -plat ABCI.plat -goal 1 --cmaxp 2048 --cmaxB 262144 --cBon=32 --paratype a --debug y >> vgg_ranking.log
