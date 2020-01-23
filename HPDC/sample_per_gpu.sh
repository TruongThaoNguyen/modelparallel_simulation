# echo " ############################ ALEXNET #################################" >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=32 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=64 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=128 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=256 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=512 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=1024 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./ALEXNET_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=2048 --paratype d >> sample_per_gpu.log

# echo " ############################ VGG #################################" >> sample_per_gpu.log
# python parallel_analysis.py -net ./VGG_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=8 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./VGG_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=16 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./VGG_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=32 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./VGG_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=64 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./VGG_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=128 --paratype d >> sample_per_gpu.log
# python parallel_analysis.py -net ./VGG_ImageNet_prof.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=256 --paratype d >> sample_per_gpu.log

echo " ############################ RESNET #################################" >> sample_per_gpu.log
python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=8 --paratype d >> sample_per_gpu.log
python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=16 --paratype d >> sample_per_gpu.log
python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=32 --paratype d >> sample_per_gpu.log
python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=64 --paratype d >> sample_per_gpu.log
python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=128 --paratype d >> sample_per_gpu.log
python parallel_analysis.py -net ./RESNET50_ImageNet.net  -plat ABCI.plat -goal 1 --cmaxp 4 --cmaxB 262144 --cBon=256 --paratype d >> sample_per_gpu.log

