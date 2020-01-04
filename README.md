# Basic DeepLearning Models

I build it for study deep learning model with pytorch

## Usage

All code started with train.py, we use config file to differentiate the model we used.

Just like: `python train -c model_config.json`

For `Factorization Machine` , run 
```
python train.py -c ./configs/config_fm.json
```

For each task of your owner, you should build dataloader in `./data_loader` and config the json file in `./configs` 

## Models

| Model | Reference | 
| ------ | ------ | 
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | 
| Field-aware Factorization Machine | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| DeepFM|[H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247)|
| Wide&Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Deep Cross Network | [R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.](https://arxiv.org/abs/1708.05123) |

## Reference 

Pytorch template based on: [pytorch-template](https://github.com/victoresque/pytorch-template.git)
Rec based on：[pytorch-fm](https://github.com/rixwew/pytorch-fm.git)