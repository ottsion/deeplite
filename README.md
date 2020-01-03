# Basic DeepLearning Models


The structure uses the pytorch template code

## Usage

All code started with train.py, we use config file to differentiate the model we used.

Just like: `python train -c model_config.json`

For `Factorization Machine` 

run `python train.py -c ./configs/config_fm.json`

## Models

| Model | Reference | 
| ------ | ------ | 
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | 
| Field-aware Factorization Machine | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| DeepFM|[H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247)|

## Reference 

Rec based on：[pytorch-fm](https://github.com/rixwew/pytorch-fm.git)