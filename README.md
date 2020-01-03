# code package

the structure uses the pytorch template code

## Usage

all code started with train.py, we use config file to differentiate the model we used.

just like: `python train -c model_config.json`

for `Factorization Machine` test you can just run `python train.py -c ./configs/config_fm.json`

## Models

| Model | Reference | 
| ------ | ------ | 
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | 
| Field-aware Factorization Machine | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |

## Reference 
rec based on：[pytorch-fm](https://github.com/rixwew/pytorch-fm.git)