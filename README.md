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
Sometimes you will get tensor type error between long\float\int, all you need is to change your `dataset` file `__getitem__`

## CTR Models

| Model | Reference | 
| ------ | ------ | 
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | 
| Field-aware Factorization Machine | [Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) |
| DeepFM|[H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247)|
| Wide&Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Deep Cross Network | [R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.](https://arxiv.org/abs/1708.05123) |
| xDeepFM | [J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.](https://arxiv.org/abs/1803.05170) |


## NLP Models

| Model | Reference | 
| ------ | ------ | 
| fastText | [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)|
|TextCNN | [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

## DataSet

|ModelType|DataSet|Source|
|---|---|---|
|CTR Prediction|CriteoDataset|[criteo](http://research.criteo.com/outreach/)|
|NLP Classify|ThucnewsDataset|[THUCNews](http://thuctc.thunlp.org/)|


## Performance

| Model | acc | loss |
| ------ | ------ | ------ | 
|FM|0.8535|0.681576|
|FastText|0.998|0.02|

## Reference 

Pytorch template based on: [pytorch-template](https://github.com/victoresque/pytorch-template.git)

Rec based on：[pytorch-fm](https://github.com/rixwew/pytorch-fm.git)