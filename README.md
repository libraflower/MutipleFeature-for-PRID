# mutipefeature-for-PRID
the code is the mutiplefeature-for-PRID

### Prerequisites
* Pytorch(0.4.0+)
* cuda 9.0
* python3.6
* 1GPUs,memory>24G
### Dataset
To use our code, firstly you should download ReID dataset (Market1501,DukeMTMC-reID,CUHK03-NP and MSMT17) from [Here](https://pan.baidu.com/s/1G_Ygn68UolKhmiu1eGliLg)(saqs).

Then, the dataset folder should be as below(such as CUHK03):

"detected" means the bounding boxes are estimated by pedestrian detector

"labeled" means the bounding boxes are labeled by human
```
CUHK03_np
│ 
└───Labelde
│   └───bounding_box_test
│   │   │   0003_c1_21.jpg
│   │   │   0003_c1_23.jpg
│   │   │   ...
│   └───bounding_box_train
│   │   │   0001_c1_1.png
│   │   │   0001_c1_2.png
│   │   │   ...
│   └───query
│   │   │   0003_c1_22.png
│   │   │   0003_c2_27.png
│   │   │   ...
└───detected
│   └───bounding_box_test
│   │   │   0003_c1_21.jpg
│   │   │   0003_c1_23.jpg
│   │   │   ...
│   └───bounding_box_train
│   │   │   0001_c1_1.png
│   │   │   0001_c1_2.png
│   │   │   ...
│   └───query
│   │   │   0003_c1_22.png
│   │   │   0003_c2_27.png
│   │   │   ...
```

### Train
first ,you must run the perpare.py to get 'pytorch' folder
In our train.py,we give you some options,as follow:
```
parser = argparse.ArgumentParser(description='Training')
#you can choose the gpu to run the trainmodel.
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
#the model name you want to save.
parser.add_argument('--name',default='...', type=str, help='output model name')
#the dataset direction.
parser.add_argument('--data_dir',default='.../cuhk03-np/labeled/pytorch',type=str, help='training dir path')
#the batchsize you choose in train process,we recommend 64,and you also choose 128 batchsize.
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
#REA p=0.5,you can use other number [0,1].
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
#warm up epoch.
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')
#all epoch you should use in the train process.
parser.add_argument('--epochnum', default=150, type=int, help='please to select the epoch num')
#LR.
parser.add_argument('--base_lr', default=0.01, type=float, help='the base_learning rate')
#triplet loss margin.
parser.add_argument('--tripletmargin', default=1.0, type=float, help='the tripletmargin')
#warm up LR.
parser.add_argument('--warmup_begin_lr', default=3.5e-5, type=float, help='warmup learning rate')
#LR decray.
parser.add_argument('--factor', default=0.1, type=float, help='the learning rate decracy')
#using the color jitter.
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
#using the attention models.
parser.add_argument('--attentionmodel', action='store_true', help='use the attention model')
#change to the test process.
parser.add_argument('--testing', action='store_true', help='import testing features')
```

### Usage
```
python3 train.py --gpu_ids .. --name .. --data_dir ../cuhk03-np/labeled/pytorch --batchsize 64 --erasing_p 0.5 --warm_epoch 10 --epochnum 150 --base_lr 0.01 --tripletmargin 1.0 --warmup_begin_lr 3e-4 --factor 0.5  --attentionmodel

python3 train.py --gpu_ids .. --name .. --data_dir ../cuhk03-np/detected/pytorch --batchsize 64 --erasing_p 0.5 --warm_epoch 10 --epochnum 150 --base_lr 0.01 --tripletmargin 1.0 --warmup_begin_lr 3e-4 --factor 0.5  --attentionmodel
```

### Test
```
parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='duke', type=str, help='output model name')
parser.add_argument('--test_dir',default='.../cuhk03-np/labeled/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--epochnum', default='last', type=str, help='please to select the epoch num')
parser.add_argument('--testing', action='store_true', help='import testing features')
parser.add_argument('--attentionmodel', action='store_true', help='use the attention model')
opt = parser.parse_args()
```

```
python3 test.py --gpu_ids ... --data_dir ../cuhk03-np/labeled/pytorch --name ... --batchsize 64 --epochnum last --train_all --attentionmodel --testing
python3 test.py --gpu_ids ... --data_dir ../cuhk03-np/detected/pytorch  --name ... --batchsize 64 --epochnum last --train_all --attentionmodel --testing
```

## Evaluation

| dataset | rank-1 | rank-5 | rank-10 |mAP|
| :------: | :------: | :------: | :------: | :------: |
| CUHK_Detected| 0.775000 | 0.894286 | 0.932857 |0.724668|
| CUHK_Labeled | 0.802857 | 0.912143 | 0.954286 |0.761700|
| DukeMTMC-reID| 0.901729 | 0.945242 | 0.960952 |0.801553|
