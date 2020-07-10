# mutipefeature-for-PRID
the code is the mutiplefeature-for-PRID

### Prerequisites
* Pytorch(0.4.0+)
* cuda 9.0
* python3.6
* 1GPUs>24G
### Dataset
To use our code, firstly you should download ReID dataset (Market1501,DukeMTMC-reID,CUHK03-NP and MSMT17) from [Here](https://pan.baidu.com/s/1G_Ygn68UolKhmiu1eGliLg)(saqs).

Then, the dataset folder should be as below(such as CUHK03):

"detected" means the bounding boxes are estimated by pedestrian detector

"labeled" means the bounding boxes are labeled by human
```
CUHK03
│ 
└───Labelde
│   │
│   │
│   └───bounding_box_test
│   │   │   0003_c1_21.jpg
│   │   │   0003_c1_23.jpg
│   │   │   ...
│   │
│   └───bounding_box_train
│   │   │   0001_c1_1.png
│   │   │   0001_c1_2.png
│   │   │   ...
│   └───query
│   │   │   0003_c1_22.png
│   │   │   0003_c2_27.png
│   │   │   ...
│
└───detected
│   │  
│   │
│   └───bounding_box_test
│   │   │   0003_c1_21.jpg
│   │   │   0003_c1_23.jpg
│   │   │   ...
│   │
│   └───bounding_box_train
│   │   │   0001_c1_1.png
│   │   │   0001_c1_2.png
│   │   │   ...
│   └───query
│   │   │   0003_c1_22.png
│   │   │   0003_c2_27.png
│   │   │   ...
```
