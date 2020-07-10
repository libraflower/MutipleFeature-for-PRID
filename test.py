from __future__ import print_function, division
import torch
import argparse
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import scipy.io
import math
from MultipleFeatureswithoutattention import MultipleFeatures


parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='duke', type=str, help='output model name')
parser.add_argument('--test_dir',default='/Project0551/guoqing/scscnet/cuhk03-np/labeled/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--epochnum', default='last', type=str, help='please to select the epoch num')
parser.add_argument('--testing', action='store_true', help='import testing features')
parser.add_argument('--attentionmodel', action='store_true', help='use the attention model')
opt = parser.parse_args()

gpu_ids = opt.gpu_ids
ms = '1'
str_ids = gpu_ids.split(',')
test_dir = opt.test_dir
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
print('We use the scale: %s'%ms)
str_ms = ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
######################################################################
# Load Data
data_transforms = transforms.Compose([
        transforms.Resize((384,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= opt.batchsize,
                                             shuffle=False, num_workers=8) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()
######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',opt.name,'net_%s.pth'%opt.epochnum)
    print(save_path)
    network.load_state_dict(torch.load(save_path))
    return network
#####################################################################
# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip
##########定义特征提取模型
def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,10752).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs= model(input_img)
            feature = outputs[0]
            for output in outputs[1:]:
                feature = torch.cat((feature,output),dim=1)
            feature = feature.data.cpu()
            ff = ff + feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

######################################################################
# Load Collected data Trained model
if __name__ == '__main__':
    print('-------test-----------')
    model_structure = MultipleFeatures(class_num=767 ,testing=opt.testing,attentionusing=opt.attentionmodel)
    input = Variable(torch.FloatTensor(8, 3, 384, 128))
    output = model_structure(input)
    for i in output:
        print(i.shape)
    model = load_network(model_structure)
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    # Extract feature
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])
    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('pytorch_result.mat',result)
