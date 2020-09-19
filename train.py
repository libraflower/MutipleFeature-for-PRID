from __future__ import print_function, division	
import torch	
import torch.optim as optim	
from torchvision import datasets, transforms	
import matplotlib	
matplotlib.use('agg')	
import time	
import os	
import argparse	
#you can change here	
from MultipleFeatureswithoutattention import MultipleFeatures	
import matplotlib.pyplot as plt	
from random_erasing import RandomErasing	
from samplers import RandomIdentitySampler	
from triplet_loss import TripletLoss, CrossEntropyLabelSmooth	
from lr_scheduler import LRScheduler	
from shutil import copyfile	
from torch.autograd import Variable	
version =  torch.__version__	
torch.manual_seed(2019)	
torch.cuda.manual_seed(2018)	
######################################################################	
#paras setting	
parser = argparse.ArgumentParser(description='Training')	
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')	
parser.add_argument('--name',default='multiplefeatures', type=str, help='output model name')	
parser.add_argument('--data_dir',default='/Project0551/guoqing/scscnet/cuhk03-np/labeled/pytorch',type=str, help='training dir path')	
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')	
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')	
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')	
parser.add_argument('--epochnum', default=150, type=int, help='please to select the epoch num')	
parser.add_argument('--base_lr', default=0.01, type=float, help='the base_learning rate')	
parser.add_argument('--tripletmargin', default=1.0, type=float, help='the tripletmargin')	
parser.add_argument('--warmup_begin_lr', default=3e-4, type=float, help='warmup learning rate')	
parser.add_argument('--factor', default=0.5, type=float, help='the learning rate decracy')	
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )	
parser.add_argument('--attentionmodel', action='store_true', help='use the attention model')	
parser.add_argument('--testing', action='store_true', help='import testing features')	
opt = parser.parse_args()	
train_all_1 = 'True'	
str_ids = opt.gpu_ids.split(',')	
gpu_ids = []	
if not os.path.exists('./model/%s' % opt.name):	
        os.makedirs('./model/%s' % opt.name)	
for str_id in str_ids:	
    gid = int(str_id)	
    if gid >=0:	
        gpu_ids.append(gid)	
# set gpu ids	
if len(gpu_ids)>0:	
    torch.cuda.set_device(gpu_ids[0])	
######################################################################	
# Load Data	
transform_train_list = [	
        transforms.Resize([384, 128]),	
        transforms.RandomHorizontalFlip(),	
        transforms.ToTensor(),	
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])	
        ]	
transform_val_list = [	
        transforms.Resize(size=(384,128),interpolation=3), #Image.BICUBIC	
        transforms.ToTensor(),	
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])	
        ]	
if opt.erasing_p>0:	
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p,mean=[0.0, 0.0, 0.0])]	
if opt.color_jitter:	
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list	
print(transform_train_list)	
data_transforms = {	
    'train': transforms.Compose( transform_train_list ),	
    'val': transforms.Compose(transform_val_list),	
}	
train_all = ''	

if train_all_1:	
     train_all = '_all'	
image_datasets = {}	
image_datasets['train'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'train' + train_all),	
                                          data_transforms['train'])	
image_datasets['val'] = datasets.ImageFolder(os.path.join(opt.data_dir, 'val'),	
                                          data_transforms['val'])	
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= opt.batchsize,	
                                              sampler= RandomIdentitySampler(image_datasets[x],opt.batchsize,4), num_workers=8) # 8 workers may work faster	
              for x in ['train']}	
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}	
class_names = image_datasets['train'].classes	
use_gpu = torch.cuda.is_available()	
inputs, classes = next(iter(dataloaders['train']))	
######################################################################	
y_loss = {}	
y_loss['train'] = []	
y_loss['val'] = []	
y_err = {}	
y_err['train'] = []	
y_err['val'] = []	
##################################train model#########################	
def train_model(model, criterion,triplet, num_epochs):	
    since = time.time()	
    # best_model_wts = model.state_dict()	
    # best_acc = 0.0	
    for epoch in range(num_epochs):	
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))	
        print('-' * 10)	
        # update learning rate	
        lr_scheduler = LRScheduler(base_lr=opt.base_lr, step=[60,90,120],	
                               factor=opt.factor, warmup_epoch= opt.warm_epoch,	
                               warmup_begin_lr=opt.warmup_begin_lr)	
        lr = lr_scheduler.update(epoch)	
        optimizer = optim.SGD(model.parameters(), lr = lr,weight_decay=5e-4,momentum=0.9, nesterov=True)	
        print(lr)	
        for param_group in optimizer.param_groups:	
        	param_group['lr'] = lr	
        # Each epoch has a training and validation phase	
        for phase in ['train']:	
            if phase == 'train':	
                #scheduler.step()	
                model.train(True)  # Set model to training mode	
            else:	
                model.train(False)  # Set model to evaluate mode	

            running_loss = 0.0	
            running_corrects = 0.0	
            # Iterate over data.	
            for data in dataloaders[phase]:	
                # get the inputs	
                inputs, labels = data	
                now_batch_size,c,h,w = inputs.shape	
                if now_batch_size<opt.batchsize: # skip the last batch	
                    continue	
                # wrap them in Variable	
                if use_gpu:	
                    inputs = inputs.cuda()	
                    labels = labels.cuda()	
                else:	
                    inputs, labels = Variable(inputs), Variable(labels)	
                temp_loss = []	
                # zero the parameter gradients	
                optimizer.zero_grad()	
                # forward	
                outputs1,outputs2,outputs3,outputs4,outputs5,outputs6,outputs7,outputs8,outputs9,outputs10,outputs11,outputs12,outputs13,outputs14,outputs15,outputs16,outputs17,outputs18,outputs19,outputs20,outputs21,q1,q2,q3,q4,q5,q6= model(inputs)	
                _, preds1 = torch.max(outputs1.data, 1)	
                _, preds2 = torch.max(outputs2.data, 1)	
                _, preds3 = torch.max(outputs3.data, 1)	
                _, preds4 = torch.max(outputs4.data, 1)	
                _, preds5 = torch.max(outputs5.data, 1)	
                _, preds6 = torch.max(outputs6.data, 1)	
                _, preds7 = torch.max(outputs7.data, 1)	
                _, preds8 = torch.max(outputs8.data, 1)	
                _, preds9 = torch.max(outputs9.data, 1)	
                _, preds10 = torch.max(outputs10.data, 1)	
                _, preds11 = torch.max(outputs11.data, 1)	
                _, preds12 = torch.max(outputs12.data, 1)	
                _, preds13 = torch.max(outputs13.data, 1)	
                _, preds14 = torch.max(outputs14.data, 1)	
                _, preds15 = torch.max(outputs15.data, 1)	
                _, preds16 = torch.max(outputs16.data, 1)	
                _, preds17 = torch.max(outputs17.data, 1)	
                _, preds18 = torch.max(outputs18.data, 1)	
                _, preds19 = torch.max(outputs19.data, 1)	
                _, preds20 = torch.max(outputs20.data, 1)	
                _, preds21 = torch.max(outputs21.data, 1)	


                loss1 = criterion(outputs1, labels)	
                loss2 = criterion(outputs2, labels)	
                loss3 = criterion(outputs3, labels)	
                loss4 = criterion(outputs4, labels)	
                loss5 = criterion(outputs5, labels)	
                loss6 = criterion(outputs6, labels)	
                loss7 = criterion(outputs7, labels)	
                loss8 = criterion(outputs8, labels)	
                loss9 = criterion(outputs9, labels)	
                loss10 = criterion(outputs10, labels)	
                loss11 = criterion(outputs11, labels)	
                loss12 = criterion(outputs12, labels)	
                loss13 = criterion(outputs13, labels)	
                loss14 = criterion(outputs14, labels)	
                loss15 = criterion(outputs15, labels)	
                loss16 = criterion(outputs16, labels)	
                loss17 = criterion(outputs17, labels)	
                loss18 = criterion(outputs18, labels)	
                loss19 = criterion(outputs19, labels)	
                loss20 = criterion(outputs20, labels)	
                loss21 = criterion(outputs21, labels)	

                tloss1 = triplet(q1, labels)[0]	
                tloss2 = triplet(q2, labels)[0]	
                tloss3 = triplet(q3, labels)[0]	
                tloss4 = triplet(q4, labels)[0]	
                tloss5 = triplet(q5, labels)[0]	
                tloss6 = triplet(q6, labels)[0]	
                #	
                temp_loss.append(loss1)	
                temp_loss.append(loss2)	
                temp_loss.append(loss3)	
                temp_loss.append(loss4)	
                temp_loss.append(loss5)	
                temp_loss.append(loss6)	
                temp_loss.append(loss7)	
                temp_loss.append(loss8)	
                temp_loss.append(loss9)	
                temp_loss.append(loss10)	
                temp_loss.append(loss11)	
                temp_loss.append(loss12)	
                temp_loss.append(loss13)	
                temp_loss.append(loss14)	
                temp_loss.append(loss15)	
                temp_loss.append(loss16)	
                temp_loss.append(loss17)	
                temp_loss.append(loss18)	
                temp_loss.append(loss19)	
                temp_loss.append(loss20)	
                temp_loss.append(loss21)	
                loss = sum(temp_loss)/21+(tloss1+tloss2+tloss3+tloss4+tloss5+tloss6)/6	
                # backward + optimize only if in training phase	
                if phase == 'train':	
                    loss.backward()	
                    optimizer.step()	
                running_loss += loss.item() * now_batch_size	
                a   = float(torch.sum(preds1 == labels.data))	
                b   = float(torch.sum(preds2 == labels.data))	
                c   = float(torch.sum(preds3 == labels.data))	
                d   = float(torch.sum(preds4 == labels.data))	
                e   = float(torch.sum(preds5 == labels.data))	
                f   = float(torch.sum(preds6 == labels.data))	
                g   = float(torch.sum(preds7 == labels.data))	
                h   = float(torch.sum(preds8 == labels.data))	
                a9  = float(torch.sum(preds9 == labels.data))	
                a10 = float(torch.sum(preds10 == labels.data))	
                a11 = float(torch.sum(preds11 == labels.data))	
                a12 = float(torch.sum(preds12 == labels.data))	
                a13 = float(torch.sum(preds13 == labels.data))	
                a14 = float(torch.sum(preds14 == labels.data))	
                a15 = float(torch.sum(preds15 == labels.data))	
                a16 = float(torch.sum(preds16 == labels.data))	
                a17 = float(torch.sum(preds17 == labels.data))	
                a18 = float(torch.sum(preds18 == labels.data))	
                a19 = float(torch.sum(preds19 == labels.data))	
                a20 = float(torch.sum(preds20 == labels.data))	
                a21 = float(torch.sum(preds21 == labels.data))	
                #	
                running_corrects_1 = a + b +c + d+ e+ f+ g+ h +a9+a10+a11+a12+a13+a14+a15+a16+a17+a18+a19+a20+a21	
                running_corrects_2 = running_corrects_1 /21	
                running_corrects +=running_corrects_2	


            epoch_loss = running_loss / dataset_sizes[phase]	
            epoch_acc = running_corrects / dataset_sizes[phase]	
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(	
                phase, epoch_loss, epoch_acc))	
            time_elapsed = time.time() - since	
            print('Training times in {:.0f}m {:.0f}s'.format(	
                time_elapsed // 60, time_elapsed % 60))	

            y_loss[phase].append(epoch_loss)	
            y_err[phase].append(1.0-epoch_acc)            	
            # deep copy the model	
            if phase == 'train':	
                last_model_wts = model.state_dict()	
                if epoch < 150:	
                     if epoch%10 == 9:	
                         save_network(model, epoch)	
                     draw_curve(epoch)	
                else:	
                    #if epoch%2 == 0:	
                    save_network(model, epoch)	
                    draw_curve(epoch)	

        print()	
    time_elapsed = time.time() - since	
    print('Training complete in {:.0f}m {:.0f}s'.format(	
        time_elapsed // 60, time_elapsed % 60))	
    # load best model weights	
    model.load_state_dict(last_model_wts)	
    save_network(model, 'last')	
    return model	
######################################################################	
# Draw Curve	
x_epoch = []	
fig = plt.figure(figsize=(32,16))	
ax0 = fig.add_subplot(121, title="loss")	
ax1 = fig.add_subplot(122, title="top1err")	
def draw_curve(current_epoch):	
    x_epoch.append(current_epoch)	
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')	
    #ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')	
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')	
    #ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')	
    if current_epoch == 0:	
        ax0.legend()	
        ax1.legend()	
    fig.savefig( os.path.join('./model',opt.name,'train.jpg'))	
######################################################################	
# Save model	
def save_network(network, epoch_label):	
    save_filename = 'net_%s.pth'% epoch_label	
    save_path = os.path.join('./model',opt.name,save_filename)	
    torch.save(network.cpu().state_dict(), save_path)	
    if torch.cuda.is_available():	
        network.cuda(gpu_ids[0])	
######################################################################	
model = MultipleFeatures(class_num=len(class_names),testing=opt.testing,attentionusing=opt.attentionmodel)	
# print(model)	
input = Variable(torch.FloatTensor(8, 3, 384, 128))	
output = model(input)	
for i in output:	
    print(i.shape)	
if use_gpu:	
    model = model.cuda()	
triplet = TripletLoss(margin=opt.tripletmargin)	
criterion = CrossEntropyLabelSmooth(num_classes=len(class_names))	
print(len(class_names))	
dir_name = os.path.join('./model',opt.name)	
if os.path.isdir(dir_name):	
    copyfile('./train.py', dir_name+'/train.py')	
    copyfile('MultipleFeatures.py', dir_name + '/model.py')	
model = train_model(model, criterion, triplet, num_epochs=opt.epochnum)
