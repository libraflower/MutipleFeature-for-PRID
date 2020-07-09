import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

#####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)


###################the classifier-basic model####################################
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv1x1 = conv1x1(input_dim * 2, 256)
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(256)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(256, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        xmax = self.maxpool(x)
        xavg = self.avgpool(x)
        x = torch.cat((xmax,xavg),dim=1)
        x = self.conv1x1(x)
        x = torch.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x2,x1,x1

#######################1*1 conv#################################################
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
####################MultipleFeatures model#####################################
class MultipleFeatures(nn.Module):

    def __init__(self,class_num,testing=False,attentionusing=True):
        super(MultipleFeatures, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        #some agrs
        self.channels = 3840
        self.test = testing
        self.attentionmodel = attentionusing
        #ResNEt-50
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.layer4[0].downsample[0].stride = (1,1)
        self.layer4[0].conv2.stride = (1,1)
        #pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool1x1 = nn.AdaptiveMaxPool2d((1,1))
        self.avgpoolall = nn.AdaptiveAvgPool2d((24, 8))
        self.maxpoolall = nn.AdaptiveMaxPool2d((24, 8))
        #classfier blocks
        self.classifierlayer1_1 = ClassBlock(self.channels, class_num)

        self.classifierlayer2_1 = ClassBlock(self.channels, class_num)
        self.classifierlayer2_2 = ClassBlock(self.channels, class_num)

        self.classifierlayer3_1 = ClassBlock(self.channels, class_num)
        self.classifierlayer3_2 = ClassBlock(self.channels, class_num)
        self.classifierlayer3_3 = ClassBlock(self.channels, class_num)

        self.classifierlayer4_1 = ClassBlock(self.channels, class_num)
        self.classifierlayer4_2 = ClassBlock(self.channels, class_num)
        self.classifierlayer4_3 = ClassBlock(self.channels, class_num)
        self.classifierlayer4_4 = ClassBlock(self.channels, class_num)

        self.classifierlayer5_1 = ClassBlock(self.channels, class_num)
        self.classifierlayer5_2 = ClassBlock(self.channels, class_num)
        self.classifierlayer5_3 = ClassBlock(self.channels, class_num)
        self.classifierlayer5_4 = ClassBlock(self.channels, class_num)
        self.classifierlayer5_5 = ClassBlock(self.channels, class_num)

        self.classifierlayer6_1 = ClassBlock(self.channels, class_num)
        self.classifierlayer6_2 = ClassBlock(self.channels, class_num)
        self.classifierlayer6_3 = ClassBlock(self.channels, class_num)
        self.classifierlayer6_4 = ClassBlock(self.channels, class_num)
        self.classifierlayer6_5 = ClassBlock(self.channels, class_num)
        self.classifierlayer6_6 = ClassBlock(self.channels, class_num)

    def forward(self, x):
        x = x.view(-1, *x.size()[-3:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #平均池化到24*8 x4的大小本身就为24*8

        x1_avg = self.maxpoolall(x1)
        x2_avg = self.maxpoolall(x2)
        x3_avg = x3
        x4_avg = x4
        x_cat = torch.cat((x1_avg,x2_avg,x3_avg,x4_avg),dim=1)

        #对feature map分块成P1-P6 每一块的维度为 4*8*3840
        p1 = x_cat[:, :, 0:4, :]
        p2 = x_cat[:, :, 4:8, :]
        p3 = x_cat[:, :, 8:12, :]
        p4 = x_cat[:, :, 12:16, :]
        p5 = x_cat[:, :, 16:20, :]
        p6 = x_cat[:, :, 20:24, :]

        #金字塔第一层 维度为24*8*3840
        py1 = x_cat
        py1id , py1tr ,test11= self.classifierlayer1_1(py1)
        #金字塔第二层 维度为20*8*3840
        py21 = torch.cat((p1, p2, p3, p4, p5), dim=2)
        py21id, py21tr , test21 = self.classifierlayer2_1(py21)
        py22 = torch.cat((p2, p3, p4, p5, p6), dim=2)
        py22id, py22tr , test22 = self.classifierlayer2_2(py22)
        #金字塔第三层 维度为16*8*3840
        py31 = torch.cat((p1, p2, p3, p4), dim=2)
        py31id, py31tr ,test31 = self.classifierlayer3_1(py31)
        py32 = torch.cat((p2, p3, p4, p5), dim=2)
        py32id, py32tr ,test32 = self.classifierlayer3_2(py32)
        py33 = torch.cat((p3, p4, p5, p6), dim=2)
        py33id, py33tr ,test33 = self.classifierlayer3_3(py33)
        #金字塔第四层 维度为12*8*3840
        py41 = torch.cat((p1, p2, p3), dim=2)
        py41id, py41tr ,test41 = self.classifierlayer4_1(py41)
        py42 = torch.cat((p2, p3, p4), dim=2)
        py42id, py42tr ,test42 = self.classifierlayer4_2(py42)
        py43 = torch.cat((p3, p4, p5), dim=2)
        py43id, py43tr ,test43 = self.classifierlayer4_3(py43)
        py44 = torch.cat((p4, p5, p6), dim=2)
        py44id, py44tr ,test44 = self.classifierlayer4_4(py44)
        #金字塔第五层 维度为8*8*3840
        py51 = torch.cat((p1, p2), dim=2)
        py51id, py51tr ,test51 = self.classifierlayer5_1(py51)
        py52 = torch.cat((p2, p3), dim=2)
        py52id, py52tr ,test52 = self.classifierlayer5_2(py52)
        py53 = torch.cat((p3, p4), dim=2)
        py53id, py53tr ,test53 = self.classifierlayer5_3(py53)
        py54 = torch.cat((p4, p5), dim=2)
        py54id, py54tr ,test54 = self.classifierlayer5_4(py54)
        py55 = torch.cat((p5, p6), dim=2)
        py55id, py55tr ,test55 = self.classifierlayer5_5(py55)
        #金字塔第六层  4*8*3840
        py61 = p1
        py61id, py61tr ,test61 = self.classifierlayer6_1(py61)
        py62 = p2
        py62id, py62tr ,test62 = self.classifierlayer6_2(py62)
        py63 = p3
        py63id, py63tr ,test63 = self.classifierlayer6_3(py63)
        py64 = p4
        py64id, py64tr ,test64 = self.classifierlayer6_4(py64)
        py65 = p5
        py65id, py65tr ,test65 = self.classifierlayer6_5(py65)
        py66 = p6
        py66id, py66tr ,test66 = self.classifierlayer6_6(py66)

        triplet1 = py1tr
        triplet2 = torch.cat((py21tr,py22tr),dim=1)
        triplet3 = torch.cat((py31tr,py32tr,py33tr),dim=1)
        triplet4 = torch.cat((py41tr,py42tr,py43tr,py44tr),dim=1)
        triplet5 = torch.cat((py51tr,py52tr,py53tr,py54tr,py55tr),dim=1)
        triplet6 = torch.cat((py61tr,py62tr,py63tr,py64tr,py65tr,py66tr),dim=1)


        if self.test == True:
            return test11,\
               test21,test22,\
               test31,test32,test33,\
               test41,test42,test43,test44,\
               test51,test52,test53,test54,test55,\
               test61,test62,test63,test64,test65,test66
        else:
            return py1id,\
               py21id,py22id,\
               py31id,py32id,py33id,\
               py41id,py42id,py43id,py44id,\
               py51id,py52id,py53id,py54id,py55id,\
               py61id,py62id,py63id,py64id,py65id,py66id,\
               triplet1,triplet2,triplet3,triplet4,triplet5,triplet6

