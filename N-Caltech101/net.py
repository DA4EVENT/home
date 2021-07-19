import torch.nn as nn
import torch
from torchvision import models
from torch.autograd import Function
from mmd_loss import MMDLoss




class ResBase(nn.Module):
    def __init__(self, architecture, channels_event=3, AvgChannels=False, use_projection=False, device=None):
        super(ResBase, self).__init__()

        if architecture == 'resnet50':
            model_resnet = models.resnet50(pretrained=True)
            self.out_channel = 2048
        elif architecture == 'resnet34':
            model_resnet = models.resnet34(pretrained=True) #todo adapt to take in input K channels for the event data
            self.out_channel = 512
        elif architecture == 'resnet18':
            model_resnet = models.resnet18(pretrained=True)
        else:
            raise ValueError("{} is not a valid known ResNet architecture.".format(architecture))

        self.use_projection = use_projection
        self.channels_event = channels_event
        self.device = device

        self.conv1 = model_resnet.conv1
        self.AvgChannels = AvgChannels
        if self.AvgChannels:
            self.avgTemporal = nn.AvgPool2d((self.channels_event//3, 1))
            self.avgTemporal_Spatial = nn.AvgPool3d((self.channels_event//3, 1,1))

        if self.channels_event != 3 and not self.AvgChannels:
            print("Create New InputConv ")
            self.conv1 = nn.Conv2d(self.channels_event, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)


        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

    def forward(self, x):
        bs = x.size(0)
        c_init = x.size(1)
        if self.AvgChannels:

            if c_init == 2: # EXTEND TO  3 Channels
                new_channel = torch.zeros(bs,1,224,224).to(self.device)
                new_channel[:,0,:,:] = x[:,0,:,:] + x[:,1,:,:]
                x = torch.cat((x,new_channel), 1)

            #BS C, 224,224
            x = x.reshape(-1, 3, 224, 224)
            #BS*Nchann/3, 3, 224,224

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x_p = x
        x = self.avgpool(x)
        if self.AvgChannels and self.channels_event > 3 :
            x = x.view(bs, (self.channels_event//3), self.out_channel)
            x_p = x_p.view(bs, (self.channels_event//3), self.out_channel, 7, 7)
            x_p = x_p.permute(0,2,1,3,4)

            x = self.avgTemporal(x)
            x_p = self.avgTemporal_Spatial(x_p)
            x_p = x_p.view(bs, self.out_channel, 7, 7)

        x = x.view(bs, -1)

        return x, x_p


class ResClassifier(nn.Module):
    def __init__(self, input_dim=2048, class_num=51, extract=True, dropout_p=0.5):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.fc2 = nn.Linear(1000, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        emb = self.fc1(x)
        logit = self.fc2(emb)

        if self.extract:
            return emb, logit
        return logit


class RelativeRotationClassifier(nn.Module):
    def __init__(self, input_dim, projection_dim=100, class_num=4):
        super(RelativeRotationClassifier, self).__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(self.input_dim, self.projection_dim, [1,1], stride=[1,1]),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
            )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(self.projection_dim, self.projection_dim, [3,3], stride=[2,2]),
            nn.BatchNorm2d(self.projection_dim),
            nn.ReLU(inplace=True)
            )
        self.fc1 = nn.Sequential(
            nn.Linear(self.projection_dim*3*3, self.projection_dim),
            nn.BatchNorm1d(self.projection_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
            )
        self.fc2 = nn.Linear(projection_dim, class_num)

    def forward(self, x):
        x = self.conv_1x1(x)
        x = self.conv_3x3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

### ADD GRL
class GradientReversalLayer(Function):

    @staticmethod
    def forward(self, x, lambda_val):
        self.lambda_val = lambda_val

        return x.view_as(x)

    @staticmethod
    def backward(self, grad_out):
        grad_in = grad_out.neg() * self.lambda_val

        return grad_in, None

### MMD
class ResClassifierMMD(ResClassifier):
    def __init__(self, mmd_loss=MMDLoss(), **kwargs):
        kwargs['extract'] = False
        super().__init__(**kwargs)

        self.mmd = mmd_loss

    def forward(self, source, target=None):
        if not self.training and target is None:
            return super().forward(source)

        assert target is not None

        mmd_loss = 0

        for layer in self.fc1:
            source = layer(source)
            target = layer(target)

            if isinstance(layer, nn.Linear):
                mmd_loss += self.mmd(source, target)

        source = self.fc2(source)
        target = self.fc2(target)
        mmd_loss += self.mmd(source, target)

        return source, mmd_loss
