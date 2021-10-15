import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBottleNeck(nn.Module):
    # ResNeXt는 2를 곱해줌
    mul = 2
    def __init__(self, in_planes, group_width, cardinality, stride=1):
        super(ResNeXtBottleNeck, self).__init__()
        
        #첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        
        # ResNeXt 논문의 C 아키텍쳐 구현
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=1, padding=1, groups = cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        
        self.conv3 = nn.Conv2d(group_width, group_width*self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(group_width*self.mul)
        
        self.shortcut = nn.Sequential()
        
        # 합연산이 가능하도록 Identifier를 맞춰줌
        if stride != 1 or in_planes != group_width*self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, group_width*self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(group_width*self.mul)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out       
    
    
class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality = 32, width = 4, num_classes=10):
        super(ResNeXt, self).__init__()
        #RGB 3개채널에서 64개의 Kernel 사용
        self.in_planes = 64
        self.group_conv_width = cardinality * width # 128
        
        # ResNeXt 논문 구조 그대로 구현
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding = 3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(block, cardinality, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, cardinality, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, cardinality, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, cardinality, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(self.group_conv_width, num_classes)
        
    def make_layer(self, block, cardinality, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, self.group_conv_width, cardinality, strides[i]))
            self.in_planes = block.mul * self.group_conv_width
        self.group_conv_width *= block.mul
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out
    
    
def ResNeXt50():
    return ResNeXt(ResNeXtBottleNeck, [3, 4, 6, 3])

def ResNeXt101():
    return ResNeXt(ResNeXtBottleNeck, [3, 4, 23, 3])

def ResNeXt152():
    return ResNeXt(ResNeXtBottleNeck, [3, 8, 36, 3])