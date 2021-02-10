import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import numpy as np
torch.manual_seed(1)

transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=True)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1))
        self.fc1_2 = nn.Linear(in_features=50176,out_features=256)
        self.fc2_3 = nn.Linear(in_features=256,out_features=32)
                             
        self.fc3_2 = nn.Linear(in_features=32,out_features=256)
        self.fc2_1 = nn.Linear(in_features=256,out_features=50176)
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(3,3), stride=(1,1))
        
        # self.fc1 = nn.Linear(in_features=2304, out_features=2048)
        # self.fc2 = nn.Linear(in_features=2048, out_features=512)
        # self.fc3 = nn.Linear(in_features=512, out_features=128)
        # self.fc4 = nn.Linear(in_features=128, out_features=32)
        self.softmax = nn.Linear(in_features=32, out_features=10)
        
    def forward(self,x):

        xe = F.relu(self.conv1(x))        
        # xe = F.max_pool2d(xe, 2, 2)
        xe = F.relu(self.conv2(xe))
        # xe = F.relu(self.conv3(xe))
        shp = [xe.shape[0],xe.shape[1],xe.shape[2],xe.shape[3]]
        xe = xe.view(-1,shp[1]*shp[2]*shp[3])
        xe = F.relu(self.fc1_2(xe))
        xe = F.relu(self.fc2_3(xe))
        
        xd = F.relu(self.fc3_2(xe))
        xd = F.relu(self.fc2_1(xd))
        xd = torch.reshape(xd,(shp[0],shp[1],shp[2],shp[3]))
        xd = F.relu(self.conv3(xd))
        # xd = F.upsample(xd,30)
        x_hat = F.relu(self.conv4(xd))
        
        y_hat = self.softmax(xe)
        
                
        # xd = F.relu(self.conv3(latent))
        # x_hat = F.relu(self.conv4(xd))
        
        # x = F.max_pool2d(latent, 2, 2)
        # x = x.view(-1, 2304)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))        
        # y_hat = self.fc5(x)
        return x_hat,y_hat
        
    def name(self):
        return 'lenet'

#arch1: conv32-mp-conv64-[convt32-convt3/mp-f2048-f512-f128-f32-10]
#arch2: conv32-mp-conv64-[convt32-convt3/mp-f256-10]
#arch3 conv32-mp-conv64-D256-D32[D32-D256-conv64-up-conv32/D10]
#arch4 conv32-conv64-D256[D256-conv64-conv32/D10]
#arch5 conv32-conv64-D256-D32[D32-D256-conv64-conv32/D10]
def train(weight):
    path = '/home/morteza/Documents/Dr. Wang/Semi_Supervised_Auto_Encoder-master/models/pytorch/'#curr_DIR = os.path.dirname(os.path.realpath(__file__))+'/models/'#'/home/morteza/Documents/Dr. Wang/Semi_Supervised_Auto_Encoder-master/models/pytorch/'
    filename = '20epochs-wr'+str(weight)+'-arch5'
    wp=1
    wr=weight
    load_model = False
    init_epoch = 0
    history = []

    net = ConvNet()
    net.cuda()
    classificationLoss = nn.CrossEntropyLoss()
    reconstructionLoss = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.9)

    if load_model:
        checkpoint = torch.load(path+filename+".ckp")
        net.load_state_dict(checkpoint[0]['model_state_dict'])
        optimizer.load_state_dict(checkpoint[0]['optimizer_state_dict'])
        init_epoch = checkpoint[0]['epoch']+1
        history=(np.load(path+filename+'-history.npy',history)).tolist()
        
    for epoch in range(init_epoch,30,1):    
        # train
        running_loss=0.0
        model_acc = 0.0
        total_images=0
        for i,data in enumerate(trainloader,0):
            inputs, labels = Variable(data[0]).cuda(), Variable(data[1]).cuda()
            optimizer.zero_grad()
            
            x_hat,y_hat = net(inputs)        
            
            loss = wp*classificationLoss(y_hat,labels)+wr*reconstructionLoss(x_hat,inputs)
            loss.backward()
            optimizer.step()        
            _,predicted = torch.max(y_hat.data,1)        
            total_images += labels.size(0)
            # train_acc += (pred_label==y).sum()/float(len(train_loader)*x.size(0))
            num_correct = (predicted==labels).sum().item()
            model_acc += num_correct/(len(trainloader)*inputs.size(0))
            running_loss += loss.item()/len(trainloader)
            
        print("Epoch: ", epoch, " - model_loss: "+ str(running_loss)+' - train_acc: '+str(model_acc))
        
        # sys.stdout.write()
        # sys.stdout.write('\r')
        # sys.stdout.flush()
            
        # test
        with torch.no_grad():
            total_correct = 0.0
            total_images = 0.0
            test_acc = 0.0
            for data in testloader:
                images, labels = Variable(data[0]).cuda(),Variable(data[1]).cuda()
                x_hat, y_hat = net(images)
                _,predicted = torch.max(y_hat.data,1)
                total_images += labels.size(0)
                test_acc += (predicted==labels).sum().item()/(len(testloader)*images.size(0))
            # test_acc = total_correct/(total_images * 100
            print ('Model Test accuracy:',test_acc)
        
        history.append([model_acc,test_acc])    
        
        # store model
        checkpoint = {'epoch':epoch,'model_state_dict':net.state_dict(),'optimizer_state_dict': optimizer.state_dict()},
        torch.save(checkpoint, path+filename+'.ckp') #save and load for inference
        #model.load_state_dict(torch.load(path))
        #model.eval()
        torch.save(net,path+filename+'.mdl') #saving entire model
        #model = torch.load(path)
        #model.eval()
    np.save(path+filename+'-history.npy',history)
        

for weight in [0.04,0.06,0.08]:#range(6,10):
    w=weight
    train(w)   
