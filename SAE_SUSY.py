import os,sys
import numpy as np
import torch # pytorch package, allows using GPUs
import torch.nn as nn # construct NN
import torch.nn.functional as F # implements forward and backward definitions of an autograd operation
import torch.optim as optim # different update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc
from torch.autograd import Variable
# fix seed
seed=17
np.random.seed(seed)
torch.manual_seed(seed)

from torchvision import datasets # load data

class SUSY_Dataset(torch.utils.data.Dataset):
    """SUSY pytorch dataset."""

    def __init__(self, data_file, root_dir, dataset_size, train=True, transform=None, high_level_feats=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            train (bool, optional): If set to `True` load training data.
            transform (callable, optional): Optional transform to be applied on a sample.
            high_level_festures (bool, optional): If set to `True`, working with high-level features only. 
                                        If set to `False`, working with low-level features only.
                                        Default is `None`: working with all features
        """

        import pandas as pd

        features=['SUSY','lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi', 
                'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2', 
                'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']

        low_features=['lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi', 
                'missing energy magnitude', 'missing energy phi']

        high_features=['MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2','S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']


        #Number of datapoints to work with
        df = pd.read_csv(root_dir+data_file, header=None,nrows=dataset_size,engine='python')
        df.columns=features
        Y = df['SUSY']
        X = df[[col for col in df.columns if col!="SUSY"]]

        # set training and test data size
        train_size=int(0.8*dataset_size)
        self.train=train

        if self.train:
            X=X[:train_size]
            Y=Y[:train_size]
            print("Training on {} examples".format(train_size))
        else:
            X=X[train_size:]
            Y=Y[train_size:]
            print("Testing on {} examples".format(dataset_size-train_size))


        self.root_dir = root_dir
        self.transform = transform

        # make datasets using only the 8 low-level features and 10 high-level features
        if high_level_feats is None:
            self.data=(X.values.astype(np.float32),Y.values.astype(int))
            print("Using both high and low level features")
        elif high_level_feats is True:
            self.data=(X[high_features].values.astype(np.float32),Y.values.astype(int))
            print("Using both high-level features only.")
        elif high_level_feats is False:
            self.data=(X[low_features].values.astype(np.float32),Y.values.astype(int))
            print("Using both low-level features only.")


    # override __len__ and __getitem__ of the Dataset() class

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):

        sample=(self.data[0][idx,...],self.data[1][idx])

        if self.transform:
            sample=self.transform(sample)

        return sample
    
def load_data(train_batch_size,test_batch_size,dataset_size,high_level_feats=None):
    
    data_file='SUSY.csv'
    root_dir='agnews/'

    kwargs = {} # CUDA arguments, if enabled
    # load and noralise train and test data
    train_loader = torch.utils.data.DataLoader(
        SUSY_Dataset(data_file,root_dir,dataset_size,train=True,high_level_feats=high_level_feats),
        batch_size=train_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        SUSY_Dataset(data_file,root_dir,dataset_size,train=False,high_level_feats=high_level_feats),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

class model(nn.Module):
    def __init__(self):
        # inherit attributes and methods of nn.Module
        super(model, self).__init__()
        #=========== encoder ================
        # an affine operation: y = Wx + b        
        self.fc_en1 = nn.Linear(18, 200) # all features
        # self.batchnorm1=nn.BatchNorm1d(200, eps=1e-05, momentum=0.1)
        # self.batchnorm2=nn.BatchNorm1d(100, eps=1e-05, momentum=0.1)
        self.fc_en2 = nn.Linear(200, 100) # see forward function for dimensions
        #=========== decoder ================
        self.fc_de1 = nn.Linear(100,200)
        self.fc_de2 = nn.Linear(200,18)
        #=========== classifier ================
        self.cl1 = nn.Linear(100, 2)

    def forward(self, x):
        #===== Encoder ==============
        # apply rectified linear unit
        en1 = F.relu(self.fc_en1(x))
        en1 = F.dropout(en1, training=self.training)
        en2 = F.relu(self.fc_en2(en1))
        #===== Decoder ================
        de1 = F.relu(self.fc_de1(en2))
        de1 = F.dropout(de1, training=self.training)
        x_hat = F.relu(self.fc_de2(de1))
        #===== Classifier =============
        cl1 = self.cl1(en2)
        y_hat = F.log_softmax(cl1,dim=1)

        return y_hat, x_hat

def train(train_loader,test_loader,wp,wr):
    print('=================== wr: ',wr,' ===================')
    load_model = False
    history = []
    path = './saved_models/'
    filename = '20epoch-wr'+str(wr)+'-SUSY'
    
    DNN = model()
    DNN.cuda()
    
    epochs=20
    classificationLoss = torch.nn.NLLLoss().cuda()
    reconstructionLoss = nn.L1Loss().cuda()
    optimizer = optim.SGD(DNN.parameters(), lr = 0.001, momentum = 0.8)

    if load_model:
        checkpoint = torch.load(path+filename+".ckp")
        DNN.load_state_dict(checkpoint[0]['model_state_dict'])
        optimizer.load_state_dict(checkpoint[0]['optimizer_state_dict'])
        init_epoch = checkpoint[0]['epoch']+1
        history=(np.load(path+filename+'-history.npy',history)).tolist()

    for epoch in range(epochs):
        DNN.train()
        model_acc = 0.0
        model_loss = 0.0
        for batch_idx, (data,label) in enumerate(train_loader):
            data = Variable(data).cuda()
            label = Variable(label).cuda()
            optimizer.zero_grad()
            y_hat, x_hat = DNN(data)
            loss = wp*classificationLoss(y_hat,label)+wr*reconstructionLoss(x_hat,data)
            loss.backward()
            optimizer.step()
            _,predicted = torch.max(y_hat.data,1)
            num_correct = (predicted==label).sum().item()
            model_acc += num_correct/(len(train_loader)*data.size(0))
            model_loss += loss.item()/len(train_loader)
        print('Epoch: ', epoch, ' - train_loss: ',model_loss, ' - train_acc: ',model_acc)
        
        
        DNN.eval()
        
        test_acc = 0.0
        test_loss = 0.0
        total_data = 0.0
        for data, label in test_loader:
            data = Variable(data).cuda()
            label = Variable(label).cuda()
            y_hat, x_hat = DNN(data)
            _,predicted = torch.max(y_hat.data,1)
            total_data += label.size(0)
            test_acc += (predicted==label).sum().item()/(len(test_loader)*data.size(0))
        print('Model Test accuracy: ', test_acc)
        history.append([model_acc,test_acc])
        
        checkpoint = {'epoch':epoch,'model_state_dict':DNN.state_dict(),'optimizer_state_dict': optimizer.state_dict()},
        torch.save(checkpoint, path+filename+'.ckp') #save and load for inference
        torch.save(DNN,path+filename+'.mdl') #saving entire model
    np.save(path+filename+'-history.npy',history)


train_batch_size,test_batch_size,dataset_size = 100,1000,200000
train_loader, test_loader = load_data(train_batch_size,test_batch_size,dataset_size)   
for wr in range(1,10,1): 
    train(train_loader,test_loader,1,wr/10.)
    break
