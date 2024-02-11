import os
import time
import glob
import torch
import random
import numpy as np
import myutils as myutils
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch import nn
from PIL import Image
from torch import optim
from model_kd2 import generate_model
from model_tpp2 import generate_model_tpp
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore") 

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2train',      default='.',   type=str, help='')
    parser.add_argument('--path2test',       default='.',   type=str, help='')
    parser.add_argument('--batch_size',      default='',    type=int, help='')
    parser.add_argument('--timesteps',       default='',    type=int, help='')
    parser.add_argument('--num_epochs',      default='',    type=int, help='')
    parser.add_argument('--n_classes',       default='',    type=int, help='')
    parser.add_argument('--sample_size',     default='',    type=int, help='')
    parser.add_argument('--MSE_weight',      default= ,   type=int, help='')
    parser.add_argument('--KLD_weight',      default= ,   type=int, help='')
    parser.add_argument('--CE_weight',       default= ,   type=int, help='')
    parser.add_argument('--CT_model',        default='',    type=str, help='')
    parser.add_argument('--WSI_model',       default='',    type=str, help='')
    parser.add_argument('--FUS_model',       default='',    type=str, help='')
    parser.add_argument('--MEM_model',       default='',    type=str, help='')
    parser.add_argument('--sample_duration', default='',    type=int, help='')
    parser.add_argument('--memory',          default='',    type=str, help='')
    parser.add_argument('--mode',            default='',    type=str, help='Mode (score | feature).')
    parser.add_argument('--model_name',      default='',    type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth',     default=34,    type=int, help='ResNet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='',    type=str, help='Shortcut type of resnet (A | B)')
    args = parser.parse_args()
    return args

opt = parse_opts()
np.random.seed(2022)
random.seed(2022)
torch.manual_seed(2022)
mean = [0.5]
std  = [0.5]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('USE GPU', device)
else:
    print('USE CPU')


train_ids, train_labels, catgs = myutils.get_vids(opt.path2train)
print(len(train_ids), len(train_labels), len(catgs))

test_ids, test_labels, catgs = myutils.get_vids(opt.path2test)
print(len(test_ids), len(test_labels), len(catgs))

labels_dict = {}
ind = 0
for uc in catgs:
	labels_dict[uc] = ind
	ind += 1
print(labels_dict)

class VideoDatasetTrain(Dataset):
	def __init__(self, ids, labels, transform):
		self.transform = transform
		self.ids = ids
		self.labels = labels    
	def __len__(self):
		return len(self.ids)
	def __getitem__(self, idx):
		cla = os.path.split(os.path.split(self.ids[idx])[0])[1]
		num = random.randint( , )
		path = ("" + cla + "\\" + cla + "_(" + str(num) + ")")
        
		pathCTimgs = glob.glob(self.ids[idx]+"/*")
		pathWSIimgs = glob.glob(path+"/*")
		pathCTimgs = pathCTimgs[:opt.timesteps]
		pathWSIimgs = pathWSIimgs[:opt.timesteps]
		label = labels_dict[self.labels[idx]]
		
		framesCT = []
		for pCTi in pathCTimgs: # 将图像添加到一组帧
			frameCT = Image.open(pCTi)
			framesCT.append(frameCT)
		frames_trCT = []
        
		framesWSI = []
		for pWSIi in pathWSIimgs: 
			frameWSI = Image.open(pWSIi)
			framesWSI.append(frameWSI)
		frames_trWSI = []
         
		for frameCT in framesCT:
			frameCT = self.transform(frameCT)
			frames_trCT.append(frameCT)
		if len(frames_trCT)>0:
			frames_trCT = torch.stack(frames_trCT) # 2维拼接为3维
            
		for frameWSI in framesWSI:
			frameWSI = self.transform(frameWSI)
			frames_trWSI.append(frameWSI)
		if len(frames_trWSI)>0:
			frames_trWSI = torch.stack(frames_trWSI) 
        
		frames_tr = torch.cat((frames_trCT, frames_trWSI), dim=1)            
		return frames_tr, label


class VideoDatasetTest(Dataset):
	def __init__(self, ids, labels, transform):
		self.transform = transform
		self.ids = ids
		self.labels = labels
	def __len__(self):
		return len(self.ids)
	def __getitem__(self, idx):
		cla = os.path.split(os.path.split(self.ids[idx])[0])[1]
		num = random.randint(, )
		path = ("" + cla + "\\" + cla + "_(" + str(num) + ")")
        
		pathCTimgs = glob.glob(self.ids[idx]+"/*")
		pathWSIimgs = glob.glob(path+"/*")
		pathCTimgs = pathCTimgs[:opt.timesteps]
		pathWSIimgs = pathWSIimgs[:opt.timesteps]
		label = labels_dict[self.labels[idx]]
		
		framesCT = []
		for pCTi in pathCTimgs: 
			frameCT = Image.open(pCTi)
			framesCT.append(frameCT)
		frames_trCT = []
        
		framesWSI = []
		for pWSIi in pathWSIimgs: 
			frameWSI = Image.open(pWSIi)
			framesWSI.append(frameWSI)
		frames_trWSI = []
         
		for frameCT in framesCT:
			frameCT = self.transform(frameCT)
			frames_trCT.append(frameCT)
		if len(frames_trCT)>0:
			frames_trCT = torch.stack(frames_trCT) 
            
		for frameWSI in framesWSI:
			frameWSI = self.transform(frameWSI)
			frames_trWSI.append(frameWSI)
		if len(frames_trWSI)>0:
			frames_trWSI = torch.stack(frames_trWSI) 
        
		frames_tr = torch.cat((frames_trCT, frames_trWSI), dim=1)            
		return frames_tr, label
    
    
train_transformer = transforms.Compose([transforms.Resize((opt.sample_size, opt.sample_size)),
					                    transforms.RandomHorizontalFlip(p=0.5),
					                    transforms.ToTensor(),
					                    transforms.Normalize(mean,std) ])		

train_ds = VideoDatasetTrain(ids=train_ids, labels=train_labels, transform=train_transformer)


test_transformer = transforms.Compose([transforms.Resize((opt.sample_size, opt.sample_size)),
				                       transforms.ToTensor(),
				                       transforms.Normalize(mean,std) ])

test_ds = VideoDatasetTest(ids=test_ids, labels=test_labels, transform=test_transformer)

def collate_fn_3dcnn(batch):
	imgs_batch, label_batch = list(zip(*batch)) #组合每个batch的img和label
	imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
	label_batch = [torch.tensor(l) for l,imgs in zip(label_batch,imgs_batch) if len(imgs)>0]
	imgs_tensor = torch.stack(imgs_batch)
	imgs_tensor = torch.transpose(imgs_tensor,2,1)
	labels_tensor = torch.stack(label_batch)
	return imgs_tensor, labels_tensor

train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_3dcnn)
test_dl = DataLoader(test_ds, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn_3dcnn)

model_CT = torch.load(opt.CT_model).to(device)
for param_CT in model_CT.parameters():
    param_CT.requires_grad = False
    
model_WSI = torch.load(opt.WSI_model).to(device)
for param_WSI in model_WSI.parameters():
    param_WSI.requires_grad = False
    
model_MEM = torch.load(opt.MEM_model).to(device)
for param_MEM in model_MEM.parameters():
    param_MEM.requires_grad = False

model_FUS = torch.load(opt.FUS_model).to(device)
for param_FUS in model_FUS.parameters():
    param_FUS.requires_grad = False

model_TP = torch.load(opt.FUS_model).to(device)
model_FUS = generate_model_tpp(opt).to(device)
model_dict = model_FUS.state_dict()
pretrained_dict = model_TP.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model_FUS.load_state_dict(model_dict)

for param_FUS in model_FUS.parameters():
    param_FUS.requires_grad = False

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group["lr"]

S_model = generate_model(opt).to(device)

criterionMSE = nn.L1Loss().to(device)
criterionKLD = nn.KLDivLoss().to(device)
criterionCE = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(S_model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=1)
os.makedirs("./models", exist_ok=True)


'''主函数'''
if __name__=='__main__': 
	best_test_acc_CT = 0
	best_test_acc_WSI = 0
	best_test_acc_FUS = 0
	best_test_acc_STU = 0
    
	S_model.train()
    
	for epoch in range(opt.num_epochs):
		print()
		start_time = time.time()
		current_lr = get_lr(optimizer)
		iter_loss = 0.0
		iter_CE = 0.0
		iter_KLD = 0.0
		iter_MSE = 0.0
        
		correct_train_CT = 0
		correct_train_WSI = 0
		correct_train_FUS = 0
		correct_train_STU = 0
		total_train = 0
        
		correct_test_CT = 0
		correct_test_WSI = 0
		correct_test_FUS = 0
		correct_test_STU = 0
		total_test = 0
		
		print("Epoch {}/{}, current lr={}".format(epoch+1, opt.num_epochs, current_lr))
        
		for i, (img,label) in enumerate(train_dl):
            
		    img = img.to(device)
		    CT = img[:,0:1,:,:,:]
		    WSI = img[:,1:,:,:,:]
		    label = label.to(device)
            
		    optimizer.zero_grad()

		    CT1, CT2, CT3, CT4, CT_out = model_CT(CT)
		    WSI1, WSI2, WSI3, WSI4, WSI_out = model_WSI(WSI)
		    MM1, MM2, MM3 = model_MEM(WSI1, WSI2, WSI3, label)
		    MM1, MM2, MM3 = model_MEM(WSI1, WSI2, WSI3, label)
		    FUS1, FUS2, FUS3, FUS_out = model_FUS(CT1.detach(), CT2.detach(), CT3.detach(), WSI1.detach(), WSI2.detach(), WSI3.detach())
            
		    smpre = F.softmax(CT_out.detach(), dim=1)
		    S1, S2, S3, STU_out = S_model(CT, MM1.detach(), MM2.detach(), MM3.detach(), smpre)

		    loss_MSE1 = criterionMSE(S1, FUS1.detach())
		    loss_MSE2 = criterionMSE(S2, FUS2.detach()) 
		    loss_MSE3 = criterionMSE(S3, FUS3.detach()) 
		    loss_MSE = (1/3)*loss_MSE1 + (1/3)*loss_MSE2 + (1/3)*loss_MSE3
            
		    T = 20   
		    SS_out = F.log_softmax(STU_out/T, dim=1)
		    TT_out = F.softmax(FUS_out.detach()/T, dim=1)
		    loss_KLD = criterionKLD(SS_out, TT_out)*T*T
            
		    loss_CE = criterionCE(STU_out, label)  
            

		    all_loss = opt.MSE_weight*loss_MSE + opt.KLD_weight*loss_KLD + opt.CE_weight*loss_CE
            
		    iter_loss += all_loss.item()
		    iter_CE += opt.CE_weight*loss_CE.item()
		    iter_KLD += opt.KLD_weight*loss_KLD.item()
		    iter_MSE += opt.MSE_weight*loss_MSE.item()

		    _, pre_CT = torch.max(CT_out.data, 1)   
		    _, pre_WSI = torch.max(WSI_out.data, 1) 
		    _, pre_FUS = torch.max(FUS_out.data, 1) 
		    _, pre_STU = torch.max(STU_out.data, 1) 
            
		    total_train += label.size(0)
		    correct_train_CT += (pre_CT == label).sum().item() 
		    correct_train_WSI += (pre_WSI == label).sum().item()
		    correct_train_FUS += (pre_FUS == label).sum().item()
		    correct_train_STU += (pre_STU == label).sum().item()
            
		    all_loss.backward()     
		    optimizer.step()        
            
		    if i % 40 == 39:
		        print('[%d, %d] CE_loss: %.03f | KDL_loss: %.03f | MSE_loss: %.03f' % (epoch+1, i+1, iter_CE, iter_KLD, iter_MSE))
		        iter_loss = 0.0
		        iter_CE = 0.0
		        iter_KLD = 0.0
		        iter_MSE = 0.0
                

		train_acc_CT = 100.0*correct_train_CT/total_train
		train_acc_WSI = 100.0*correct_train_WSI/total_train
		train_acc_FUS = 100.0*correct_train_FUS/total_train
		train_acc_STU = 100.0*correct_train_STU/total_train
        
		print('Train_ACC_CT={}'.format(train_acc_CT))
		print('Train_ACC_WSI={}'.format(train_acc_WSI))
		print('Train_ACC_FUS={}'.format(train_acc_FUS))
		print('Train_ACC_STU={}'.format(train_acc_STU))
		lr_scheduler.step(all_loss)
        
        
        # 模型测试
		S_model.eval()
		with torch.no_grad():
		    for j, (img_test, label_test) in enumerate(test_dl):
                
		        img_test = img_test.to(device)
		        CT_test = img_test[:,0:1,:,:,:]
		        WSI_test = img_test[:,1:,:,:,:]
		        label_test = label_test.to(device)
                
		        tCT1, tCT2, tCT3, tCT4, CT_test_out = model_CT(CT_test)
		        tWSI1, tWSI2, tWSI3, tWSI4, WSI_test_out = model_WSI(WSI_test)
		        tMM1, tMM2, tMM3 = model_MEM(tWSI1, tWSI2, tWSI3, label_test)
#		        tMM1, tMM2, tMM3 = model_MEM(tWSI1, tWSI2, tWSI3, label_test, update=False)
		        tFS1, tFS2, tFS3, FUS_test_out = model_FUS(tCT1, tCT2, tCT3, tWSI1, tWSI2, tWSI3)
		        
		        tsmpre = F.softmax(CT_test_out, dim=1)
		        tS1, tS2, tS3, STU_test_out = S_model(CT_test, tMM1, tMM2, tMM3, tsmpre)
                
		        _, pre_CT_test = torch.max(CT_test_out.data, 1)     # 行最大值的索引
		        _, pre_WSI_test = torch.max(WSI_test_out.data, 1)   # 行最大值的索引
		        _, pre_FUS_test = torch.max(FUS_test_out.data, 1)   # 行最大值的索引
		        _, pre_STU_test = torch.max(STU_test_out.data, 1)   # 行最大值的索引
                
		        total_test += label_test.size(0)
		        correct_test_CT += (pre_CT_test == label_test).sum().item()
		        correct_test_WSI += (pre_WSI_test == label_test).sum().item()
		        correct_test_FUS += (pre_FUS_test == label_test).sum().item()
		        correct_test_STU += (pre_STU_test == label_test).sum().item()
        
        # 可视化测试准确率        
		test_acc_CT = 100.0*correct_test_CT/total_test
		test_acc_WSI = 100.0*correct_test_WSI/total_test
		test_acc_FUS = 100.0*correct_test_FUS/total_test
		test_acc_STU = 100.0*correct_test_STU/total_test
		print('Test_ACC_CT={}'.format(test_acc_CT))
		print('Test_ACC_WSI={}'.format(test_acc_WSI))
		print('Test_ACC_FUS={}'.format(test_acc_FUS))
		print('Test_ACC_STU={}'.format(test_acc_STU))
		print()
        
		# 可视化最高准确率
		if test_acc_CT > best_test_acc_CT:
		    best_test_acc_CT = test_acc_CT
		    print('Best_CT_ACC={}'.format(best_test_acc_CT))  
		else:
		    print('Best_CT_ACC={}'.format(best_test_acc_CT)) 
            
		if test_acc_WSI > best_test_acc_WSI:
		    best_test_acc_WSI = test_acc_WSI
		    print('Best_WSI_ACC={}'.format(best_test_acc_WSI))  
		else:
		    print('Best_WSI_ACC={}'.format(best_test_acc_WSI)) 
            
		if test_acc_FUS > best_test_acc_FUS:
		    best_test_acc_FUS = test_acc_FUS
		    print('Best_FUS_ACC={}'.format(best_test_acc_FUS))  
		else:
		    print('Best_FUS_ACC={}'.format(best_test_acc_FUS)) 
            
		if test_acc_STU > best_test_acc_STU:
		    torch.save(S_model, '')
		    best_test_acc_STU = test_acc_STU
		    print('Best_STU_ACC={}'.format(best_test_acc_STU))  
		else:
		    print('Best_STU_ACC={}'.format(best_test_acc_STU)) 
        
        
		S_model.train()
		total_time = time.time() - start_time
		print('Total-Time: {:.6f} '.format(total_time)) 




