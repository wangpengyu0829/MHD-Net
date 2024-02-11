import os
import time
import glob
import torch
import random
import numpy as np
import myutils as myutils
import torchvision.transforms as transforms

from torch import nn
from PIL import Image
from torch import optim
from model_ori import generate_model
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data',       default='',         type=str, help='')
    parser.add_argument('--sub_folder_jpg',  default='',         type=str, help='')
    parser.add_argument('--batch_size',      default='',         type=int, help='')
    parser.add_argument('--timesteps',       default='',         type=int, help='')
    parser.add_argument('--num_epochs',      default='',         type=int, help='')
    parser.add_argument('--n_classes',       default='',         type=int, help='')
    parser.add_argument('--sample_size',     default='',         type=int, help='')
    parser.add_argument('--sample_duration', default='',         type=int, help='')
    parser.add_argument('--mode',            default='',         type=str, help='Mode (score | feature)')
    parser.add_argument('--model_name',      default='',         type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth',     default=34,         type=int, help='ResNet (18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='',         type=str, help='Shortcut type of resnet (A | B)')
    args = parser.parse_args()
    return args

opt = parse_opts()

np.random.seed()
random.seed()
torch.manual_seed()

mean = []
std  = []

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('USE GPU', device)
else:
    print('USE CPU')

path2ajpgs = os.path.join(opt.path2data, opt.sub_folder_jpg)

all_vids, all_labels, catgs = myutils.get_vids(path2ajpgs)
print(len(all_vids), len(all_labels), len(catgs))

labels_dict = {}
ind = 0
for uc in catgs:
	labels_dict[uc] = ind
	ind += 1
print(labels_dict)

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
train_indx, test_indx = next(sss.split(all_vids, all_labels))

train_ids = [all_vids[ind] for ind in train_indx]
train_labels = [all_labels[ind] for ind in train_indx]
print(len(train_ids), len(train_labels))

test_ids = [all_vids[ind] for ind in test_indx]
test_labels = [all_labels[ind] for ind in test_indx]
print(len(test_ids), len(test_labels))

class VideoDataset(Dataset):
	def __init__(self, ids, labels, transform):
		self.transform = transform
		self.ids = ids
		self.labels = labels
	def __len__(self):
		return len(self.ids)
	def __getitem__(self, idx):
		path2imgs = glob.glob(self.ids[idx]+"/*.png")
		path2imgs = path2imgs[:opt.timesteps]
		label = labels_dict[self.labels[idx]]
		frames = []
		for p2i in path2imgs: 
			frame = Image.open(p2i)
			frames.append(frame)
		frames_tr = []
		for frame in frames:
			frame = self.transform(frame)
			frames_tr.append(frame)
		if len(frames_tr)>0:
			frames_tr = torch.stack(frames_tr) 
		return frames_tr, label


train_transformer = transforms.Compose([transforms.Resize((opt.sample_size, opt.sample_size)),
					                    transforms.RandomHorizontalFlip(p=0.5),
					                    transforms.ToTensor(),
					                    transforms.Normalize(mean,std) ])		

train_ds = VideoDataset(ids=train_ids, labels=train_labels, transform=train_transformer)

test_transformer = transforms.Compose([transforms.Resize((opt.sample_size, opt.sample_size)),
				                       transforms.ToTensor(),
				                       transforms.Normalize(mean,std)])

test_ds = VideoDataset(ids=test_ids, labels=test_labels, transform=test_transformer)


def collate_fn_3dcnn(batch):
	imgs_batch, label_batch = list(zip(*batch)) 
	imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
	label_batch = [torch.tensor(l) for l,imgs in zip(label_batch,imgs_batch) if len(imgs)>0]
	imgs_tensor = torch.stack(imgs_batch)
	imgs_tensor = torch.transpose(imgs_tensor,2,1)
	labels_tensor = torch.stack(label_batch)
	return imgs_tensor, labels_tensor

train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn_3dcnn)
test_dl = DataLoader(test_ds, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn_3dcnn)

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group["lr"]


model = generate_model(opt).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, verbose=1)
os.makedirs("", exist_ok=True)


'''主函数'''
if __name__=='__main__':
	best_test_acc = 0
    
	model.train()
	for epoch in range(opt.num_epochs):
		print()
		start_time = time.time()
		iter_loss = 0.0
		correct_train = 0
		total_train = 0
		correct_test = 0
		total_test = 0
		current_lr = get_lr(optimizer)
		print("Epoch {}/{}, current lr={}".format(epoch+1, opt.num_epochs, current_lr))
        
		for i, (img, label) in enumerate(train_dl):
            
		    img = img.to(device)
		    label = label.to(device)
		    optimizer.zero_grad()
            
		    _, _, _, _, out = model(img)
		    _, pre = torch.max(out.data, 1) # 获取输出的预测类别
 
		    total_train += label.size(0)
		    correct_train += (pre == label).sum().item()
            
		    loss = criterion(out, label)          
		    iter_loss += loss.item()
            
		    loss.backward()   
		    optimizer.step() 
            
		    if i % 30 == 29:
		        print('[%d, %d] loss: %.03f' % (epoch+1, i+1, iter_loss))
		        iter_loss = 0.0

		train_acc = 100.0*correct_train/total_train
		print('Training_ACC={}'.format(train_acc))
		lr_scheduler.step(loss)
            
		model.eval()
		with torch.no_grad():
		    for j, (img_test, label_test) in enumerate(test_dl):
                
		        img_test = img_test.to(device)
		        label_test = label_test.to(device)
            
		        _, _, _, _, out_test = model(img_test)
		        _, pre_test = torch.max(out_test.data, 1)     # 行最大值的索引
		        total_test += label_test.size(0)
		        correct_test += (pre_test == label_test).sum().item()
                
		test_acc = 100.0*correct_test/total_test
		print('Test_ACC={}'.format(test_acc))
		print()
        
		if test_acc > best_test_acc:
		    torch.save(model, '')
		    best_test_acc = test_acc
		    print('Best_test_ACC={}'.format(best_test_acc))  
		else:
		    print('Best_test_ACC={}'.format(best_test_acc)) 
        
		model.train()
		total_time = time.time() - start_time
		print('Total-Time: {:.6f} '.format(total_time)) 




