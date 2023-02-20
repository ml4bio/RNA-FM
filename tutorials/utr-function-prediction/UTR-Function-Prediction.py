#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
#sys.path.append('./')
#sys.path.append('../')
#sys.path.append('../..')

import os
import pandas as pd
from sklearn import preprocessing
import string
from typing import Sequence, Tuple, List, Union
from tqdm import tqdm
import fm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import numpy as np
import random
import pdb

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(2021)


# ## 1. Load Model

# ### (1) define utr_function_predictor

# In[2]:


class Human5PrimeUTRPredictor(torch.nn.Module):
    """
    contact predictor with inner product
    """
    def __init__(self, alphabet=None, task="rgs", arch="cnn", input_types=["seq", "emb-rnafm"]):
        """
        :param depth_reduction: mean, first
        """       
        super().__init__()     
        self.alphabet = alphabet   # backbone alphabet: pad_idx=1, eos_idx=2, append_eos=True, prepend_bos=True
        self.task = task
        self.arch = arch
        self.input_types = input_types        
        self.padding_mode = "right"
        self.token_len = 100
        self.out_plane = 1
        self.in_channels = 0
        if "seq" in self.input_types:
            self.in_channels = self.in_channels + 4

        if "emb-rnafm" in self.input_types:
            self.reductio_module = nn.Linear(640, 32)
            self.in_channels = self.in_channels + 32  

        if self.arch == "cnn" and self.in_channels != 0:
            self.predictor = self.create_1dcnn_for_emd(in_planes=self.in_channels, out_planes=1)
        else:
            raise Exception("Wrong Arch Type")

    def forward(self, tokens, inputs):
        ensemble_inputs = []
        if "seq" in self.input_types:
            # padding one-hot embedding            
            nest_tokens = (tokens[:, 1:-1] - 4)   # covert token for RNA-FM (20 tokens) to nest version (4 tokens A,U,C,G)
            nest_tokens = torch.nn.functional.pad(nest_tokens, (0, self.token_len - nest_tokens.shape[1]), value=-2)
            token_padding_mask = nest_tokens.ge(0).long()
            one_hot_tokens = torch.nn.functional.one_hot((nest_tokens * token_padding_mask), num_classes=4)
            one_hot_tokens = one_hot_tokens.float() * token_padding_mask.unsqueeze(-1)            
            # reserve padded one-hot embedding
            one_hot_tokens = one_hot_tokens.permute(0, 2, 1)  # B, L, 4
            ensemble_inputs.append(one_hot_tokens)

        if "emb-rnafm" in self.input_types:
            embeddings = inputs["emb-rnafm"]
            # padding RNA-FM embedding
            embeddings, padding_masks = self.remove_pend_tokens_1d(tokens, embeddings)  # remove auxiliary tokens
            batch_size, seqlen, hiddendim = embeddings.size()
            embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, self.token_len - embeddings.shape[1]))            
            # channel reduction
            embeddings = self.reductio_module(embeddings)
            # reserve padded RNA-FM embedding
            embeddings = embeddings.permute(0, 2, 1)
            ensemble_inputs.append(embeddings)        

        ensemble_inputs = torch.cat(ensemble_inputs, dim=1)        
        output = self.predictor(ensemble_inputs).squeeze(-1)
        return output
 
    def create_1dcnn_for_emd(self, in_planes, out_planes):
        main_planes = 64
        dropout = 0.2
        emb_cnn = nn.Sequential(
            nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1), 
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),  
            ResBlock(main_planes * 1, main_planes * 1, stride=2, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d), 
            ResBlock(main_planes * 1, main_planes * 1, stride=1, dilation=1, conv_layer=nn.Conv1d,
                     norm_layer=nn.BatchNorm1d),       
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(main_planes * 1, out_planes),
        )
        return emb_cnn
    
    def remove_pend_tokens_1d(self, tokens, seqs):
        padding_masks = tokens.ne(self.alphabet.padding_idx)

        # remove eos token  （suffix first）
        if self.alphabet.append_eos:     # default is right
            eos_masks = tokens.ne(self.alphabet.eos_idx)
            eos_pad_masks = (eos_masks & padding_masks).to(seqs)
            seqs = seqs * eos_pad_masks.unsqueeze(-1)
            seqs = seqs[:, ..., :-1, :]
            padding_masks = padding_masks[:, ..., :-1]

        # remove bos token
        if self.alphabet.prepend_bos:    # default is left
            seqs = seqs[:, ..., 1:, :]
            padding_masks = padding_masks[:, ..., 1:]

        if not padding_masks.any():
            padding_masks = None

        return seqs, padding_masks

class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        dilation=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
    ):
        super(ResBlock, self).__init__()        
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False)       
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, padding=dilation, bias=False)

        if stride > 1 or out_planes != in_planes: 
            self.downsample = nn.Sequential(
                conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_planes),
            )
        else:
            self.downsample = None
            
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if isinstance(m.bias, nn.Parameter):
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# ### (2) create RNA-FM backbone

# In[3]:


device="cuda"   # "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2] #"1"
#backbone, alphabet = fm.pretrained.esm1b_rna_t12("../../redevelop/pretrained/RNA-FM_pretrained.pth")
#backbone, alphabet = fm.pretrained.esm1b_rna_t12("/home/chenjiayang/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth")
#backbone, alphabet = fm.pretrained.esm1b_rna_t12("/mnt/nas/user/chenjiayang/RNA-FM-github-final-model/RNA-FM_pretrained.pth")
backbone, alphabet = fm.pretrained.rna_fm_t12()
backbone.to(device)
print("create RNA-FM_backbone sucessfully")
print(backbone)


# ### (3) create UTR function downstream predictor

# In[4]:


task="rgs"
arch="cnn"
input_items = sys.argv[1].split("_")   #["seq","emb-rnafm"]   # ["seq"], ["emb-rnafm"], ["seq","emb-rnafm"]
model_name = arch.upper() + "_" + "_".join(input_items) 
utr_func_predictor = Human5PrimeUTRPredictor(
    alphabet, task=task, arch=arch, input_types=input_items    
)
utr_func_predictor.apply(weights_init)
utr_func_predictor.to(device)
print("create utr_func_predictor sucessfully")
print(utr_func_predictor)


# ### (4) define loss function and optimizer

# In[5]:


criterion = nn.MSELoss(reduction="none")
optimizer = optim.Adam(utr_func_predictor.parameters(), lr=0.001)


# ## 2. Load Data
# You should download data from gdrive link: https://drive.google.com/file/d/10zCfOHOaOa__J2AIuZyidZ9sVJ9L11rI/view?usp=sharing and place it in the tutorials/utr-function-prediction

# ### (1) define utr dataset

# In[6]:


class Human_5Prime_UTR_VarLength(object):
    def __init__(self, root, set_name="train"):
        """
        :param root: root path of dataset - CATH. however not all of stuffs under this root path
        :param data_type: seq, msa
        :param label_type: 1d, 2d
        :param set_name: "train", "valid", "test"
        """
        self.root = root
        self.set_name = set_name
        self.src_scv_path = os.path.join(self.root, "data", "GSM4084997_varying_length_25to100.csv")
        self.seqs, self.scaled_rls = self.__dataset_info(self.src_scv_path)

    def __getitem__(self, index):
        seq_str = self.seqs[index].replace("T", "U")
        label = self.scaled_rls[index]

        return seq_str, label

    def __len__(self):
        return len(self.seqs)

    def __dataset_info(self, src_csv_path):
        # 1.Filter Data
        # (1) Random Set
        src_df = pd.read_csv(src_csv_path)
        src_df.loc[:, "ori_index"] = src_df.index
        random_df = src_df[src_df['set'] == 'random']
        ## Filter out UTRs with too few less reads
        random_df = random_df[random_df['total_reads'] >= 10]    # 87000 -> 83919             
        random_df.sort_values('total_reads', inplace=True, ascending=False)
        random_df.reset_index(inplace=True, drop=True)

        # (2) Human Set
        human_df = src_df[src_df['set'] == 'human']
        ## Filter out UTRs with too few less reads
        human_df = human_df[human_df['total_reads'] >= 10]   # 16739 -> 15555             
        human_df.sort_values('total_reads', inplace=True, ascending=False)
        human_df.reset_index(inplace=True, drop=True)       

        # 2.Split Dataset
        # (1) Generate Random Test set
        random_df_test = pd.DataFrame(columns=random_df.columns)
        for i in range(25, 101):
            tmp = random_df[random_df['len'] == i].copy()
            tmp.sort_values('total_reads', inplace=True, ascending=False)
            tmp.reset_index(inplace=True, drop=True)
            random_df_test = random_df_test.append(tmp.iloc[:100])
        
        # (2) Generate Human Test set
        human_df_test = pd.DataFrame(columns=human_df.columns)
        for i in range(25, 101):
            tmp = human_df[human_df['len'] == i].copy()
            tmp.sort_values('total_reads', inplace=True, ascending=False)
            tmp.reset_index(inplace=True, drop=True)
            human_df_test = human_df_test.append(tmp.iloc[:100])            
        
        # (3) Exclude Test data from Training data
        train_df = pd.concat([random_df, random_df_test]).drop_duplicates(keep=False)  #  76319        
        
        # 3.Label Normalization (ribosome loading value)
        label_col = 'rl'
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(train_df.loc[:, label_col].values.reshape(-1, 1))
        train_df.loc[:,'scaled_rl'] = self.scaler.transform(train_df.loc[:, label_col].values.reshape(-1, 1))
        random_df_test.loc[:, 'scaled_rl'] = self.scaler.transform(random_df_test.loc[:, label_col].values.reshape(-1, 1))
        human_df_test.loc[:, 'scaled_rl'] = self.scaler.transform(human_df_test.loc[:, label_col].values.reshape(-1, 1))

        # 4.Pickup Target Set
        if self.set_name == "train":
            set_df = train_df
        elif self.set_name == "valid":
            set_df = random_df_test
        else:
            set_df = human_df_test 
        seqs = set_df['utr'].values
        scaled_rls = set_df['scaled_rl'].values 
        names = set_df["ori_index"].values       

        print("Num samples of {} dataset: {} ".format(self.set_name, set_df["len"].shape[0]))
        return seqs, scaled_rls

# covert tokens of different length to a batch tensor with the same length for each sample.
def generate_token_batch(alphabet, seq_strs):
    batch_size = len(seq_strs)
    max_len = max(len(seq_str) for seq_str in seq_strs)
    tokens = torch.empty(
        (
            batch_size,
            max_len
            + int(alphabet.prepend_bos)
            + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    for i, seq_str in enumerate(seq_strs):              
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor([alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
        tokens[i, int(alphabet.prepend_bos): len(seq_str)+ int(alphabet.prepend_bos),] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_str) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    return tokens
    
def collate_fn(batch):
    seq_strs, labels = zip(*batch)
    tokens = generate_token_batch(alphabet, seq_strs)
    labels = torch.Tensor(labels)    
    return seq_strs, tokens, labels    


# ### (2) generate dataloaders

# In[7]:


root_path = "./"
train_set =  Human_5Prime_UTR_VarLength(root=root_path, set_name="train")
val_set =  Human_5Prime_UTR_VarLength(root=root_path, set_name="valid")
test_set =  Human_5Prime_UTR_VarLength(root=root_path, set_name="test")

num_workers = 0
batch_size = 64

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, collate_fn=collate_fn, drop_last=False
)

val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, collate_fn=collate_fn, drop_last=False
)

test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, collate_fn=collate_fn, drop_last=False
)

scaler = train_set.scaler


# ## 3. Training Model

# ### (1) define eval function

# In[8]:


def model_eval(data_loader, i_epoch, set_name="unknown"):
    all_losses = []
    true_rl_mses = []
    for index, (seq_strs, tokens, labels) in enumerate(data_loader):
        backbone.eval()
        utr_func_predictor.eval()
        tokens = tokens.to(device)
        labels = labels.to(device)
        with torch.no_grad():             
            inputs = {}
            results = {}
            if "emb-rnafm" in input_items:
                results = backbone(tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)
                inputs["emb-rnafm"] = results["representations"][12] 
            results["rl"] = utr_func_predictor(tokens, inputs)        
            losses = criterion(results["rl"], labels)  
            all_losses.append(losses.detach().cpu())    
            
            # true value metric
            pds = scaler.inverse_transform(results["rl"].detach().cpu().numpy())
            gts = scaler.inverse_transform(labels.detach().cpu().numpy())
            true_rl_mse = criterion(torch.Tensor(pds), torch.Tensor(gts))  
            true_rl_mses.append(true_rl_mse.detach().cpu())  

    avg_loss = torch.cat(all_losses, dim=0).mean()
    avg_true_rl_mses = torch.cat(true_rl_mses, dim=0).mean()
    print("Epoch {}, Evaluation on {} Set - MSE loss: {:.3f}".format(i_epoch, set_name, avg_loss))
    print("Epoch {}, Evaluation on {} Set - True MSE: {:.3f}".format(i_epoch, set_name, avg_true_rl_mses))
    
    return avg_loss


# ### (2) training process

# In[9]:


n_epoches = 50
best_mse = 10
best_epoch = 0

for i_e in range(1, n_epoches+1):
    all_losses = []
    n_sample = 0
    n_iter = len(train_loader)

    pbar = tqdm(train_loader, desc="Epoch {}, Train Set - MSE loss: {}".format(i_e, "NaN"), ncols=100)
    for index, (seq_strs, tokens, labels) in enumerate(pbar):
        backbone.eval()
        utr_func_predictor.train()
        tokens = tokens.to(device)
        labels = labels.to(device)      
        
        inputs = {}
        results = {}  
        if "emb-rnafm" in  input_items:            
            with torch.no_grad():
                results = backbone(tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)            
            inputs["emb-rnafm"] = results["representations"][12]                
        results["rl"] = utr_func_predictor(tokens, inputs)
        losses = criterion(results["rl"], labels)
        batch_loss = losses.mean()
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        all_losses.append(losses.detach().cpu())
        current_avg_loss = torch.cat(all_losses, dim=0).mean()
        
        pbar.set_description("Epoch {}, Train Set - MSE loss: {:.3f}".format(i_e, current_avg_loss))
    
    random_mse = model_eval(val_loader, i_e, set_name="Random")
    
    if random_mse < best_mse:
        best_epoch = i_e
        best_mse = random_mse
        torch.save(utr_func_predictor.state_dict(), "result/{}_best_utr_predictor.pth".format(model_name))
    print("--------- Model: {}, Best Epoch {}, Best MSE {:.3f}".format(model_name, best_epoch, best_mse))


# In[ ]:




