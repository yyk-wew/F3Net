from xception import Xception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import types

class F3Net(nn.Module):
    def __init__(self, num_classes=1, img_width=299, img_height=299, LFS_window_size=10, LFS_stride=2, LFS_M = 6, mode='FAD', device=None):
        super(F3Net, self).__init__()
        # hyper-parameter
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self._LFS_M = LFS_M
        self.mode = mode
        self.window_size = LFS_window_size
        self.device = device
        
        # init network
        if mode == 'Both' or mode == 'FAD' or mode == 'Mix':
            self._init_FAD_branch(img_size)
        if mode == 'Both' or mode == 'LFS' or mode == 'Mix':
            self._init_LFS_branch(LFS_window_size, LFS_stride, LFS_M)
        if mode == 'Mix':
            self._init_MixBlock()


        # some layers
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096 if self.mode == 'Both' or self.mode == 'Mix' else 2048, num_classes)
        self.dp = nn.Dropout(p=0.2)


    def _init_FAD_branch(self, size):
        self.xcep_FAD = Xception(self.num_classes)

        # modify layers
        self.xcep_FAD.conv1 = nn.Conv2d(9, 32, 3, 2, 0, bias=False)

        # init DCT matrix
        self._DCT_all = torch.tensor(DCT_mat(size), requires_grad=False).float().cuda()
        self._DCT_all_T = torch.transpose(self._DCT_all, 0, 1)

        # init base filter
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        base_low = [[0. if i + j > size // 16 else 1. for j in range(size)] for i in range(size)]
        f_base_low = torch.tensor(base_low, requires_grad=False).cuda()
        base_middle = [[0. if i + j > size // 8 or i + j <= size // 16 else 1. for j in range(size)] for i in range(size)]
        f_base_middle = torch.tensor(base_middle, requires_grad=False).cuda()
        base_high = [[0. if i + j <= size // 8 else 1. for j in range(size)] for i in range(size)]
        f_base_high = torch.tensor(base_high, requires_grad=False).cuda()
        self.F_base_filters = [f_base_low, f_base_middle, f_base_high]
        # self.F_base_filters = [f_base_high]

        # init learnable filter
        f_w_low = nn.Parameter(torch.Tensor(size, size).cuda())
        f_w_middle = nn.Parameter(torch.Tensor(size, size).cuda())
        f_w_high = nn.Parameter(torch.Tensor(size, size).cuda())
        self.F_w_filters = [f_w_low, f_w_middle, f_w_high]
        # self.F_w_filters = [f_w_high]

    def _FAD(self, x):
        self._F_all_filters = [b_filter + self._norm_sigma(w_filter) for b_filter, w_filter in zip(self.F_base_filters, self.F_w_filters)]
        # shape of x: [3, w, h]
        # x = dct.dct_2d(x, norm='ortho') #[3, w, h]
        x_freq = self._DCT_all @ x @ self._DCT_all_T     #[3, w, h]
        masks = self._F_all_filters  # 3 masks
        x_freq_list = [x_freq * m for m in masks]  # [3 * [3, w, h]]
        # y = [dct.idct_2d(x) for x in y] # [3 * [3, w, h]]
        y = [self._DCT_all_T @ one_freq @ self._DCT_all for one_freq in x_freq_list] # [3 * [3, w, h]]
        y = torch.cat(y, dim=1) # [9, w, h]
        return y

    def _init_LFS_branch(self, window_size, stride, M):
        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)
        self.xcep_LFS = Xception(self.num_classes)

        # modify conv1, stride = 1, padding=1 (same)
        self.xcep_LFS.conv1 = nn.Conv2d(M, 32, 3, 1, 1, bias=False)

        # init DCT mat
        self._DCT_patch = torch.tensor(DCT_mat(window_size), requires_grad=False).float().cuda()
        self._DCT_patch_T = torch.transpose(self._DCT_patch, 0, 1)

        # init base filter
        base_filters = [[[1. if i + j >= window_size / M * z and i + j < window_size / M * (z + 1) else 0. for j in range(window_size)] for i in range(window_size)] for z in range(M)]
        mat_r_d = [[1. if i + j >= window_size else 0. for j in range(window_size)] for i in range(window_size)]
        self.L_base_filters = [torch.tensor(b_filter, requires_grad=False).cuda() for b_filter in base_filters]
        self.L_base_filters[-1]  = self.L_base_filters[-1] + torch.tensor(mat_r_d, requires_grad=False).cuda()

        # init learnable filter
        self.L_w_filters = [nn.Parameter(torch.Tensor(window_size, window_size).cuda()) for i in range(M)]


    def _LFS(self, x):
        self._L_all_filters = [b_filter + self._norm_sigma(w_filter) for b_filter, w_filter in zip(self.L_base_filters, self.L_w_filters)]
        # shape of x: [N, 3, w, h]
        kernels = self._L_all_filters   # M masks
        fea_list = []
        
        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        assert size_after == 149
        
        # sliding window
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        for k in kernels:
            y = x_dct * k
            y = torch.abs(y)
            y = torch.sum(y, dim=[2,3,4])   # [N, L]
            y = torch.log10(y)
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)   # [N, 1, 149, 149]
            fea_list.append(y)
        fea_LFS = torch.cat(fea_list, dim=1)    # [N, M, 149, 149]
        return fea_LFS

    def _init_MixBlock(self):
        self.m_block1 = MixBlock(728)
        self.m_block2 = MixBlock(2048)

        self.xcep_FAD.fea_0_7 = types.MethodType(fea_0_7, self.xcep_FAD)
        self.xcep_FAD.fea_8_12 = types.MethodType(fea_8_12, self.xcep_FAD)

        self.xcep_LFS.fea_0_7 = types.MethodType(fea_0_7, self.xcep_LFS)
        self.xcep_LFS.fea_8_12 = types.MethodType(fea_8_12, self.xcep_LFS)


    def load_pretrained_xception(self, pretrained_path='pretrained/xception-b5690688.pth'):
        # load model
        state_dict = torch.load(pretrained_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        # print(state_dict)

        if self.mode == 'Both' or self.mode == 'FAD' or self.mode == 'Mix':
            # restore layer to load pre-trained model
            self.xcep_FAD.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
            self.xcep_FAD.load_state_dict(state_dict, False)
            
            # modify layer again and do initialization
            self.xcep_FAD.conv1 = nn.Conv2d(9, 32, 3, 2, 0, bias=False)
            self.xcep_FAD.conv1.weight.data.normal_(0.0,0.02)

        # LFS part
        if self.mode == 'Both' or self.mode == 'LFS' or self.mode == 'Mix':
            self.xcep_LFS.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
            self.xcep_LFS.load_state_dict(state_dict, False)

            self.xcep_LFS.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 1, bias=False)
            self.xcep_LFS.conv1.weight.data.normal_(0.0,0.02)

        # init filters
        self.init_filters()

    
    def init_filters(self):
        if self.mode == 'Both' or self.mode == 'LFS' or self.mode == 'Mix':
            for f in self.L_w_filters:
                f.data.zero_()
        if self.mode == 'Both' or self.mode == 'FAD' or self.mode == 'Mix':
            for f in self.F_w_filters:
                f.data.zero_()
        if self.mode == 'Mix':
            self.m_block1.conv1.weight.data.normal_(0.0, 0.02)
            self.m_block1.conv2.weight.data.normal_(0.0, 0.02)
            self.m_block2.conv1.weight.data.normal_(0.0, 0.02)
            self.m_block2.conv2.weight.data.normal_(0.0, 0.02)



    def forward(self, x):
        if self.mode == 'Mix':
            return self.forward_mix(x)

        if self.mode == 'Both' or self.mode == 'FAD':  
            fea_FAD = self._FAD(x)
            fea_FAD = self.xcep_FAD.features(fea_FAD)
            fea_FAD = self._norm_fea(fea_FAD)
        
        if self.mode == 'Both' or self.mode == 'LFS': 
            # print("x:")
            # print(x)
            fea_LFS = self._LFS(x)
            # print("fea_LFS:")
            # print(fea_LFS)
            fea_LFS = self.xcep_LFS.features(fea_LFS)
            fea_LFS = self._norm_fea(fea_LFS)
        # print(fea_LFS)
        # raise RuntimeError()

        if self.mode == 'Both':
            y = torch.cat((fea_FAD, fea_LFS), dim=1)
        if self.mode == 'FAD':
            y = fea_FAD
        if self.mode == 'LFS':
            y = fea_LFS

        f = self.dp(y)
        f = self.fc(f)
        return y,f

    def forward_mix(self, x):
        fea_FAD = self._FAD(x)
        fea_FAD = self.xcep_FAD.fea_0_7(fea_FAD)
        
        fea_LFS = self._LFS(x)
        fea_LFS = self.xcep_LFS.fea_0_7(fea_LFS)

        att1_1, att2_1 = self.m_block1(fea_FAD, fea_LFS)
        fea_FAD_1 = fea_LFS * att1_1 + fea_FAD
        fea_LFS_1 = fea_FAD * att2_1 + fea_LFS

        fea_FAD_1 = self.xcep_FAD.fea_8_12(fea_FAD_1)
        fea_LFS_1 = self.xcep_LFS.fea_8_12(fea_LFS_1)

        att1_2, att2_2 = self.m_block2(fea_FAD_1, fea_LFS_1)
        fea_FAD_2 = fea_LFS_1 * att1_2 + fea_FAD_1
        fea_LFS_2 = fea_FAD_1 * att2_2 + fea_LFS_1

        fea_FAD_2 = self._norm_fea(fea_FAD_2)
        fea_LFS_2 = self._norm_fea(fea_LFS_2)

        y = torch.cat((fea_FAD_2, fea_LFS_2), dim=1)
        f = self.dp(y)
        f = self.fc(f)
        return y, f

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f

    def _norm_sigma(self, x):
        return (1. - torch.exp(-x)) / (1. + torch.exp(-x))

    def _norm_sigma_new(self, x):
        res = torch.where(x >= 0, (1. - torch.exp(-x)) / (1. + torch.exp(-x)), (torch.exp(x) - 1) / (1. + torch.exp(x)))
        return res


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m
    

# overwrite method for xception in LFS branch
# plan A

def new_xcep_features(self, input):
    # x = self.conv1(input)
    # x = self.bn1(x)
    # x = self.relu(x)

    x = self.conv2(input)   # input :[149, 149, 6]  conv2:[in_filter:32]
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

# function for mix block

def fea_0_7(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    return x

def fea_8_12(self, x):
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

class MixBlock(nn.Module):
    def __init__(self, c_in):
        super(MixBlock, self).__init__()
        c_in = c_in * 2
        self.conv1 = nn.Conv2d(c_in, c_in, (1,1))
        self.conv2 = nn.Conv2d(c_in, 2, (3,3), 1, 1)
        self.bn = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        y1, y2 = torch.split(x, 1, dim=1)
        return y1, y2
