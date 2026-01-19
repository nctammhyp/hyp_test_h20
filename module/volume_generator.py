# volume_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out=1, use_sigmoid=True): # Thêm use_sigmoid
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Conv3d(ch_hid, ch_out, (1, 1, 1))
        self.use_sigmoid = use_sigmoid
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        if self.use_sigmoid:
            x = self.out_act(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        ch_in = opts.base_channel
        
        # SỬA TẠI ĐÂY: 
        # 1. ch_out phải là ch_in (base_channel)
        # 2. use_sigmoid=False vì đây là đặc trưng (feature), không phải trọng số (weight)
        self.reference_mapping = MLP(ch_in + 2, ch_in, ch_out=ch_in, use_sigmoid=False) 
        
        # Target mapping vẫn để ch_out=1 và use_sigmoid=True (mặc định) để tạo trọng số kết hợp 2 ảnh
        self.target_mapping = MLP(2 * ch_in + 4, ch_in, ch_out=1)

    def forward(self, spherical_feats):
        f0, f1, f2 = spherical_feats[0], spherical_feats[1], spherical_feats[2]
        g0, g1, g2 = spherical_feats[3], spherical_feats[4], spherical_feats[5]

        # Tạo Reference Feat (từ Cam 0) -> Bây giờ sẽ trả về đúng base_channel (ví dụ: 4)
        reference_feat = self.reference_mapping(torch.cat([f0, g0], dim=1))

        # Tạo Target Feat (Kết hợp Cam 1 và Cam 2 bằng trọng số)
        target_input = torch.cat([f1, f2, g1, g2], dim=1)
        target_weight = self.target_mapping(target_input)
        target_feat = target_weight * f1 + (1 - target_weight) * f2

        context_feat = reference_feat
        return [reference_feat, target_feat], context_feat