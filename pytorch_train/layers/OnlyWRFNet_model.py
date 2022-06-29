import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D

class Attention_model(nn.Module):
    def __init__(self, channels):
        super(Attention_model, self).__init__()
        self.channels = channels
        self.DWCNN_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, groups=channels),
        )
        self.CNN_hw = nn.Conv2d(128 * 2, 1, kernel_size=1, stride=1)


    def forward(self, wrf, h, c):
        att_wrf = self.DWCNN_att(wrf)
        att_hc = self.CNN_hw(torch.cat([h, c], dim=1))
        att = torch.mul(att_hc, att_wrf)
        e = torch.div(torch.sum(att, dim=[2, 3], keepdim=True), att_wrf.shape[3])
        alpha = torch.softmax(e, dim=1)
        att_res = torch.mul(wrf, alpha)
        return att_res



class OnlyWRFNet_Model(nn.Module):
    def __init__(self, wrf_tra_frames, wrf_channels, config_dict):
        super(OnlyWRFNet_Model, self).__init__()
        self.config_dict = config_dict
        self.wrf_tra_frames = wrf_tra_frames
        self.wrf_batchnorm = nn.BatchNorm3d(wrf_channels)
        self.CNN_module2 = nn.Sequential(
            nn.Conv2d(wrf_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder_ConvLSTM = ConvLSTM2D(32, 128, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.CNN_module3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )
        self.attention = Attention_model(wrf_channels)

    def forward(self, wrf, obs):
        # batch normalization
        wrf = wrf.permute(0, 4, 1, 2, 3)
        wrf = self.wrf_batchnorm(wrf)
        wrf = wrf.permute(0, 2, 3, 4, 1)
        
        # wrf : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        wrf = wrf.permute(1, 0, 4, 2, 3).contiguous()

        batch_size = wrf.shape[1]
        pre_frames = torch.zeros([self.wrf_tra_frames, batch_size, 1, wrf.shape[3], wrf.shape[4]]).to(wrf.device)

        h = torch.zeros([batch_size, 128, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(wrf.device)
        c = torch.zeros([batch_size, 128, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(wrf.device)
        for t in range(self.wrf_tra_frames):
            wrf_att = self.attention(wrf[t], h, c)
            wrf_encoder = self.CNN_module2(wrf_att)
            h, c = self.decoder_ConvLSTM(wrf_encoder, h, c)
            pre = self.CNN_module3(h)
            pre_frames[t] = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        # print(pre_frames.shape)
        return pre_frames

if __name__ == "__main__":
    wrf = torch.zeros(1, 12, 159, 159, 29)
    obs = None
    model = OnlyWRFNet_Model(wrf_tra_frames=12, wrf_channels=29)
    pre_frames = model(wrf, obs)
    print(pre_frames.shape)
