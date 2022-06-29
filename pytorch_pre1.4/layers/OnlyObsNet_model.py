import torch
import torch.nn as nn
from layers.ConvLSTM import ConvLSTM2D


class OnlyObsNet_Model(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels, pre_frames, config_dict):
        super(OnlyObsNet_Model, self).__init__()
        self.config_dict = config_dict
        self.obs_tra_frames = obs_tra_frames
        self.pre_frames = pre_frames
        self.CNN_module1 = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder_ConvLSTM = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.wrf_batchnorm = nn.BatchNorm3d(obs_channels)
        self.CNN_module2 = nn.Sequential(
            nn.Conv2d(obs_channels, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder_ConvLSTM = ConvLSTM2D(8, 32, kernel_size=5, img_rowcol=(config_dict['GridRowColNum']//2)//2)
        self.CNN_module3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1)
        )

    def forward(self, wrf, obs):
        # obs : [batch_size, frames, x, y, channels] -> [frames, batch_size, channels, x, y]
        obs = obs.permute(1, 0, 4, 2, 3).contiguous()

        batch_size = obs.shape[1]
        pre_frames = torch.zeros([self.pre_frames, batch_size, 1, obs.shape[3], obs.shape[4]]).to(obs.device)

        h = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, (self.config_dict['GridRowColNum']//2)//2, (self.config_dict['GridRowColNum']//2)//2], dtype=torch.float32).to(obs.device)
        for t in range(self.obs_tra_frames):
            obs_encoder = self.CNN_module1(obs[t])
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c)
        h = self.encoder_h(h)
        c = self.encoder_c(c)
        last_res = obs[-1]
        for t in range(self.pre_frames):
            pre_encoder = self.CNN_module2(last_res)
            h, c = self.decoder_ConvLSTM(pre_encoder, h, c)
            pre = self.CNN_module3(h)
            pre_frames[t] = pre
            last_res = pre
        pre_frames = pre_frames.permute(1, 0, 3, 4, 2).contiguous()
        # print(pre_frames.shape)
        return pre_frames

if __name__ == "__main__":
    obs = torch.zeros(1, 3, 159, 159, 1)
    model = OnlyObsNet_Model(obs_tra_frames=3, obs_channels=1, pre_frames=12)
    pre_frames = model(None, obs)
    print(pre_frames.shape)

