import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(MCNNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder1_mid = nn.Conv3d(32, 32, 3, stride=1, padding=1)  # b, 16, 10, 10        
        
        self.encoder2=   nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder2_mid = nn.Conv3d(64, 64, 3, stride=1, padding=1)  # b, 16, 10, 10  
        self.encoder3=   nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder3_mid = nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.encoder4_mid = nn.Conv3d(256, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder2_mid =   nn.Conv3d(128, 128, 3, stride=1, padding=1)  # b, 8, 15, 1        
        self.decoder3 =   nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder3_mid =   nn.Conv3d(64, 64, 3, stride=1, padding=1)  # b, 8, 15, 1      
        self.decoder4 =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder4_mid =   nn.Conv3d(32, 32, 3, stride=1, padding=1)  # b, 8, 15, 1      
        self.decoder5 =   nn.Conv3d(32, 16, 3, stride=1, padding=1)
        self.decoder5_mid =   nn.Conv3d(16, 16, 3, stride=1, padding=1)  # b, 8, 15, 1      

        # 256*256*48 尺度下的映射
        self.map5 = nn.Sequential(
            nn.Conv3d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # 128*128*24 尺度下的映射
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # 64*64*12 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 5, stride=1, padding=2),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # 32*32*6 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, out_channel, 5, stride=1, padding=2),
            nn.Sigmoid()
        )
        # 16*16*3 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, out_channel, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x),2,2))
        out = F.relu(self.encoder1_mid(out))
        out = F.relu(self.encoder1_mid(out))     
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out),2,2))
        out = F.relu(self.encoder2_mid(out))
        out = F.relu(self.encoder2_mid(out))  
        t2 = out

        out = F.relu(F.max_pool3d(self.encoder3(out),2,2))
        out = F.relu(self.encoder3_mid(out))
        out = F.relu(self.encoder3_mid(out))  
        t3 = out

        out = F.relu(F.max_pool3d(self.encoder4(out),2,2))
        out = F.relu(self.encoder4_mid(out))
        out = F.relu(self.encoder4s_mid(out))  
        t4 = out


        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = F.relu(self.decoder2_mid(out))     
        out = F.relu(self.decoder2_mid(out))        
        out = torch.add(out,t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = F.relu(self.decoder3_mid(out))     
        out = F.relu(self.decoder3_mid(out))     
        out = torch.add(out,t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = F.relu(self.decoder4_mid(out))             
        out = F.relu(self.decoder4_mid(out))     
        out = torch.add(out,t1)
        output4 = self.map4(out)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = F.relu(self.decoder5_mid(out))             
        out = F.relu(self.decoder5_mid(out))    
        output5 = self.map5(out)
        #print(output1.shape)
        #print(output2.shape)
        #print(output3.shape)
        #print(output4.shape)
        #print(output5.shape)

        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return [output1, output2, output3, output4, output5]
        else:
            return output5