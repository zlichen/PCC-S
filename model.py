import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Refine_Model(nn.Module):
    def __init__(self, block_size=9, out_dim = 3):
        super(Refine_Model, self).__init__()
        self.Conv_block = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(256+4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self,octree_info,block):
        out = self.Conv_block(block).squeeze()
        out = out.view(octree_info.shape[0], -1)
        out = torch.cat((out, octree_info), dim=1)
        out = self.linear_block(out)
        return out



class EntropyModel(nn.Module):
    def __init__(self, block_size=9, surface_num = 1,surface_params_num = 6,class_num = 256):
        super(EntropyModel, self).__init__()
        self.surface_num = surface_num
        self.surface_params_num = surface_params_num
        self.Conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )

        self.Conv_block_2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=96, kernel_size=3),
        )
        """
        par features
        """
        self.linear_block_down = nn.Sequential(
            nn.Linear(96+4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.linear_block_up = nn.Linear(32, 96)

        """
        Surface features
        """
        self.Conv_block_surface = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3),
        )

        self.linear_block_plane_1 = nn.Sequential(
            nn.Linear(32+4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.linear_block_plane_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, self.surface_params_num),
        )
        """
        Sibling features
        """
        self.Conv_hr_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        self.Conv_hr_block_2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.Conv_hr_block_3 = nn.Sequential(
            nn.Conv3d(in_channels=16+32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4,padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=32, kernel_size=3),
        )

        """
        Fuse features
        """
        self.linear_block_fuse = nn.Sequential(
            nn.Linear(96 + 32 + 32 + 32 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, class_num),
        )

    def forward(self, feats_prev, octree_node_this, block_this, hr_blocks):
        """

        Args:
            feats_prev: torch.Size([N, 32])
            octree_node_this: torch.Size([N, 4])
            block_this: torch.Size([N, 1, 9, 9, 9])
            hr_blocks: torch.Size([N, 1, 4, 4, 4])

        Returns:

        """
        low_feats = self.Conv_block_1(block_this)
        high_feats = self.Conv_block_2(low_feats).squeeze()
        high_feats = high_feats.view(octree_node_this.shape[0], -1)
        feats_cat = torch.cat((high_feats, octree_node_this), dim=1)
        par_feats = self.linear_block_down(feats_cat)

        feats_this = F.relu(self.linear_block_up(par_feats) + high_feats)

        """Surface Feats Extraction"""
        surface_feats = self.Conv_block_surface(low_feats).squeeze()
        surface_feats = surface_feats.view(octree_node_this.shape[0], -1)
        surface_feats = torch.cat((surface_feats, octree_node_this), dim=1)

        """Surface params"""
        surface_feats = self.linear_block_plane_1(surface_feats)
        if self.training:
            out_plane_params = self.linear_block_plane_2(surface_feats)

        """
        Sibing Feature Extraction
        """
        out_hr_1 = self.Conv_hr_block_1(hr_blocks)
        out_hr_2 = self.Conv_hr_block_2(out_hr_1)
        out_hr_3 = self.Conv_hr_block_3(torch.cat((out_hr_1,out_hr_2),dim=1))
        out_hr_3 = out_hr_3.view(octree_node_this.shape[0], -1)

        """Fuse features"""
        fuse_feats = torch.cat((feats_prev, feats_this,out_hr_3, surface_feats, octree_node_this), dim=1)
        out = self.linear_block_fuse(fuse_feats)
        if self.training:
            return out, par_feats.detach(), out_plane_params.view(octree_node_this.shape[0],self.surface_num,self.surface_params_num)
        else:
            return out, par_feats.detach(),None

class Quadratic_Surface_Loss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, block, plane_param):
        '''
            plane_param: B x N x 10 (ax + by + cz + d)
            point to plane distance: |ax + by + cz + d| / (a^2 + b^2 + c^2)^0.5
            use square of the distance as the loss

            https://stackoverflow.com/questions/49054841/fitting-quadratic-surface-to-data-points-in-3d

             z = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        '''
        block = block.view(plane_param.shape[0],-1,1)

        mesh = torch.from_numpy(np.mgrid[-4:5:1, -4:5:1, -4:5:1].reshape(3, -1).T.astype(np.float32)).cuda().unsqueeze(0).repeat(plane_param.shape[0],1,1)

        # a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        quadratic = mesh[:, :, 0:1]**2 * plane_param[:,:, 0].unsqueeze(1)*block[:,:,0].unsqueeze(2) + \
                    mesh[:, :, 1:2]**2 * plane_param[:, :, 1].unsqueeze(1) * block[:, :, 0].unsqueeze(2) + \
                    mesh[:, :, 0:1] * mesh[:, :, 1:2] * plane_param[:, :, 2].unsqueeze(1) * block[:, :, 0].unsqueeze(2) + \
                    mesh[:, :, 0:1] * plane_param[:, :, 3].unsqueeze(1) * block[:, :, 0].unsqueeze(2) + \
                    mesh[:, :, 1:2] * plane_param[:, :, 4].unsqueeze(1) * block[:, :, 0].unsqueeze(2) + \
                    plane_param[:, :, 5].unsqueeze(2) * block[:, :, 0].unsqueeze(2)

        # vertical distance
        residuals = mesh[:, :, 2:3] - quadratic
        loss = torch.mean(torch.square(residuals))
        return loss