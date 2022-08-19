import os

import torch
import torch.utils.data
import numpy as np





def collate_fn(list_data):
    blocks, hr_blocks_with_mask, octree_nodes, label,max_bound, min_bound, data_path, kitti_pts,ent_feats, ent_pred_occu = list(
        zip(*list_data))
    blocks_batch = []
    hr_blocks_with_mask_batch = []
    octree_nodes_batch = []
    len_batch = []
    label_batch = []
    max_bound_batch = []
    min_bound_batch = []
    data_path_batch = []
    kitti_pts_batch = []
    ent_feats_batch = []
    ent_pred_occu_batch = []

    batch_id = 0
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(octree_nodes):
        N0 = octree_nodes[batch_id].shape[0]
        octree_nodes_batch.append(to_tensor(octree_nodes[batch_id].float()))
        blocks_batch.append(to_tensor(blocks[batch_id].float()))
        hr_blocks_with_mask_batch.append(to_tensor(hr_blocks_with_mask[batch_id].float()))
        if ent_feats[batch_id] is not None:
            ent_feats_batch.append(to_tensor(ent_feats[batch_id].float()))
            ent_pred_occu_batch.append(to_tensor(ent_pred_occu[batch_id].float()))
        len_batch.append(N0)
        label_batch.append(to_tensor(label[batch_id]))
        max_bound_batch.append(to_tensor(max_bound[batch_id]).float())
        min_bound_batch.append(to_tensor(min_bound[batch_id]).float())

        data_path_batch.append(data_path[batch_id])
        if kitti_pts[batch_id] is not None:
            kitti_pts_batch.append(to_tensor(kitti_pts[batch_id]).float())
    # Concatenate all lists
    octree_nodes_batch = torch.cat(octree_nodes_batch,0).float()
    blocks_batch = torch.cat(blocks_batch,0).float()
    hr_blocks_with_mask_batch = torch.cat(hr_blocks_with_mask_batch, 0).float()
    label_batch = torch.cat(label_batch,0).long()
    if ent_feats[batch_id] is not None:
        ent_feats_batch=torch.cat(ent_feats_batch, 0).float()
        ent_pred_occu_batch = torch.cat(ent_pred_occu_batch,0).float()
    return {
        'octree_nodes': octree_nodes_batch,
        'blocks': blocks_batch,
        'mask_hr_blocks': hr_blocks_with_mask_batch,
        'labels': label_batch,
        'len_batch': len_batch,
        'max_bound': max_bound_batch,
        'min_bound': min_bound_batch,
        'data_path': data_path_batch,
        'kitti_pts': kitti_pts_batch,
        'ent_feats_batch':ent_feats_batch,
        'ent_pred_occu_batch':ent_pred_occu_batch
    }



class KITTIOdometry(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg,
        split,
        min_tree_level = 0,
        max_tree_level = 12,
        cached_par_feats=False,
        pre_cache_feats_dir=''
    ):
        super().__init__()
        self.cfg = cfg
        self.processed_data = os.path.join(cfg['ROOT_dir'], cfg['PROCESSED_DATA_dir'])
        self.split = split
        self.data_file_ls = None
        self.min_tree_level = min_tree_level
        self.max_tree_level = max_tree_level
        assert self.min_tree_level < self.max_tree_level
        self.kitti_bin_dir = cfg['KITTI_BIN_dir']

        self.load_data_file_ls()
        self.cached_par_feats = cached_par_feats
        self.pre_cache_feats_dir = cfg['pre_cache_feats_dir']
        if self.cached_par_feats==True:
            assert len(self.pre_cache_feats_dir)>0




    def load_data_file_ls(self):
        data_file_ls = []
        if self.split == 'train':
            data_split = self.cfg['TRAIN_SPLIT']
        else:
            data_split = self.cfg['TEST_SPLIT']
        with open(data_split,'r') as infile:
            lines = infile.readlines()
        for i, ele in enumerate(lines):
            lines[i] = ele.rstrip('\n')
            seq = lines[i].split('/')[-3]
            frame = lines[i].split('/')[-1]
            data_file_ls.append(
                os.path.join(self.processed_data, seq, frame.replace('bin', 'npz')))
            if not os.path.isfile(data_file_ls[i]):
                print('Preprocessed data not found!')
                assert os.path.isfile(data_file_ls[i])
        self.data_file_ls = data_file_ls



    def __getitem__(self, idx):

        frame_data = np.load(self.data_file_ls[idx])
        blocks = torch.from_numpy(frame_data['blocks'].astype(np.float32)).unsqueeze(1).float()
        hr_blocks_with_mask = torch.from_numpy(frame_data['hr_mask_block'].astype(np.float32)).unsqueeze(1).float()
        octree_nodes = frame_data['octree_nodes']

        """Consider part of the tree to achieve smaller bitrate"""

        mask = octree_nodes[:,3]<self.max_tree_level

        blocks = blocks[mask]
        octree_nodes = octree_nodes[mask]
        hr_blocks_with_mask = hr_blocks_with_mask[mask]

        max_bound = frame_data['max_bound']
        min_bound = frame_data['min_bound']
        octree_nodes_info = torch.from_numpy(octree_nodes[:,:4].astype(np.float32)).float()
        label = torch.from_numpy(octree_nodes[:,4].astype(np.float32)).long()
        """load kitti data"""
        kitti_pts = None
        if self.split=='val':
            seq = self.data_file_ls[idx].split('/')[-2]
            frame = self.data_file_ls[idx].split('/')[-1].replace('.npz', '')
            kitti_frame_path = os.path.join(self.kitti_bin_dir, seq, 'velodyne', frame + '.bin')
            kitti_pts = np.fromfile(kitti_frame_path, dtype=np.float32).reshape(-1, 4)[:,:3]
        ent_feats = None
        ent_pred_occu = None
        if self.cached_par_feats:
            seq = self.data_file_ls[idx].split('/')[-2]
            frame = self.data_file_ls[idx].split('/')[-1]
            ent_feats_path = os.path.join(self.pre_cache_feats_dir,seq,frame)
            ent_feats = torch.from_numpy(np.load(ent_feats_path)['par_feats'])
            ent_pred_occu = torch.from_numpy(np.load(ent_feats_path)['pred_occu_labels'])



        return (blocks, hr_blocks_with_mask,octree_nodes_info, label, max_bound,min_bound, self.data_file_ls[idx], kitti_pts, ent_feats, ent_pred_occu)

    def __len__(self):
        return len(self.data_file_ls)