import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import KITTIOdometry, collate_fn
from torch.utils.data import DataLoader
from model import Refine_Model
import yaml
from torch.utils.tensorboard import SummaryWriter
import time
from util.utils import AverageMeter, occupancy_utils
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as ChamferDistance
import open3d as o3d
import argparse

def train(start_epoch, epoch, refine_model_dict,ent_model_dict, opt_dict,cham_loss_fn_dict, train_loader,device, tb_summary_dict, ckpt_dir_dict, min_tree_level, max_tree_level):
    # model.train()
    occ_util_class = occupancy_utils()
    softmax_dict = {}
    for level in range(min_tree_level, max_tree_level):
        softmax_dict.update({level:nn.Softmax(dim=1)})
    # softmax = nn.Softmax(dim=1)
    for ep in tqdm(range(start_epoch, epoch), desc='epoch'):
        tic = time.time()
        losses = {}
        for level in range(min_tree_level, max_tree_level):
            losses.update({level: AverageMeter('Loss', ':.4e')})

        for i, input_dict in enumerate(tqdm(train_loader)):
            for level in range(min_tree_level, max_tree_level):
                refine_model_dict[level].train()
                opt_dict[level].zero_grad()
                mask = (input_dict['octree_nodes'][:,3]==level)
                # par_feats = input_dict['ent_feats_batch'][mask][:,:32]
                octree_nodes = input_dict['octree_nodes'][mask]
                par_coords = input_dict['octree_nodes'][mask][:, :3]
                blocks = input_dict['blocks'][mask]
                labels = input_dict['labels'][mask]
                mask_1 = (input_dict['ent_feats_batch'][:,-1]==level)
                pre_occu_labels = input_dict['ent_pred_occu_batch'][mask_1]

                max_bound = torch.cat(input_dict['max_bound']).reshape(-1, 3)
                min_bound = torch.cat(input_dict['min_bound']).reshape(-1, 3)
                par_vox_size = ((max_bound - min_bound) / (2 ** level))

                gt_coords, cnt_info, all_coords = occ_util_class.occu_to_coords(par_vox_size, par_coords, labels)

                for left in range(0,octree_nodes.shape[0],step):
                    if left + step >= octree_nodes.shape[0]:
                        right = octree_nodes.shape[0]
                    else:
                        right = left + step
                    """max_tree_level's coords"""
                    cumu_octree_node = octree_nodes[left:right].to(device)
                    cumu_block = blocks[left:right].to(device)
                    cumu_pre_occu = pre_occu_labels[left:right].to(device)

                    """Get gt coords of hr nodes"""
                    left_gt_coords = cnt_info[:left].sum()
                    right_gt_coords = cnt_info[:right].sum()
                    gt_coord = gt_coords[left_gt_coords:right_gt_coords].to(device)
                    """Get all coords"""
                    all_coord = all_coords[left:right].to(device)

                    """Get offset"""
                    offset = refine_model_dict[level](cumu_octree_node,cumu_block).reshape(-1, 8 , 3)
                    bin_mask = [torch.tensor(occ_util_class.dec_to_bin_idx_dict[occu.item()]['bin_occus']) for occu in cumu_pre_occu]
                    bin_labels = torch.cat(bin_mask).bool().to(device)
                    pred_coords = (offset + all_coord).reshape(-1,3)[bin_labels]

                    dist1, dist2, _, _ = cham_loss_fn_dict[level](gt_coord.unsqueeze(0),pred_coords.unsqueeze(0))
                    loss = torch.max(torch.mean(dist1), torch.mean(dist2))
                    losses[level].update(loss.item())
                    loss.backward()
                opt_dict[level].step()
                tb_summary_dict[level].add_scalar('train/loss', np.sqrt(losses[level].avg), (ep) * len(train_loader)+i+1)

        toc = time.time()
        for level in range(min_tree_level, max_tree_level):
            print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e' % (
                ep,
                toc - tic,
                np.sqrt(losses[level].avg)
            )
                  )

            if (ep + 1) % 1== 0:
                print('[INFO] Saving')
                torch.save({'state_dict': refine_model_dict[level].state_dict(),
                            'opt': opt_dict[level].state_dict(),
                            'epoch': ep + 1
                            },
                           '%s/%04d.ckpt' %
                           (ckpt_dir_dict[level], ep+1))

            """Save the latest ckpt"""
            torch.save({'state_dict': refine_model_dict[level].state_dict(),
                        'opt': opt_dict[level].state_dict(),
                        'epoch': ep + 1
                        },
                       '%s/latest.ckpt' %
                       (ckpt_dir_dict[level]))

@torch.no_grad()
def val(refine_model, val_loader, device, min_tree_level,max_val_num, save_ply_dir):
    cham_loss = ChamferDistance.chamfer_3DDist()
    occ_util_class = occupancy_utils()

    # softmax = nn.Softmax(dim=1)
    val_num = min(max_val_num,len(val_loader))
    data_loader_iter = val_loader.__iter__()
    refined_cham_max_ls = []
    refined_cham_mean_ls = []

    quant_cham_max_ls = []
    quant_cham_mean_ls = []

    for i in tqdm(range(0,val_num)):
        input_dict = data_loader_iter.next()
        original_points = torch.stack(input_dict['kitti_pts'])

        """to Save ply name"""
        data_file = input_dict['data_path'][0]
        seq = data_file.split('/')[-2]
        frame = data_file.split('/')[-1].replace('.npz', '')
        ply_name = seq+'_'+frame+'.ply'

        mask = (input_dict['octree_nodes'][:, 3] == min_tree_level)
        octree_nodes = input_dict['octree_nodes'][mask]
        par_coords = input_dict['octree_nodes'][mask][:, :3]
        blocks = input_dict['blocks'][mask]

        mask = input_dict['ent_feats_batch'][:,-1]==min_tree_level
        pre_occu_labels = input_dict['ent_pred_occu_batch'][mask]

        max_bound = torch.cat(input_dict['max_bound']).reshape(-1, 3)
        min_bound = torch.cat(input_dict['min_bound']).reshape(-1, 3)
        par_vox_size = ((max_bound - min_bound) / (2 ** min_tree_level))
        """All coordinate of the expanding parant nodes"""
        all_coords = occ_util_class.gen_all_coords(par_vox_size,par_coords)

        pred_coords = []
        refine_model.eval()
        for left in range(0,octree_nodes.shape[0],step):
            if left + step >= octree_nodes.shape[0]:
                right = octree_nodes.shape[0]
            else:
                right = left + step
            cumu_octree_node = octree_nodes[left:right].to(device)
            cumu_block = blocks[left:right].to(device)
            cumu_pre_occu = pre_occu_labels[left:right].to(device)

            """Get offset"""
            offset = refine_model(cumu_octree_node, cumu_block).reshape(-1, 8, 3)

            all_coord = all_coords[left:right].to(device)

            """Explain cached predicted occupancy code"""
            bin_mask = [torch.tensor(occ_util_class.dec_to_bin_idx_dict[occu.item()]['bin_occus']) for occu in
                        cumu_pre_occu]
            bin_labels = torch.cat(bin_mask).bool().to(device)
            pred_coords.append((offset + all_coord).reshape(-1, 3)[bin_labels])
        refine_coords = torch.cat(pred_coords)
        """Write our refine point cloud"""
        refined_pcd = o3d.geometry.PointCloud()
        refined_pcd.points = o3d.utility.Vector3dVector(refine_coords.cpu().numpy())

        o3d.io.write_point_cloud(
            os.path.join(save_ply_dir, ply_name), refined_pcd, write_ascii=True)
        dist1, dist2, _, _ = cham_loss(original_points.to(device).float(),
                                       refine_coords.unsqueeze(
                                           0).to(device).float())

        cham_dist1 = torch.sqrt(dist1).mean().item()
        cham_dist2 = torch.sqrt(dist2).mean().item()

        refined_cham_max_ls.append(max(cham_dist1, cham_dist2))
        refined_cham_mean_ls.append((cham_dist1 + cham_dist2) / 2)


        dist1, dist2, _, _ = cham_loss(original_points.to(device).float(),
                                          par_coords.unsqueeze(
                                              0).to(device).float())
        cham_dist1 = torch.sqrt(dist1).mean().item()
        cham_dist2 = torch.sqrt(dist2).mean().item()
        quant_cham_max_ls.append(max(cham_dist1, cham_dist2))
        quant_cham_mean_ls.append((cham_dist1 + cham_dist2) / 2)



    print('-'*15)
    print(f'\n[Final Result]')
    print(f'After chamfer max:\t{sum(refined_cham_max_ls)/len(refined_cham_max_ls)}')
    print(f'After chamfer mean:\t{sum(refined_cham_mean_ls)/len(refined_cham_mean_ls)}')

    print(f'Before chamfer max:\t{sum(quant_cham_max_ls)/len(quant_cham_max_ls)}')
    print(f'Before chamfer mean:\t{sum(quant_cham_mean_ls)/len(quant_cham_mean_ls)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refinement",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--is_validation", action="store_true",default=False, help="to eval model.")

    args = parser.parse_args()
    config = vars(args)
    is_validation = config['is_validation']

    config_dir = 'config_refine.yml'
    cfg = yaml.safe_load(open(config_dir, 'r'))

    ent_config_dir = 'config_ent.yml'
    ent_cfg = yaml.safe_load(open(ent_config_dir, 'r'))


    root_dir = ent_cfg['ROOT_dir']
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    epoch = cfg['epoch']
    start_epoch = 0
    step = cfg['node_each_iter']
    max_val_num = cfg['max_val_num']
    min_tree_level = cfg['refine_tree_level']
    max_tree_level = min_tree_level+1
    pre_cache_feats_dir = os.path.join(ent_cfg['ROOT_dir'], ent_cfg['pre_cache_feats_dir'])
    save_ply_dir = os.path.join(ent_cfg['ROOT_dir'],cfg['SAVE_PLY_dir'],'level_' + str(min_tree_level).zfill(2))

    if not os.path.exists(save_ply_dir):
        os.makedirs(save_ply_dir)

    """I can infer (max_tree_level)'s coords with the blocks"""
    # print(f'\nRefine for the depth\t{training_level}\n')

    ckpt_dir = os.path.join(root_dir,'refine/ckpts', cfg['CKPT_DIR'])

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_level_dir_dict = {}
    for level in range(min_tree_level,max_tree_level):
        if not os.path.exists(os.path.join(ckpt_dir,str(level))):
            os.makedirs(os.path.join(ckpt_dir,str(level)))
        ckpt_level_dir_dict.update({level:os.path.join(ckpt_dir,str(level))})
    print(ckpt_level_dir_dict)
    time.sleep(2.0)

    tb_dir = os.path.join(root_dir, 'refine/logs', cfg['CKPT_DIR'])
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    tb_summary_dict = {}
    for level in range(min_tree_level, max_tree_level):
        if not os.path.exists(os.path.join(tb_dir,str(level))):
            os.makedirs(os.path.join(tb_dir,str(level)))
        tb_level_dir = os.path.join(tb_dir,str(level))
        if not is_validation:
            tb_summary = SummaryWriter(log_dir=tb_level_dir)
            tb_summary_dict.update({level: tb_summary})

    """model"""
    refine_model_level_dict = {}
    ent_model_level_dict = {}
    opt_level_dict = {}
    cham_loss_fn_level_dict = {}
    start_epoch = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for level in range(min_tree_level, max_tree_level):
        ref_model = Refine_Model(out_dim=8*3)
        ref_model = nn.DataParallel(ref_model)
        ref_model = ref_model.to(device)
        if is_validation:
            for param in ref_model.parameters():
                param.requires_grad = False
        opt = torch.optim.Adam(ref_model.parameters(), lr=lr)
        if os.path.isfile(os.path.join(ckpt_level_dir_dict[level],'latest.ckpt')):
            ckpt = torch.load(os.path.join(ckpt_level_dir_dict[level],'latest.ckpt'))
            ref_model.load_state_dict(ckpt['state_dict'])
            opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch']
            print(f"Load level {level} from {os.path.join(ckpt_level_dir_dict[level],'latest.ckpt')}...")
        elif is_validation:
            print('Trying to eval but ckpt not found!')
            exit()
        else:
            print(f"Init model at level {level}...")



        refine_model_level_dict.update({level:ref_model})
        if not is_validation:
            opt_level_dict.update({level: opt})
            cham_loss_fn_level_dict.update({level:ChamferDistance.chamfer_3DDist()})



    if is_validation:
        val_dataset = KITTIOdometry(cfg=ent_cfg, split='val',min_tree_level=min_tree_level, max_tree_level=max_tree_level,cached_par_feats=True, pre_cache_feats_dir=pre_cache_feats_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,
                              pin_memory=False, collate_fn=collate_fn)
    else:
        train_dataset = KITTIOdometry(cfg=ent_cfg, split='train', min_tree_level=min_tree_level,
                                      max_tree_level=max_tree_level,cached_par_feats=True, pre_cache_feats_dir=pre_cache_feats_dir)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=False, collate_fn=collate_fn)

    print(f'Start from epoch:\t{start_epoch}')
    if is_validation:
        assert len(refine_model_level_dict.keys())==1
        # assert len(ent_model_level_dict.keys())==1
        val(refine_model_level_dict[min_tree_level], val_loader, device, min_tree_level, max_val_num, save_ply_dir)

    else:
        train(start_epoch=start_epoch, epoch = epoch, refine_model_dict = refine_model_level_dict,ent_model_dict=ent_model_level_dict, opt_dict = opt_level_dict,
        cham_loss_fn_dict = cham_loss_fn_level_dict,
              train_loader = train_loader,device = device,tb_summary_dict=tb_summary_dict,
              ckpt_dir_dict = ckpt_level_dir_dict,min_tree_level = min_tree_level, max_tree_level = max_tree_level)

    print("Finish")