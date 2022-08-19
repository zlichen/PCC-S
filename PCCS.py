import os

import IPython
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataloader import KITTIOdometry, collate_fn
from torch.utils.data import DataLoader
from model import EntropyModel, Quadratic_Surface_Loss
import yaml
from torch.utils.tensorboard import SummaryWriter
import time
from util.utils import accuracy,AverageMeter, occupancy_utils
import torchac_utils
import torchac
import bz2
import argparse

def train(start_epoch, epoch, model_dict, opt_dict, occu_loss_fn_dict, surface_loss_fn_dict, train_loader,device, tb_summary_dict, ckpt_dir_dict, min_tree_level, max_tree_level):
    occ_util_class = occupancy_utils()
    for ep in tqdm(range(start_epoch, epoch), desc='epoch'):
        tic = time.time()
        losses = {}
        top1 = {}
        top5 = {}
        for level in range(min_tree_level, max_tree_level):
            losses.update({level: AverageMeter('Loss', ':.4e')})
            top1.update({level: AverageMeter('Acc@1', ':6.2f')})
            top5.update({level: AverageMeter('Acc@5', ':6.2f')})
        for i, input_dict in enumerate(tqdm(train_loader)):
            """Init par features for the root node"""
            par_feats = torch.zeros((len(input_dict['len_batch']), 32)).float().to(device)

            for level in range(min_tree_level, max_tree_level):
                model_dict[level].train()
                opt_dict[level].zero_grad()
                mask = input_dict['octree_nodes'][:,3]==level
                octree_nodes = input_dict['octree_nodes'][mask]
                blocks = input_dict['blocks'][mask]
                labels = input_dict['labels'][mask]
                hr_blocks = input_dict['mask_hr_blocks'][mask]
                cumu_par_feats_ls = []

                for left in range(0,octree_nodes.shape[0],step):
                    if left + step >= octree_nodes.shape[0]:
                        right = octree_nodes.shape[0]
                    else:
                        right = left + step
                    cumu_octree_node = octree_nodes[left:right].to(device)
                    cumu_block = blocks[left:right].to(device)
                    cumu_label = labels[left:right].to(device)
                    cumu_par_feats = par_feats[left:right].to(device)
                    cumu_hr_block = hr_blocks[left:right].to(device)

                    model_output, cumu_par_feats, out_plane_params = model_dict[level](cumu_par_feats, cumu_octree_node, cumu_block, cumu_hr_block)
                    cumu_par_feats_ls.append(cumu_par_feats)
                    surface_loss = surface_loss_fn_dict[level](cumu_block, out_plane_params)

                    acc1, acc5 = accuracy(model_output, cumu_label, topk=(1, 5))
                    ent_loss = occu_loss_fn_dict[level](model_output, cumu_label)
                    loss = ent_loss + surface_loss * 0.2

                    losses[level].update(loss.item(), cumu_octree_node.size(0))
                    top1[level].update(acc1[0], cumu_octree_node.size(0))
                    top5[level].update(acc5[0], cumu_octree_node.size(0))
                    loss.backward()
                opt_dict[level].step()
                par_feats = torch.cat(cumu_par_feats_ls)
                par_repeads_idx = torch.tensor([occ_util_class.dec_to_bin_idx_dict[label.item()]['num_pos'] for label in labels]).long().to(device)
                """repeat the number of children nodes for each parent."""
                par_feats = torch.repeat_interleave(par_feats, par_repeads_idx, dim=0)

                tb_summary_dict[level].add_scalar('train/loss', losses[level].avg, ep * len(train_loader) + i + 1)
                tb_summary_dict[level].add_scalar('train/acc1', top1[level].avg, ep * len(train_loader) + i + 1)
                tb_summary_dict[level].add_scalar('train/acc5', top5[level].avg, ep * len(train_loader) + i + 1)

                if (i + 1) % 500 == 0:
                    print('[INFO] Saving')
                    torch.save({'state_dict': model_dict[level].state_dict(),
                                'opt': opt_dict[level].state_dict(),
                                'epoch': ep + 1,
                                'iter': i,
                                },
                               '%s/latest_iter.ckpt' %
                               (ckpt_dir_dict[level]))

        toc = time.time()

        for level in range(min_tree_level,max_tree_level):
            print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e, acc1: %.4e, acc5: %.4e' % (
                ep,
                toc - tic,
                losses[level].avg,
                top1[level].avg,
                top5[level].avg
            )
                  )

            if (ep+1) % 1 == 0 and (ep+1) <= 50:
                print('[INFO] Saving')
                torch.save({'state_dict': model_dict[level].state_dict(),
                            'opt': opt_dict[level].state_dict(),
                            'epoch' : ep+1
                            },
                           '%s/%04d.ckpt' %
                           (ckpt_dir_dict[level], ep+1))

            """Save the latest ckpt"""
            torch.save({'state_dict': model_dict[level].state_dict(),
                        'opt': opt_dict[level].state_dict(),
                        'epoch': ep + 1
                        },
                       '%s/latest.ckpt' %
                        (ckpt_dir_dict[level]))




def save_byte_stream(prob, sym):
    # torch.Size([178724, 256]) torch.Size([178724])
    bt, Q_len =prob.shape
    prob = prob.view(bt,Q_len)
    sym = sym.view(sym.shape[0])
    # Convert to a torchac-compatible CDF.

    output_cdf = torchac_utils.pmf_to_cdf(prob)
    # torchac expects sym as int16, see README for details.
    sym = sym.to(torch.int16)
    # torchac expects CDF and sym on CPU.
    output_cdf = output_cdf.detach().cpu()
    sym = sym.detach().cpu()
    # Get real bitrate from the byte_stream.
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    real_bits = len(byte_stream) * 8
    return real_bits

@torch.no_grad()
def val(start_epoch, model_dict, val_loader, device,min_tree_level, max_tree_level):
    softmax = nn.Softmax(dim=1)
    bpp_dict = dict()
    time_dict = dict()
    compression_rate_ls = []
    for level in range(min_tree_level, max_tree_level):
        bpp_dict.update({level: []})
        time_dict.update({level: []})
    acc1_ls = []
    acc5_ls = []
    val_num = min(max_val_num,len(val_loader))
    data_loader_iter = val_loader.__iter__()
    occ_util_class = occupancy_utils()
    for i in tqdm(range(0,val_num)):
        input_dict = data_loader_iter.next()
        labels_of_tree = []
        probs_of_tree = []
        original_points = input_dict['kitti_pts'][0].shape[0]
        min_bound, max_bound = input_dict['min_bound'][0].numpy(), input_dict['max_bound'][0].numpy()
        meta_data = np.stack([min_bound,max_bound])
        meta_data_bits = len(bz2.compress(meta_data))
        """Init par features for the root node"""
        par_feats = torch.zeros((len(input_dict['len_batch']), 32)).float().to(device)
        frame_total_time = 0.0
        for level in range(max_tree_level):
            model_dict[level].eval()
            mask = input_dict['octree_nodes'][:, 3] == level
            octree_nodes = input_dict['octree_nodes'][mask]
            blocks = input_dict['blocks'][mask]
            labels = input_dict['labels'][mask]
            hr_blocks = input_dict['mask_hr_blocks'][mask]
            cumu_par_feats_ls = []

            for left in range(0,octree_nodes.shape[0],step):
                if left + step >= octree_nodes.shape[0]:
                    right = octree_nodes.shape[0]
                else:
                    right = left + step
                cumu_octree_node = octree_nodes[left:right].to(device)
                cumu_block = blocks[left:right].to(device)
                cumu_label = labels[left:right].to(device)
                cumu_par_feats = par_feats[left:right].to(device)
                cumu_hr_block = hr_blocks[left:right].to(device)

                # Distribution: [num_nodes, 256]
                # Gt symbol: \in [0-255]: 0,1,...255
                tic = time.time()
                model_output, cumu_par_feats, _ = model_dict[level](cumu_par_feats, cumu_octree_node,
                                                                                   cumu_block, cumu_hr_block)
                model_output = softmax(model_output)
                toc = time.time()
                # if level>0:
                frame_total_time= frame_total_time+ (toc-tic)

                cumu_par_feats_ls.append(cumu_par_feats)

                labels_of_tree.append(cumu_label)
                probs_of_tree.append(model_output)

                acc1, acc5 = accuracy(model_output, cumu_label, topk=(1, 5))
                acc1_ls.append(acc1[0])
                acc5_ls.append(acc5[0])

            par_feats = torch.cat(cumu_par_feats_ls)
            par_repeads_idx = torch.tensor(
                [occ_util_class.dec_to_bin_idx_dict[label.item()]['num_pos'] for label in labels]).long().to(device)
            """repeat the number of children nodes for each parent."""
            par_feats = torch.repeat_interleave(par_feats, par_repeads_idx, dim=0)

            real_bits = save_byte_stream(torch.cat(probs_of_tree),torch.cat(labels_of_tree))
            level_bpp_ls = bpp_dict[level]
            level_bpp_ls.append((real_bits+meta_data_bits)*1.0/original_points)
            bpp_dict.update({level:level_bpp_ls})
            """Time estimation"""
            level_time_ls = time_dict[level]
            level_time_ls.append(frame_total_time)
            time_dict.update({level:level_time_ls})

        labels_of_tree = torch.cat(labels_of_tree)
        probs_of_tree = torch.cat(probs_of_tree)

        real_bits = save_byte_stream(probs_of_tree,labels_of_tree)
        compression_rate_ls.append(original_points*96.0/(real_bits+meta_data_bits))

    print('-'*15)
    print(f'\n[Final Result]\ncompression_rate:\t{sum(compression_rate_ls)/len(compression_rate_ls)}')

    for level in range(min_tree_level, max_tree_level):
        level_bpp_ls = bpp_dict[level]
        bpp_avg = sum(level_bpp_ls)/len(level_bpp_ls)
        CR_ls = [96.0 / bpp for bpp in level_bpp_ls]
        CR = sum(CR_ls) / len(CR_ls)
        print(f'Level:\t{level}\nbpp:\t{bpp_avg}\nCR:\t{CR}')

    for level in range(min_tree_level, max_tree_level):
        level_time_ls = time_dict[level]
        time_avg = sum(level_time_ls)/len(level_time_ls)
        print(f'Level\t{level}\ntime:\t{time_avg}')

@torch.no_grad()
def cache_pred(pre_cache_feats_dir, model_dict, cache_loader, device, max_tree_level):
    """
    Cache occupancy prediction for training/evaluating refinement module
    Args:
        pre_cache_feats_dir:
        model_dict:
        cache_loader:
        device:
        max_tree_level:

    Returns:

    """
    softmax = nn.Softmax(dim=1)
    occ_util_class = occupancy_utils()
    for i, input_dict in enumerate(tqdm(cache_loader)):
        """Init par features for the root node"""
        data_paths = input_dict['data_path']
        seq = data_paths[0].split('/')[-2]
        frame = data_paths[0].split('/')[-1]
        if not os.path.exists(os.path.join(pre_cache_feats_dir,seq)):
            os.makedirs(os.path.join(pre_cache_feats_dir,seq))
        save_frame_path = os.path.join(pre_cache_feats_dir,seq,frame)
        all_par_feats = []
        all_occu_labels = []
        par_feats = torch.zeros((len(input_dict['len_batch']), 32)).float()
        level_tensor = torch.ones((par_feats.shape[0], 1)) * 0
        tmp_par_feats = torch.cat((par_feats.cpu(), level_tensor), dim=1)
        all_par_feats.append(tmp_par_feats.cpu())

        for level in range(max_tree_level):
            model_dict[level].eval()
            mask = input_dict['octree_nodes'][:, 3] == level
            octree_nodes = input_dict['octree_nodes'][mask]
            blocks = input_dict['blocks'][mask]
            labels = input_dict['labels'][mask]
            hr_blocks = input_dict['mask_hr_blocks'][mask]
            cumu_par_feats_ls = []
            for left in range(0,octree_nodes.shape[0],step):
                if left + step >= octree_nodes.shape[0]:
                    right = octree_nodes.shape[0]
                else:
                    right = left + step
                cumu_octree_node = octree_nodes[left:right].to(device)
                cumu_block = blocks[left:right].to(device)
                cumu_par_feats = par_feats[left:right].to(device)
                cumu_hr_block = hr_blocks[left:right].to(device)
                cumu_hr_block = torch.zeros_like(cumu_hr_block).to(device) # Sibling context are unknown for the children octants of the last level; Set to zero

                # Distribution: [num_nodes, 256]
                # Gt symbol: \in [0-255]: 0,1,...255
                model_output, cumu_par_feats, _ = model_dict[level](cumu_par_feats, cumu_octree_node,
                                                                                   cumu_block, cumu_hr_block)
                # model_output = softmax(model_output)
                cumu_par_feats_ls.append(cumu_par_feats)

                """Explain occupancy code"""
                pred_occu = softmax(model_output)
                _, index = torch.max(pred_occu, dim=1)
                all_occu_labels.append(index.cpu())

            if (level+1)==max_tree_level:
                continue
            par_feats = torch.cat(cumu_par_feats_ls)
            par_repeads_idx = torch.tensor(
                [occ_util_class.dec_to_bin_idx_dict[label.item()]['num_pos'] for label in labels]).long().to(device)
            """repeat the number of children nodes for each parent."""
            par_feats = torch.repeat_interleave(par_feats, par_repeads_idx, dim=0)
            level_tensor = torch.ones((par_feats.shape[0],1))*(level+1)
            tmp_par_feats = torch.cat((par_feats.cpu(),level_tensor),dim=1)
            all_par_feats.append(tmp_par_feats.cpu())

        all_par_feats = torch.cat(all_par_feats)
        all_occu_labels = torch.cat(all_occu_labels)
        np.savez_compressed(save_frame_path, par_feats=all_par_feats.cpu().numpy(), pred_occu_labels = all_occu_labels)
        print(f'Cached {save_frame_path}!')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCC-S",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--is_validation", action="store_true",default=False, help="to eval model.")
    parser.add_argument("--to_cache_par_feats", action="store_true",default=False, help="to cache occupancy prediction for refinement.")

    args = parser.parse_args()
    config = vars(args)
    config_dir = 'config_ent.yml'
    cfg = yaml.safe_load(open(config_dir, 'r'))
    print(config)
    print(cfg)

    lr = cfg['lr']
    batch_size = cfg['batch_size']
    epoch = cfg['epoch']
    start_epoch = 0
    is_validation = config['is_validation']
    step = cfg['node_each_iter']
    max_val_num = cfg['max_val_num']
    to_cache_par_feats = config['to_cache_par_feats']
    max_tree_level = cfg['octree_height'] if (is_validation or to_cache_par_feats) else cfg['octree_height']+1 # train one more level for prediction while refinement
    min_tree_level = 0
    # cached_par_feats = cfg['cached_par_feats']
    pre_cache_feats_dir = os.path.join(cfg['ROOT_dir'], cfg['pre_cache_feats_dir'])

    kitti_bin_dir = cfg['KITTI_BIN_dir']

    train_split = cfg['TRAIN_SPLIT']
    test_split = cfg['TEST_SPLIT']

    root_dir = cfg['ROOT_dir']

    data_dir = root_dir


    tb_dir = os.path.join(root_dir, 'ent/logs', cfg['CKPT_DIR'])

    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    ckpt_dir = os.path.join(root_dir,'ent/ckpts/', cfg['CKPT_DIR'])


    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    ckpt_level_dir_dict = {}
    for level in range(min_tree_level,max_tree_level):
        if not os.path.exists(os.path.join(ckpt_dir,str(level))):
            os.makedirs(os.path.join(ckpt_dir,str(level)))
        ckpt_level_dir_dict.update({level:os.path.join(ckpt_dir,str(level))})
        print(f'Level:\t{level}\tSave dir:\t{os.path.join(ckpt_dir,str(level))}')
    time.sleep(2.0)
    tb_summary_dict = {}
    for level in range(min_tree_level, max_tree_level):
        if not os.path.exists(os.path.join(tb_dir,str(level))):
            os.makedirs(os.path.join(tb_dir,str(level)))
        tb_level_dir = os.path.join(tb_dir,str(level))
        if not is_validation or not to_cache_par_feats:
            tb_summary = SummaryWriter(log_dir=tb_level_dir)
            tb_summary_dict.update({level: tb_summary})
    # if not is_validation:
    #     assert max_tree_level <= 12


    """model"""
    model_level_dict = {}
    opt_level_dict = {}
    occu_loss_fn_level_dict = {}
    surface_loss_fn_level_dict = {}
    start_epoch = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for level in range(min_tree_level, max_tree_level):
        model = EntropyModel()
        model = nn.DataParallel(model)
        model = model.to(device)
        if is_validation or to_cache_par_feats:
            # if cached_par_feats:
            #     print("Check Config! Trying to freeze the parameters!")
            #     exit(0)
            for param in model.parameters():
                param.requires_grad = False
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        if os.path.isfile(os.path.join(ckpt_level_dir_dict[level],'latest.ckpt')):
            ckpt = torch.load(os.path.join(ckpt_level_dir_dict[level],'latest.ckpt'))
            model.load_state_dict(ckpt['state_dict'])
            opt.load_state_dict(ckpt['opt'])
            start_epoch = ckpt['epoch']
            print(f"Load level {level} from {os.path.join(ckpt_level_dir_dict[level],'latest.ckpt')}...")
        elif is_validation or to_cache_par_feats:
            print('Trying to eval or to_cache_par_feats but ckpt not found!')
            exit()
        else:
            print(f"Init model at level {level}...")

        model_level_dict.update({level: model})
        opt_level_dict.update({level: opt})
        occu_loss_fn_level_dict.update({level:nn.CrossEntropyLoss()})
        surface_loss_fn_level_dict.update({level: Quadratic_Surface_Loss()})



    print(f'Start from epoch:\t{start_epoch}')
    if to_cache_par_feats:
        if os.path.isfile(os.path.join(ckpt_dir,str(max_tree_level), 'latest.ckpt')):
            ckpt = torch.load(os.path.join(ckpt_dir,str(max_tree_level), 'latest.ckpt'))
            model = EntropyModel()
            model = nn.DataParallel(model)
            model = model.to(device)
            model.load_state_dict(ckpt['state_dict'])
            model_level_dict.update({max_tree_level: model})
            print(f"Load level {max_tree_level} from {os.path.join(ckpt_dir,str(max_tree_level), '0020.ckpt')}...")
        else:
            print('Trying to to_cache_par_feats but ckpt not found!')
            exit()

        print('Preparing Training split.')
        cache_dataset =  KITTIOdometry(cfg=cfg, split='train', min_tree_level=min_tree_level,
                                      max_tree_level=max_tree_level+1)
        cache_loader = DataLoader(cache_dataset, batch_size=1, shuffle=False, num_workers=8,
                                pin_memory=False, collate_fn=collate_fn)
        cache_pred(pre_cache_feats_dir, model_level_dict, cache_loader, device, max_tree_level = max_tree_level+1)

        print('Preparing validation split.')

        cache_dataset =  KITTIOdometry(cfg=cfg, split='val', min_tree_level=min_tree_level,
                                      max_tree_level=max_tree_level+1)
        cache_loader = DataLoader(cache_dataset, batch_size=1, shuffle=False, num_workers=8,
                                pin_memory=False, collate_fn=collate_fn)

        cache_pred(pre_cache_feats_dir, model_level_dict, cache_loader, device, max_tree_level = max_tree_level+1)

    elif is_validation:
        val_dataset = KITTIOdometry(cfg = cfg, split = 'val', max_tree_level=max_tree_level)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10,
                                pin_memory=False, collate_fn=collate_fn)
        val(start_epoch, model_level_dict, val_loader, device, min_tree_level = min_tree_level, max_tree_level = max_tree_level)
    else:
        train_dataset = KITTIOdometry(cfg=cfg, split='train', min_tree_level=min_tree_level,
                                      max_tree_level=max_tree_level)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=False, collate_fn=collate_fn)

        train(start_epoch=start_epoch, epoch = epoch, model_dict = model_level_dict, opt_dict = opt_level_dict, occu_loss_fn_dict = occu_loss_fn_level_dict,
              surface_loss_fn_dict = surface_loss_fn_level_dict,
              train_loader = train_loader,device = device,tb_summary_dict=tb_summary_dict,
              ckpt_dir_dict = ckpt_level_dir_dict,min_tree_level = min_tree_level, max_tree_level = max_tree_level)

    print("Finish")