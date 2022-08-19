import os

import torch.nn as nn
from tqdm import tqdm
from dataloader import KITTIOdometry, collate_fn
from torch.utils.data import DataLoader
from model import EntropyModel
import yaml
import time
from util.utils import *
import torchac_utils
import torchac
import gc
import open3d as o3d
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as ChamferDistance
from util.pc_metrics import compute_pc_metrics


def save_byte_stream(prob, sym, save_name):
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
    with open(save_name, 'wb') as fout:
        fout.write(byte_stream)
        # Read from a file.
    with open(save_name, 'rb') as fin:
        byte_stream = fin.read()
    assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
    # print(torchac.decode_float_cdf(output_cdf, byte_stream),sym)
    return real_bits

def get_symbol_from_byte_stream(byte_stream, prob):
    bt, Q_len = prob.shape
    prob = prob.view(bt, Q_len)

    output_cdf = torchac_utils.pmf_to_cdf(prob).detach().cpu()
    return torchac.decode_float_cdf(output_cdf, byte_stream)


@torch.no_grad()
def encode(input_dict, model_dict, device, max_tree_level, save_name, step):
    occ_util_class = occupancy_utils().dec_to_bin_idx_dict
    """Solved the precision error caused by the calculation"""
    all_bytes = 0.0
    real_bits = 0.0
    softmax = nn.Softmax(dim=1)

    min_bound,max_bound = input_dict['min_bound'][0].numpy(), input_dict['max_bound'][0].numpy()
    voxel_size_by_level = get_voxel_size_by_level_dict(max_bound, min_bound)
    range_bound = max_bound - min_bound
    parent_nodes = [TreeNode(min_bound, node_idx=0, curr_occu=None, level=0, voxel_size_by_level=voxel_size_by_level)]
    octree_nodes, blocks = coords_to_blocks(np.expand_dims(parent_nodes[0].coords, axis=0), level=0,
                                           range_bound=range_bound)

    # iter_cnt = 0
    """Data"""
    with open(os.path.join(save_name,'meta.b'), 'wb') as f:
        f.write(np.array((min_bound, max_bound), dtype=np.float32).tobytes())
    all_bytes+=os.path.getsize(os.path.join(save_name,'meta.b'))

    par_feats = torch.zeros((len(input_dict['len_batch']), 32)).float().to(device)
    par_repeads_idx = torch.ones((len(input_dict['len_batch']))).long()

    # HR information
    # bin_labels = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).bool()
    # hr_bin_occu_block = torch.zeros(1 * 8).int()
    idx_ls = [[0]] # record the occupied children siblings indexes; By 00110000->[2,3]

    hr_blocks = torch.zeros(len(idx_ls),1, 4, 4, 4)

    for level in range(0,max_tree_level):
        cumu_par_feats_ls = [None for i in range(par_feats.shape[0])]
        model_output_ls = []
        # Obtain the occupancy code of a specific level
        mask = input_dict['octree_nodes'][:, 3] == level
        labels = input_dict['labels'][mask]
        # print('level:\t',labels)

        for node_idx in range(0,8):
            sib_mask = [True if node.node_idx==node_idx else False for node in parent_nodes]
            sib_labels = labels[sib_mask].to(device)
            if len(sib_labels)==0:
                # print(f'Skip Level:\t{level}\tat sib idx:\t{node_idx}')
                continue
            # sib_labels_ls.append(sib_labels)
            sib_octree_nodes = octree_nodes[sib_mask].to(device)
            sib_blocks = blocks[sib_mask].to(device)
            sib_par_feats = par_feats[sib_mask].to(device)
            # Obtain the hr block which group by group of siblings
            hr_mask = [True if node_idx in sib_group else False for sib_group in idx_ls]

            sib_hr_blocks = hr_blocks[hr_mask]
            model_output, cumu_par_feats, _ = model_dict[level](sib_par_feats, sib_octree_nodes,
                                                                sib_blocks, sib_hr_blocks)
            model_output = softmax(model_output)
            # cumu_par_feats_ls.append(cumu_par_feats)

            # To ensure the order of the parent features is consistent to the next level
            sib_mask_to_idx = [i for i, x in enumerate(sib_mask) if x]
            cnt = 0
            for tmp_idx in sib_mask_to_idx:
                cumu_par_feats_ls[tmp_idx] = cumu_par_feats[cnt,:].reshape(-1,32)
                cnt+=1
            # Perform Compression
            tmp_save_name = os.path.join(save_name,str(level).zfill(2))
            if not os.path.exists(tmp_save_name):
                os.makedirs(tmp_save_name)
            tmp_save_name = os.path.join(tmp_save_name,str(node_idx) + '.b')
            real_bits += save_byte_stream(model_output, sib_labels, tmp_save_name)
            all_bytes += os.path.getsize(tmp_save_name)
            # print(f'Finished Level:\t{level}\tat sib idx:\t{node_idx}')

            # transfer the node_idx of siblings occu to bin and fill in the hr block
            sib_label_to_bin = [torch.tensor(occ_util_class[occu.item()]['bin_occus']) for occu in sib_labels]
            # Reshape to 2x2x2 small block
            sib_2x2x2_blocks = torch.cat(sib_label_to_bin).reshape(-1, 2, 2, 2).float().unsqueeze(1)
            # fill in the hr block (4x4x4);
            cnt = 0
            for row in range(0, 4, 2):
                for col in range(0, 4, 2):
                    for dep in range(0, 4, 2):
                        if cnt == node_idx:
                            sib_hr_blocks[: ,:, row:row + 2, col:col + 2, dep:dep + 2] = sib_2x2x2_blocks
                        cnt+=1
            hr_blocks[hr_mask] = sib_hr_blocks

        # get the idx_ls which group all sibs in a sublist
        idx_ls = [occ_util_class[label.item()]['idx_occus'] for label in labels]
        # hr block for next level
        hr_blocks = torch.zeros(len(idx_ls), 1, 4, 4, 4)

        # form the parent features for the next level
        par_repeads_idx = torch.tensor(
            [occ_util_class[label.item()]['num_pos'] for label in labels]).long().to(device)
        par_feats = torch.repeat_interleave(torch.cat(cumu_par_feats_ls), par_repeads_idx, dim=0)

        """get children for parents"""
        occupancy_ls = ['{0:08b}'.format(int(occu)) for occu in labels]
        child_nodes = []
        [child_nodes.extend(node.get_children_nodes(occu_symbols=occu_symbols)) for node, occu_symbols in
         zip(parent_nodes, occupancy_ls)]

        coords = [np.expand_dims(node.coords, axis=0) for node in child_nodes]
        coords = np.concatenate(coords, axis=0)
        if level + 1 < max_tree_level:
            octree_nodes, blocks = coords_to_blocks(coords, level=level + 1, range_bound=range_bound)
        parent_nodes = child_nodes

        print(f'Finished Level:\t{level}')
    print('-'*15)
    print('Finish Encoding')
    print('-'*15)
    # print("Compression Rate:\t",(input_dict['kitti_pts'][0].shape[0]*96.0)/(all_bytes*8.0))
    return all_bytes*8.0

@torch.no_grad()
def decode(model_dict, device, max_tree_level, save_name, step):
    occ_util_class = occupancy_utils().dec_to_bin_idx_dict
    softmax = nn.Softmax(dim=1).to(device)
    # with open(save_name+'meta.b', 'rb') as f:

    meta = np.fromfile(os.path.join(save_name,'meta.b'), dtype=np.float32)
    min_bound, max_bound = meta[:3],meta[3:]
    voxel_size_by_level = get_voxel_size_by_level_dict(max_bound, min_bound) # Dictionary to obtain voxel size by level
    range_bound = max_bound-min_bound

    """Initialize the root node"""
    parent_nodes = [TreeNode(min_bound, node_idx=0,curr_occu=None,level = 0,voxel_size_by_level=voxel_size_by_level)]
    octree_nodes, blocks = coords_to_blocks(np.expand_dims(parent_nodes[0].coords,axis=0), level=0, range_bound=range_bound)

    par_feats = torch.zeros((1, 32)).float().to(device)
    par_repeads_idx = torch.ones((1)).long()

    # HR information
    # bin_labels = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).bool()
    # hr_bin_occu_block = torch.zeros(1 * 8).int()
    idx_ls = [[0]] # record the occupied children siblings indexes; By 00110000->[2,3]
    hr_blocks = torch.zeros(len(idx_ls),1, 4, 4, 4)

    for level in range(0, max_tree_level):
        """Entropy model"""
        # cumu_par_feats_ls = []
        cumu_par_feats_ls = [None for i in range(par_feats.shape[0])]
        model_output_ls = []
        labels_ls = [[] for i in range(par_feats.shape[0])]
        for node_idx in range(0, 8):
            """Read from compress binary file"""
            tmp_save_name = os.path.join(save_name, str(level).zfill(2))
            tmp_save_name = os.path.join(tmp_save_name, str(node_idx) + '.b')
            if not os.path.isfile(tmp_save_name):
                # print(f'Skip Level:\t{level}\tat sib idx:\t{node_idx}')
                continue
            with open(tmp_save_name, 'rb') as fin:
                byte_stream = fin.read()

            sib_mask = [True if node.node_idx == node_idx else False for node in parent_nodes]
            sib_octree_nodes = octree_nodes[sib_mask].to(device)
            sib_blocks = blocks[sib_mask].to(device)
            sib_par_feats = par_feats[sib_mask].to(device)
            # Obtain the hr block which group by group of siblings
            hr_mask = [True if node_idx in sib_group else False for sib_group in idx_ls]
            sib_hr_blocks = hr_blocks[hr_mask]
            model_output, cumu_par_feats, _ = model_dict[level](sib_par_feats, sib_octree_nodes,
                                                                sib_blocks, sib_hr_blocks)
            model_output = softmax(model_output)
            # cumu_par_feats_ls.append(cumu_par_feats)


            """Decode"""
            sib_labels = get_symbol_from_byte_stream(byte_stream, model_output)
            # To ensure the order of the parent features is consistent to the next level
            sib_mask_to_idx = [i for i, x in enumerate(sib_mask) if x]
            cnt = 0
            for tmp_idx in sib_mask_to_idx:
                cumu_par_feats_ls[tmp_idx] = cumu_par_feats[cnt, :].reshape(-1, 32)
                labels_ls[tmp_idx].append(sib_labels[cnt])
                cnt += 1
            # labels_ls.append(sib_labels)
            # transfer the node_idx of siblings occu to bin and fill in the hr block
            sib_label_to_bin = [torch.tensor(occ_util_class[occu.item()]['bin_occus']) for occu in sib_labels]
            # Reshape to 2x2x2 small block
            sib_2x2x2_blocks = torch.cat(sib_label_to_bin).reshape(-1, 2, 2, 2).float().unsqueeze(1)
            # fill in the hr block (4x4x4);
            cnt = 0
            for row in range(0, 4, 2):
                for col in range(0, 4, 2):
                    for dep in range(0, 4, 2):
                        if cnt == node_idx:
                            sib_hr_blocks[: ,:, row:row + 2, col:col + 2, dep:dep + 2] = sib_2x2x2_blocks
                        cnt+=1
            hr_blocks[hr_mask] = sib_hr_blocks

        labels_ls = [item for sublist in labels_ls for item in sublist]
        labels_ls = torch.stack(labels_ls)
        # print('level:\t',labels_ls)
        # get the idx_ls which group all sibs in a sublist
        idx_ls = [occ_util_class[label.item()]['idx_occus'] for label in labels_ls]
        # hr block for next level
        hr_blocks = torch.zeros(len(idx_ls), 1, 4, 4, 4)
        # form the parent features for the next level
        par_repeads_idx = torch.tensor(
            [occ_util_class[label.item()]['num_pos'] for label in labels_ls]).long().to(device)
        par_feats = torch.repeat_interleave(torch.cat(cumu_par_feats_ls), par_repeads_idx, dim=0)
        """Transfer the decode int to binary"""
        occupancy_ls = ['{0:08b}'.format(int(occu)) for occu in labels_ls]

        """Init children nodes for each parent nodes with the corresponding occupancy symbols"""
        child_nodes = []
        [child_nodes.extend(node.get_children_nodes(occu_symbols=occu_symbols)) for node, occu_symbols in
         zip(parent_nodes, occupancy_ls)]

        coords = [np.expand_dims(node.coords, axis=0) for node in child_nodes]

        coords = np.concatenate(coords, axis=0)

        octree_nodes, blocks = coords_to_blocks(coords, level=level + 1, range_bound=range_bound)

        parent_nodes = child_nodes
        print(f'Finished Level:\t{level}')

    print('-'*15)
    print('Finish Decoding')
    print('-'*15)
    return coords, blocks



if __name__ == "__main__":
    config_dir = 'config_ent.yml'
    cfg = yaml.safe_load(open(config_dir, 'r'))
    root_dir = cfg['ROOT_dir']

    # epoch = cfg['epoch']
    start_epoch = 0
    step = cfg['node_each_iter']
    max_val_num = cfg['max_val_num']
    min_tree_level = 0
    max_tree_level = cfg['octree_height']
    compress_save_dir = cfg['compress_save_dir']

    if not os.path.exists(compress_save_dir):
        os.makedirs(compress_save_dir)

    kitti_bin_dir = cfg['KITTI_BIN_dir']
    chamfer_dist = ChamferDistance.chamfer_3DDist()

    """Ckpt dir"""
    ckpt_dir = os.path.join(root_dir,'ent/ckpts', cfg['CKPT_DIR'])
    ckpt_level_dir_dict = {}
    for level in range(min_tree_level,max_tree_level):
        ckpt_level_dir_dict.update({level:os.path.join(ckpt_dir,str(level))})
        print(f'Level:\t{level}\tSave dir:\t{os.path.join(ckpt_dir,str(level))}')
    time.sleep(2.0)

    """Ent Model"""
    model_level_dict = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for level in range(min_tree_level, max_tree_level):
        model = EntropyModel()
        model = nn.DataParallel(model)
        model = model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(device)
        model.eval()

        assert os.path.isfile(os.path.join(ckpt_level_dir_dict[level],'latest.ckpt'))
        ckpt = torch.load(os.path.join(ckpt_level_dir_dict[level], 'latest.ckpt'))
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']
        print(f"Load level {level} from {os.path.join(ckpt_level_dir_dict[level], 'latest.ckpt')}...")
        model_level_dict.update({level: model})




    val_dataset = KITTIOdometry(cfg=cfg, split='val', max_tree_level=max_tree_level)


    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,
                              pin_memory=False, collate_fn=collate_fn)

    print(f'Start from epoch:\t{start_epoch}')


    val_num = min(max_val_num,len(val_loader))
    data_loader_iter = val_loader.__iter__()
    compression_rate = []
    chamfer_dist_ls = []
    chamfer_ref_dist_ls = []
    encode_time = []
    decode_time = []
    gt_pts_num = []
    rec_pts_num = []
    d1_psnrs = []
    d2_psnrs = []
    d1_psnrs_ref = []
    d2_psnrs_ref = []

    progress_id = len([d for d in os.listdir(compress_save_dir) if f'tree_level_{max_tree_level}' in d]) + 1
    progress_dir_name = f'tree_level_{max_tree_level}'

    for i in tqdm(range(0,val_num)):
        input_dict = data_loader_iter.next()
        seq = input_dict['data_path'][0].split('/')[-2]
        frame = input_dict['data_path'][0].split('/')[-1].replace('.npz', '')
        if not os.path.exists(os.path.join(compress_save_dir,progress_dir_name, seq,'encode',frame)):
            os.makedirs(os.path.join(compress_save_dir,progress_dir_name, seq,'encode',frame))
        save_name = os.path.join(compress_save_dir,progress_dir_name, seq,'encode', frame)
        print(save_name)
        """Encoding"""
        tic = time.time()
        all_bits = encode(input_dict, model_level_dict, device,max_tree_level,save_name,step)
        encode_time.append(time.time()-tic)

        torch.cuda.synchronize()

        if not os.path.exists(os.path.join(compress_save_dir,progress_dir_name, seq,'decode')):
            os.makedirs(os.path.join(compress_save_dir,progress_dir_name, seq,'decode'))
        decode_file_name = os.path.join(compress_save_dir,progress_dir_name, seq,'decode',frame+'.ply')

        """Decoding"""
        tic = time.time()
        decode_coords, blocks = decode(model_level_dict, device, max_tree_level, save_name,step)
        rec_pts_num.append(decode_coords.shape[0])

        decode_time.append(time.time() - tic)

        """Write file"""
        out_coords = decode_coords

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(out_coords)
        o3d.io.write_point_cloud(decode_file_name, pcd, write_ascii=True)

        """Metrics"""
        kitti_frame_path = os.path.join(kitti_bin_dir,seq,'velodyne',frame+'.bin')
        original_points = np.fromfile(kitti_frame_path, dtype=np.float32).reshape(-1, 4)[:,:3]
        gt_pts_num.append(original_points.shape[0])

        """Compression rate"""
        kitti_bits = original_points.shape[0]*96.0
        compression_rate.append(kitti_bits / all_bits)
        """Distortion"""
        dist1, dist2, _, _ = chamfer_dist(torch.from_numpy(original_points).unsqueeze(0).cuda().float(),
                                          torch.from_numpy(decode_coords.astype(np.float32)).unsqueeze(0).cuda().float())
        cham_dist1 = torch.sqrt(dist1).mean().item()
        cham_dist2 = torch.sqrt(dist2).mean().item()
        chamfer_dist_ls.append(max(cham_dist1, cham_dist2))
        """PSNR"""
        ori_pcd = o3d.geometry.PointCloud()
        ori_pcd.points = o3d.utility.Vector3dVector(original_points)
        ori_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=12))

        pc_error_metrics = compute_pc_metrics(original_points, decode_coords.astype(np.float32),
                                              r=59.70, p1_n=np.asarray(ori_pcd.normals))
        d1_psnrs.append(pc_error_metrics["d1_psnr"])
        d2_psnrs.append(pc_error_metrics["d2_psnr"])


        gc.collect()
        torch.cuda.empty_cache()

    print(f'Compression Rate on average:\t{sum(compression_rate)/len(compression_rate)}')
    print(f'Chamfer on average:\t{sum(chamfer_dist_ls)/len(chamfer_dist_ls)}')
    print(f"d1_psnr:\t{sum(d1_psnrs)/len(d1_psnrs)}\nd2_psnr:\t{sum(d2_psnrs)/len(d2_psnrs)}\n")

    print()
    print(f'Encode time:\t{sum(encode_time)/len(encode_time)}')
    print(f'Decode time:\t{sum(decode_time) / len(decode_time)}')
    print()
    print(f'Original pts num:\t{sum(gt_pts_num)/len(gt_pts_num)}')
    print(f'Rec pts num:\t{sum(rec_pts_num)/len(rec_pts_num)}')

    print("Finish")