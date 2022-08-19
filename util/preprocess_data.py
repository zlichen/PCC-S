import os

import joblib
import MinkowskiEngine as ME
import torch
import numpy as np
import open3d as o3d
from utils import get_voxel_size_by_level_dict, occupancy_utils, get_mask_from_idx_4x4x4
import yaml
import argparse

def generate_quant_space(pts_by_level, level, range_bound):
    voxel_size = range_bound / (2 ** level)
    lidar = pts_by_level[level]
    voxel_coords = torch.from_numpy(
        (
            np.floor(lidar[:, :3] / voxel_size)
        ).astype(np.int32)
    )

    voxel_coords = ME.utils.batched_coordinates([voxel_coords])

    voxel_fests = torch.ones((len(voxel_coords), 1)).float()

    sparse_tensor = ME.SparseTensor(
        voxel_fests,
        coordinates=voxel_coords,
        device='cpu'
    )
    return sparse_tensor


def enhance_with_hr_block(filename, frame_data):
    occ_class = occupancy_utils()
    occ_class_dict = occ_class.dec_to_bin_idx_dict
    mask_ls = get_mask_from_idx_4x4x4()

    blocks = torch.from_numpy(frame_data['blocks'].astype(np.float32)).float()
    octree_nodes = torch.from_numpy(frame_data['octree_nodes'])
    labels = octree_nodes[:, 4]
    max_bound = frame_data['max_bound'].reshape(1, 3)
    min_bound = frame_data['min_bound'].reshape(1, 3)

    par_repeads_idx = torch.ones((1)).long()  # record the number of occupied children
    """Assume the is at the bottom """
    bin_labels = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).bool()
    hr_bin_occu_block = torch.zeros(1 * 8).int()
    idx_ls = [[0]]  # record the occupied children siblings indexes; By 00110000->[2,3]
    bin_masks = [mask_ls[idx] for ls in idx_ls for idx in ls]  # based on i-1 labels;

    labels = labels.int()
    hr_bin_occu_block_ls = []
    for level in range(0, 13):
        """
        Consider level i-1, i, i+1;
        i-1 is resolution at 1x1x1;
        i is resolution at 2x2x2;
        i+1 is resolution at 4x4x4;

        Consider N nodes (N<=8) at level i belong to same parent at level i-1
        Create hr block with 4x4x4 represent the N nodes' children at level i+1
        repeat this hr block for N times and multiply with corresponding mask.
        """
        mask_this = (octree_nodes[:, 3] == level)
        labels_this = labels[mask_this]
        """set the occupied positions (bool) to the occupancy code (decimal)"""
        hr_bin_occu_block[bin_labels] = labels_this
        """Expand to 4x4x4 as i+1 level"""
        hr_bin_occu_block = [torch.tensor(occ_class_dict[occu.item()]['bin_occus']) for occu in hr_bin_occu_block]

        new_hr_bin_occu_block = torch.zeros(len(par_repeads_idx), 4, 4, 4)  # group by parents. generate 4x4x4 block
        hr_bin_occu_block = torch.cat(hr_bin_occu_block).reshape(-1, 2, 2, 2).float()
        cnt = 0
        """Ensure the decoded occupancy binary representation is aligned with the mask"""
        for par_idx in range(len(par_repeads_idx)):
            for row in range(0, 4, 2):
                for col in range(0, 4, 2):
                    for dep in range(0, 4, 2):
                        new_hr_bin_occu_block[par_idx][row:row + 2, col:col + 2, dep:dep + 2] = hr_bin_occu_block[cnt]
                        cnt = cnt + 1
        hr_bin_occu_block = new_hr_bin_occu_block
        hr_bin_occu_block = torch.repeat_interleave(hr_bin_occu_block, par_repeads_idx.cpu(),
                                                    dim=0)  # Let every siblings at level i has its hr block for training

        """Ensure each new generate occupancy code is being unmask"""
        bin_masks = torch.stack(bin_masks).float()
        hr_bin_occu_block = hr_bin_occu_block * bin_masks
        hr_bin_occu_block_ls.append(hr_bin_occu_block)
        """--------------------------"""
        par_repeads_idx = torch.tensor([occ_class_dict[label.item()]['num_pos'] for label in labels_this]).long()

        idx_ls = [occ_class_dict[label.item()]['idx_occus'] for label in labels_this]
        bin_masks = [mask_ls[idx] for ls in idx_ls for idx in
                     ls]  # obtain the binary mask 4x4x4 for the next level based on the labels;

        bin_labels = [torch.tensor(occ_class_dict[occu.item()]['bin_occus']) for occu in
                      labels_this]  # expand the label to binary for the next level to fill occupancy code
        bin_labels = torch.cat(bin_labels).bool()
        hr_bin_occu_block = torch.zeros(8 * len(par_repeads_idx)).int()

    hr_bin_occu_block_ls = torch.cat(hr_bin_occu_block_ls).numpy().astype(bool)

    blocks = blocks.numpy().astype(bool)
    octree_nodes = octree_nodes.numpy().astype(np.float32)
    print('\'' + filename + '\'\t' + 'Processed!')
    np.savez_compressed(filename, blocks=blocks, octree_nodes=octree_nodes, max_bound=max_bound.reshape(-1),
                        min_bound=min_bound.reshape(-1), hr_mask_block=hr_bin_occu_block_ls)


def process_point_clouds(point_path, to_save_path):
    def traverse_get_level_points(node, node_info):
        early_stop = False

        if isinstance(node, o3d.geometry.OctreeInternalNode):
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                child_cnt = 0
                occupancy = '0b'
                for x in node.children:
                    if x:
                        occupancy = occupancy + ''.join('1')
                        child_cnt += 1
                    else:
                        occupancy = occupancy + ''.join('0')
                occupancy = np.array([float(int(occupancy, 2))], dtype=np.float32)
                occu_for_children = np.repeat(occupancy, child_cnt)
                """Prepare the occupancy code for the next level"""
                if node_info.depth + 1 not in occ_of_par.keys():
                    occ_of_par.update({node_info.depth + 1: occu_for_children})
                else:
                    tmp_occu_for_children = occ_of_par[node_info.depth + 1]
                    tmp_occu_for_children = np.concatenate((tmp_occu_for_children, occu_for_children))
                    occ_of_par[node_info.depth + 1] = tmp_occu_for_children

                depth = np.array([node_info.depth], dtype=np.float32)
                coords = (node_info.origin.reshape(-1) + (node_info.size / 2.0)).astype(np.float32)
                child_idx = np.array([node_info.child_index], dtype=np.float32)
                tmp_vox_size = np.array([voxel_size_dic_by_level[node_info.depth][0]], dtype=np.float32)
                """
                0-2: coords
                3: depth
                4: occupancy
                5: child_idx
                6: tmp_vox_size
                """
                tmp_array = np.concatenate((coords, depth, occupancy, child_idx, tmp_vox_size))

                if node_info.depth not in pts_by_level.keys():
                    pts_by_level.update({node_info.depth: [tmp_array]})
                else:
                    ls = pts_by_level[node_info.depth]
                    ls.append(tmp_array)
                    pts_by_level[node_info.depth] = ls
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            pass
        else:
            raise NotImplementedError('type not recognized!')

        return early_stop

    """Load data and build the octree with depth 12"""
    point_cloud = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    octree = o3d.geometry.Octree(max_depth=13)
    octree.convert_from_point_cloud(pcd, size_expand=0.)

    range_bound = octree.get_max_bound() - octree.get_min_bound()
    voxel_size_dic_by_level = get_voxel_size_by_level_dict(octree.get_max_bound(), octree.get_min_bound())
    pts_by_level = {}
    occ_of_par = {}
    """Get the points at every level"""
    octree.traverse(traverse_get_level_points)
    pts_by_level = [np.array(pts_by_level[i]).astype(np.float32) for i in range(len(pts_by_level))]
    occ_of_par.update({0: np.array([1.0], dtype=np.float32)})
    blocks = []

    voxel_shift = torch.from_numpy(np.mgrid[-4:5:1, -4:5:1, -4:5:1].reshape(3, -1).T.astype(np.int32))

    for level in range(len(pts_by_level)):
        """
        0-2: coords
        3: depth
        4: occupancy
        5: child_idx
        6: tmp_vox_size
        7: Parents occupancy code 
        """
        pts_by_level[level] = np.concatenate((pts_by_level[level], np.expand_dims(occ_of_par[level], axis=1)), axis=1)
        pts_info = pts_by_level[level]
        coords = pts_info[:, :3]
        # level = int(octree_node[3])
        # occupancy = pts_info[:,4]
        """Octree_node info to save"""
        # octree_node = octree_node[:4].astype(np.float32)

        """generate the block"""
        sp_tensor = generate_quant_space(pts_by_level, level, range_bound)
        voxel_size = voxel_size_dic_by_level[level]
        voxel_coord = torch.from_numpy(
            (
                np.floor(
                    coords / voxel_size
                )
            ).astype(np.int32)
        )

        tmp_voxel_shift = voxel_shift.repeat(voxel_coord.shape[0], 1, 1)
        # print(voxel_coord.shape,tmp_voxel_shift.shape)
        query_coords = voxel_coord.unsqueeze(1) + tmp_voxel_shift

        # To minkowski 4d tensor
        query_coords = ME.utils.batched_coordinates([query_coords.view(-1, 3)]).float()

        # Get the occupancy features in the quant space
        query_feats = sp_tensor.features_at_coordinates(query_coords).squeeze()
        query_feats = query_feats.view(voxel_coord.shape[0], 9, 9, 9)
        blocks.append(query_feats)

    blocks = torch.cat(blocks, dim=0).numpy().astype(bool)
    octree_nodes = np.concatenate(pts_by_level, axis=0).astype(np.float32)

    seq = point_path.split('/')[-3]
    filename = os.path.join(to_save_path, str(seq).zfill(2), os.path.basename(point_path).replace('.bin', '.npz'))

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    frame_data = {
        'blocks': blocks,
        'octree_nodes': octree_nodes,
        'max_bound': octree.get_max_bound(),
        'min_bound': octree.get_min_bound()
    }
    enhance_with_hr_block(filename, frame_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n_workers",type=int,default=16, help="set number of workers to process data.")


    args = parser.parse_args()
    config = vars(args)


    config_dir = 'config_ent.yml'
    cfg = yaml.safe_load(open(config_dir, 'r'))

    data_root_dir = cfg['KITTI_BIN_dir']
    to_save_path = cfg['PROCESSED_DATA_dir']

    """
    training_6000frs_seq_00_to_10.txt
    testing_550frs_seq_11_to_21.txt
    """

    sampled_data_files = [cfg['TRAIN_SPLIT'],
                          cfg['TEST_SPLIT']]
    for sampled_data_file in sampled_data_files:
        point_paths = []

        with open(sampled_data_file, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                point_paths.append(os.path.join(data_root_dir, line.rstrip('\n')))
                assert os.path.isfile(point_paths[-1])

        joblib.Parallel(
            n_jobs=config['n_workers'], verbose=10, pre_dispatch="all"
        )(
            [
                joblib.delayed(process_point_clouds)(point_path, to_save_path)
                for point_path in point_paths
            ]
        )
