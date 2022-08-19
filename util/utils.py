import torch
import numpy as np
import MinkowskiEngine as ME

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def get_voxel_size_by_level_dict(max_bound, min_bound):
    voxel_size_by_level = dict()
    for i in range(15):
        voxel_size_by_level.update({i: (max_bound - min_bound) / (2 ** i)})
    return voxel_size_by_level


class TreeNode:
    my_shift = np.array(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)))
    def __init__(self,
                 min_bound,
                 node_idx,
                 curr_occu,
                 level,
                 voxel_size_by_level
                 ):
        self.min_bound = min_bound
        self.node_idx = node_idx
        self.curr_occu = curr_occu
        self.child_node_ls = []
        self.level = level
        self.voxel_size_by_level = voxel_size_by_level
        self.init_origin_coords()

    def init_origin_coords(self):
        voxel_size = self.voxel_size_by_level[self.level]
        self.origin = self.min_bound + self.my_shift[self.node_idx] * voxel_size
        self.coords = self.origin + voxel_size * 0.5

    def get_children_nodes(self,occu_symbols):
        assert self.curr_occu== None
        # occupancy_ls = ['{0:08b}'.format(int(occu)) for occu in occu_symbols]
        self.curr_occu = occu_symbols

        min_bound = self.origin
        level = self.level + 1

        idx_ls = [i for i, e in enumerate(occu_symbols) if e != '0']

        for i, node_idx in enumerate(idx_ls):
            curr_occu = None
            child_node = TreeNode(min_bound, node_idx, curr_occu, level, self.voxel_size_by_level)
            self.child_node_ls.append(child_node)

        return self.child_node_ls


    def get_children_nodes_save_leaf(self, idx_ls, child_occus, leaves_ls):
        min_bound = self.origin
        level = self.level + 1

        for i, node_idx in enumerate(idx_ls):
            curr_occu = None
            if child_occus is not None:
                curr_occu = child_occus[i]
            child_node = TreeNode(min_bound, node_idx, curr_occu, level,self.voxel_size_by_level)

            if child_occus is None:
                leaves_ls.append(child_node.coords)
            else:
                self.child_node_ls.append(child_node)

        return self.child_node_ls


class occupancy_manager:
    def __init__(self, occupancy_ls, my_queue, pointer=0):
        self.occupancy_ls = occupancy_ls
        self.pointer = pointer
        self.my_queue = my_queue
        self.leaves_ls = []

    def get_occus_by_num(self, num=1):
        if self.pointer + num < len(self.occupancy_ls):
            right = self.pointer + num
            occus = self.occupancy_ls[self.pointer:right]
        else:
            return None
        self.pointer += num
        return occus

    def get_occus_by_popping_queue(self):
        curr_node = self.my_queue.popleft()
        curr_occu = curr_node.curr_occu

        idx_ls = [i for i, e in enumerate(curr_occu) if e != '0']

        occus = self.get_occus_by_num(len(idx_ls))

        children_nodes = curr_node.get_children_nodes(idx_ls, occus,self.leaves_ls)
        self.my_queue.extend(children_nodes)




def generate_quant_space(voxel_coords):

    voxel_coords = ME.utils.batched_coordinates([voxel_coords])

    voxel_fests = torch.ones((len(voxel_coords), 1)).float()

    sparse_tensor = ME.SparseTensor(
        voxel_fests,
        coordinates=voxel_coords,
        device ='cpu'
    )
        # quant_space_by_level.update({level: sparse_tensor})
    return sparse_tensor


def coords_to_blocks(coords,level,range_bound):
    voxel_shift = torch.from_numpy(np.mgrid[-4:5:1, -4:5:1, -4:5:1].reshape(3, -1).T.astype(np.int32))
    # coords = pts_info[:, :3]
    # level = int(octree_node[3])
    # occupancy = pts_info[:,4]
    """Octree_node info to save"""
    level_tensor = torch.from_numpy(np.array([level], dtype=np.float32)).unsqueeze(0).float()
    coords_tensor = torch.from_numpy(coords).float()

    level_tensor = level_tensor.repeat(coords_tensor.shape[0],1)
    octree_node = torch.cat((coords_tensor,level_tensor), dim=1).float()

    """generate the block"""
    voxel_size = range_bound / (2 ** level)
    voxel_coord = torch.from_numpy(((coords) / (
        voxel_size)).astype(np.int32))
    sp_tensor = generate_quant_space(voxel_coord)
    # sp_tensor = quant_space_by_level[level]
    tmp_voxel_shift = voxel_shift.repeat(voxel_coord.shape[0], 1, 1)

    query_coords = voxel_coord.unsqueeze(1) + tmp_voxel_shift

    # To minkowski 4d tensor
    query_coords = ME.utils.batched_coordinates([query_coords.view(-1, 3)]).float()

    # Get the occupancy features in the quant space
    query_feats = sp_tensor.features_at_coordinates(query_coords).squeeze()
    query_feats = query_feats.view(voxel_coord.shape[0], 9, 9, 9)
    return octree_node.float(), query_feats.unsqueeze(1).float()

class occupancy_utils():
    def __init__(self):
        self.gen_int_occu_to_bin_idx_list()
    def gen_int_occu_to_bin_idx_list(self):
        dec_array = torch.arange(256).tolist()
        occ_array = [list('{0:08b}'.format(int(occu))) for occu in dec_array]
        import numpy as np
        occus_array = (torch.from_numpy(np.array([[int(x) for x in list] for list in occ_array]))).tolist()
        """00100010->[2,6]"""
        occ_to_idx_ls = []
        rev_dec_ls = []
        self.dec_to_bin_idx_dict = dict()
        for i,occu in enumerate(occus_array):
            idx_ls = [i for i, e in enumerate(occu) if e != 0]
            occ_to_idx_ls.append(idx_ls)
            rev_dec_ls.append(int(''.join(map(str, occu)),2))
            bits = '{0:08b}'.format(int(i))
            cnt = 0
            for bit in bits:
                if bit != '0':
                    cnt += 1

            self.dec_to_bin_idx_dict.update({dec_array[i]:{'bin_occus':occus_array[i],'idx_occus':occ_to_idx_ls[i],'num_pos': cnt,'rev_dec':rev_dec_ls[i]}})

    def occu_to_coords(self,par_voxel_size,par_coords,occu_code):
        my_shift = np.array(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
             (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)))
        """[Bt, 3]"""
        # child_voxel_size = par_voxel_size/2
        min_bound = par_coords-(par_voxel_size/2/2)
        bin_mask = [torch.tensor(self.dec_to_bin_idx_dict[occu.item()]['bin_occus']) for occu in occu_code]
        cnt_ls = [torch.tensor(self.dec_to_bin_idx_dict[occu.item()]['num_pos']) for occu in occu_code]
        """The number of occupied voxel (points number) in hr blocks of each lr block"""
        cnt_info = torch.stack(cnt_ls)
        bin_labels = torch.cat(bin_mask).bool()


        # child_coords_ls = []
        # for i, node_idx in enumerate(idx_ls):
        #     origin = min_bound + torch.from_numpy(my_shift[node_idx]) * child_voxel_size
        #     coords = origin + child_voxel_size * 0.5
        #     child_coords_ls.append(coords)
        """[Bt, 1, 3] + [Bt, 8, 3] -> [Bt, 8, 3]"""
        block = min_bound.unsqueeze(1) + torch.from_numpy(my_shift).unsqueeze(0).repeat(par_coords.shape[0],1,1) * (par_voxel_size.unsqueeze(1)/2)
        coords = block.clone().reshape(-1, 3)[bin_labels]

        return coords, cnt_info, block

    def gen_all_coords(self,par_voxel_size,par_coords):
        """Get all the children nodes coordinates of each parent voxel."""
        my_shift = np.array(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
             (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)))
        """[Bt, 3]"""
        # child_voxel_size = par_voxel_size/2
        min_bound = par_coords-(par_voxel_size/2/2)
        block = min_bound.unsqueeze(1) + torch.from_numpy(my_shift).unsqueeze(0).repeat(par_coords.shape[0],1,1) * (par_voxel_size.unsqueeze(1)/2)
        return block

def get_mask_from_idx_4x4x4():
    mask_ls = [torch.zeros((2, 2, 2)) for i in range(8)]
    cnt = 0
    my_shift = np.array(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)))
    for i in range(len(my_shift)):
        for j in range(i):
            mask_ls[cnt][int(my_shift[j][0]),int(my_shift[j][1]),int(my_shift[j][2])] = 1
        mask_ls[cnt] = mask_ls[cnt].permute(2,1,0)
        cnt += 1
    for i in range(len(mask_ls)):
        mask_ls[i] = mask_ls[i].reshape(2,1, 2,1,2,1).repeat(1,2,1,2,1,2).reshape(4,4,4).bool()

    return mask_ls
