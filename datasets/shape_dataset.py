import os, re
import numpy as np
import scipy.io as sio
from itertools import product, combinations, permutations
from glob import glob

import torch
import scipy
from torch.utils.data import Dataset

from utils.shape_util import read_shape
from utils.geometry_util import get_operators, get_elas_operators
from utils.tensor_util import to_numpy
from utils.registry import DATASET_REGISTRY

from tqdm import tqdm
import copy


def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)

def treat_nan(tensor, verts, name):
    """
    Replaces NaN values in a tensor with the nearest non-NaN values from a given vertices tensor.

    Args:
        tensor (torch.Tensor): The input tensor containing potential NaN values.
        verts (torch.Tensor): The vertices tensor used for finding nearest non-NaN entries.
        name (str): The name of the tensor for informative output.

    Returns:
        torch.Tensor: The modified tensor with NaN values replaced.
    """
    # Detect rows with NaN values
    nan_rows = torch.isnan(tensor).any(dim=1)
    if nan_rows.any():
        num_nan_rows = nan_rows.sum().item()
        nan_row_indices = nan_rows.nonzero(as_tuple=True)[0]
        print(f'Warning: {num_nan_rows} rows have NaN values in {name}, replacing with nearest non-NaN neighbor')
        print(f'Row numbers: {nan_row_indices}')

    # Identify non-NaN rows
    non_nan_rows = ~nan_rows
    # Calculate distances to nearest non-NaN neighbors
    nan_verts = verts[nan_rows]
    non_nan_verts = verts[non_nan_rows]
    distances = torch.cdist(nan_verts, non_nan_verts)
    _, nearest_indices = torch.min(distances, dim=1)

    # Replace NaN rows with their nearest non-NaN neighbors
    tensor[nan_rows] = tensor[non_nan_rows][nearest_indices]

    return tensor

def get_spectral_ops(item, num_evecs, cache_dir=None, dirichlet=False):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    _, mass, L, evals, evecs, _, _ = get_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs,
                                                   cache_dir=cache_dir)
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    if dirichlet:
        item['L'] = L.to_dense() # this to_dense() operation is expensive, opt out if possible

    return item

def get_elas_spectral_ops(item, num_evecs, bending_weight=1e-2, cache_dir=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    mass, evals, evecs = get_elas_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs, bending_weight=bending_weight, 
                                                   cache_dir=cache_dir)
    evecs = treat_nan(evecs, item['verts'], item['name'])
    mass = mass + 1e-8 * mass.mean()
    sqrtmass = torch.sqrt(mass)
    
    def const_proj(evec, sqrtmass):
        # orthogonal projector for elastic basis
        sqrtM = torch.diag(sqrtmass)
        return torch.linalg.pinv(sqrtM @ evec) @ sqrtM
    evecs_trans = const_proj(evecs[:, :num_evecs], sqrtmass)
    Mk = evecs.T @ torch.diag(mass) @ evecs
    item['elas_evecs'] = evecs[:, :num_evecs]
    item['elas_evecs_trans'] = evecs_trans[:num_evecs]
    item['elas_evals'] = evals[:num_evecs]
    item['elas_mass'] = mass
    item['elas_Mk'] = Mk
    sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness 
    sqrtMk = torch.tensor(sqrtMk).float().to(Mk.device)
    invsqrtMk = torch.linalg.pinv(sqrtMk)
    item['elas_sqrtMk'] = sqrtMk
    item['elas_invsqrtMk'] = invsqrtMk
    

    return item


class SingleShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
            return_corr (bool, optional): Indicate whether return the correspondences to reference shape. Default True.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs
        self.return_elas_evecs = return_elas_evecs
        self.bending_weight = bending_weight

        self.off_files = []
        self.corr_files = [] if self.return_corr else None
        self.dist_files = [] if self.return_dist else None

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        if self.return_dist:
            assert self._size == len(self.dist_files)

        if self.return_corr:
            assert self._size == len(self.corr_files)
        
        # treat phase attribute
        if not hasattr(self, 'phase'):
            self.phase = 'no_phase'
            # warn message print
            print("WARNING: no phase specified for dataset, using no_phase")

        # check the cache
        self.cache = cache
        if cache:
            cache_path_name = os.path.join(self.data_root, 'cache', str(self.num_evecs)+'_'+self.phase+"_dataset.pt")
            if self.return_elas_evecs:
                cache_path_name = os.path.join(self.data_root, 'cache', str(self.bending_weight)+'_'+str(self.num_evecs)+'_'+self.phase+"_elas_dataset.pt")
            print("using dataset cache path: " + str(cache_path_name))
            if os.path.exists(cache_path_name):
                print("  --> loading dataset from cache")
                self.item_list = torch.load(cache_path_name)
                return
            print("  --> dataset not in cache, repopulating")
            
            # populate the cache
            self.item_list = []
            for index in tqdm(range(self._size)):
                item = dict()

                # get shape name
                off_file = self.off_files[index]
                basename = os.path.splitext(os.path.basename(off_file))[0]
                item['name'] = basename

                # get vertices and faces
                verts, faces = read_shape(off_file)
                item['verts'] = torch.from_numpy(verts).float()
                if True or self.return_faces:
                    item['faces'] = torch.from_numpy(faces).long()
                
                # get eigenfunctions/eigenvalues
                if self.return_evecs:
                    item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=os.path.join(self.data_root, 'diffusion'))
                
                # get elastic eigenfunctions/eigenvalues
                if self.return_elas_evecs:
                    item = get_elas_spectral_ops(item, num_evecs=self.num_evecs, bending_weight=self.bending_weight, cache_dir=os.path.join(self.data_root, 'elastic'))

                # get correspondences
                if self.return_corr:
                    corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
                    item['corr'] = torch.from_numpy(corr).long()
                
                # do not cache geodesic distance matrix because it is too large

                self.item_list.append(item)
            os.makedirs(os.path.join(self.data_root, 'cache'), exist_ok=True)
            torch.save(self.item_list, cache_path_name)


    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(self.data_root, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))

    def __getitem__(self, index):
        if self.cache:
            item = self.item_list[index]
            # get geodesic distance matrix
            if self.return_dist:
                mat = sio.loadmat(self.dist_files[index])
                item['dist'] = torch.from_numpy(mat['dist']).float()
            # special case for DT4D dataset
            if isinstance(self, SingleDT4DDataset):
                # we still need to load the corr on the fly because the Pair DT4D Dataset is modifying it on the fly
                if self.return_corr:
                    corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
                    item['corr'] = torch.from_numpy(corr).long()
            return item
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        verts, faces = read_shape(off_file)
        item['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            item['faces'] = torch.from_numpy(faces).long()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            item = get_spectral_ops(item, num_evecs=self.num_evecs, cache_dir=os.path.join(self.data_root, 'diffusion'))
        
        if self.return_elas_evecs:
            item = get_elas_spectral_ops(item, num_evecs=self.num_evecs, bending_weight=self.bending_weight, cache_dir=os.path.join(self.data_root, 'elastic'))

        # get geodesic distance matrix
        if self.return_dist:
            mat = sio.loadmat(self.dist_files[index])
            item['dist'] = torch.from_numpy(mat['dist']).float()

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['corr'] = torch.from_numpy(corr).long()

        return item

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class SingleFaustDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        self.phase = phase
        super(SingleFaustDataset, self).__init__(data_root, return_faces,
                                                 return_evecs, num_evecs,
                                                 return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'

    def _init_data(self):
         # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(self.data_root, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))
        
        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0 

        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if self.phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
            if self.dist_files:
                self.dist_files = self.dist_files[:80]
            self._size = 80
        elif self.phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
            if self.dist_files:
                self.dist_files = self.dist_files[80:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleScapeDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        self.phase = phase
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        super(SingleScapeDataset, self).__init__(data_root, return_faces,
                                                 return_evecs, num_evecs,
                                                 return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        
    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

        # check the data path contains .mat files
        if self.return_dist:
            dist_path = os.path.join(self.data_root, 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} not containing .mat files'
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))
        
        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        assert len(self) == 71, f'FAUST dataset should contain 71 human body shapes, but get {len(self)}.'
        if self.phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:51]
            if self.corr_files:
                self.corr_files = self.corr_files[:51]
            if self.dist_files:
                self.dist_files = self.dist_files[:51]
            self._size = 51
        elif self.phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[51:]
            if self.corr_files:
                self.corr_files = self.corr_files[51:]
            if self.dist_files:
                self.dist_files = self.dist_files[51:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleShrec19Dataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        super(SingleShrec19Dataset, self).__init__(data_root, return_faces, return_evecs, num_evecs, False, return_dist, return_elas_evecs, bending_weight, cache)


@DATASET_REGISTRY.register()
class SingleSmalDataset(SingleShapeDataset):
    def __init__(self, data_root, phase='train', category=True,
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        self.phase = phase
        self.category = category
        super(SingleSmalDataset, self).__init__(data_root, return_faces, return_evecs, num_evecs,
                                                return_corr, return_dist, return_elas_evecs, bending_weight, cache)

    def _init_data(self):
        if self.category:
            txt_file = os.path.join(self.data_root, f'{self.phase}_cat.txt')
        else:
            txt_file = os.path.join(self.data_root, f'{self.phase}.txt')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                if self.return_corr:
                    self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                if self.return_dist:
                    self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]


@DATASET_REGISTRY.register()
class SingleDT4DDataset(SingleShapeDataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        self.phase = phase
        self.ignored_categories = ['pumpkinhulk']
        super(SingleDT4DDataset, self).__init__(data_root, return_faces,
                                                return_evecs, num_evecs,
                                                return_corr, return_dist, return_elas_evecs, bending_weight, cache)

    def _init_data(self):
        with open(os.path.join(self.data_root, f'{self.phase}.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.split('/')[0] not in self.ignored_categories:
                    self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                    if self.return_corr:
                        self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                    if self.return_dist:
                        self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]


@DATASET_REGISTRY.register()
class SingleShrec20Dataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200):
        super(SingleShrec20Dataset, self).__init__(data_root, return_faces,
                                                   return_evecs, num_evecs, False, False)


@DATASET_REGISTRY.register()
class SingleTopKidsDataset(SingleShapeDataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        self.phase = phase
        super(SingleTopKidsDataset, self).__init__(data_root, return_faces,
                                                   return_evecs, num_evecs, False, return_dist, return_elas_evecs, bending_weight, cache)


class PairShapeDataset(Dataset):
    def __init__(self, dataset):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        # if dataset.include_same_pair exists, use it, otherwise, set it to True
        include_same_pair = getattr(dataset, 'include_same_pair', True)
        if include_same_pair:
            self.combinations = list(product(range(len(dataset)), repeat=2))
        else:
            self.combinations = list(permutations(range(len(dataset)), 2))


    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        return len(self.combinations)


@DATASET_REGISTRY.register()
class PairDataset(PairShapeDataset):
    def __init__(self, data_root, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        dataset = SingleShapeDataset(data_root, return_faces, return_evecs, num_evecs,
                                     return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        super(PairDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairFaustDataset(PairShapeDataset):
    def __init__(self, data_root,
                 phase, include_same_pair=True, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        dataset = SingleFaustDataset(data_root, phase, return_faces,
                                     return_evecs, num_evecs,
                                     return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        dataset.include_same_pair = include_same_pair
        super(PairFaustDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairScapeDataset(PairShapeDataset):
    def __init__(self, data_root,
                 phase, include_same_pair=True, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        dataset = SingleScapeDataset(data_root, phase, return_faces,
                                     return_evecs, num_evecs,
                                     return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        dataset.include_same_pair = include_same_pair
        super(PairScapeDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairShrec19Dataset(Dataset):
    def __init__(self, data_root, phase='test',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=False, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        assert phase in ['train', 'test'], f'Invalid phase: {phase}'
        self.dataset = SingleShrec19Dataset(data_root, return_faces, return_evecs, num_evecs, return_dist, return_elas_evecs, bending_weight, cache)
        self.phase = phase
        if phase == 'test':
            corr_path = os.path.join(data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            # ignore the shape 40, since it is a partial shape
            self.corr_files = list(filter(lambda x: '40' not in x, sort_list(glob(f'{corr_path}/*.map'))))
            self._size = len(self.corr_files)
        else:
            self.combinations = list(product(range(len(self.dataset)), repeat=2))
            self._size = len(self.combinations)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.phase == 'train':
            # get index
            first_index, second_index = self.combinations[index]
        else:
            # extract pair index
            basename = os.path.basename(self.corr_files[index])
            indices = os.path.splitext(basename)[0].split('_')
            first_index = int(indices[0]) - 1
            second_index = int(indices[1]) - 1

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        if self.phase == 'test':
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['first']['corr'] = torch.arange(0, len(corr)).long()
            item['second']['corr'] = torch.from_numpy(corr).long()
        return item


@DATASET_REGISTRY.register()
class PairSmalDataset(PairShapeDataset):
    def __init__(self, data_root, phase='train', include_same_pair=True,
                 category=True, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        dataset = SingleSmalDataset(data_root, phase, category, return_faces,
                                    return_evecs, num_evecs,
                                    return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        dataset.include_same_pair = include_same_pair
        super(PairSmalDataset, self).__init__(dataset=dataset)


@DATASET_REGISTRY.register()
class PairDT4DDataset(PairShapeDataset):
    def __init__(self, data_root, phase='train',  include_same_pair=True,
                 inter_class=False, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        dataset = SingleDT4DDataset(data_root, phase, return_faces,
                                    return_evecs, num_evecs,
                                    return_corr, return_dist, return_elas_evecs, bending_weight, cache)
        super(PairDT4DDataset, self).__init__(dataset=dataset)
        self.inter_class = inter_class
        self.combinations = []
        if self.inter_class:
            self.inter_cats = set()
            files = os.listdir(os.path.join(self.dataset.data_root, 'corres', 'cross_category_corres'))
            for file in files:
                cat1, cat2 = os.path.splitext(file)[0].split('_')
                self.inter_cats.add((cat1, cat2))
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                # same category
                cat1, cat2 = self.dataset.off_files[i].split('/')[-2], self.dataset.off_files[j].split('/')[-2]
                if cat1 == cat2:
                    if not self.inter_class:
                        # whether include same pair
                        if include_same_pair:
                            self.combinations.append((i, j))
                        else: # exclude same pair
                            if i != j:
                                self.combinations.append((i, j))
                else:
                    if self.inter_class and (cat1, cat2) in self.inter_cats:
                        self.combinations.append((i, j))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]
        if self.dataset.return_corr and self.inter_class:
            # read inter-class correspondence
            first_cat = self.dataset.off_files[first_index].split('/')[-2]
            second_cat = self.dataset.off_files[second_index].split('/')[-2]
            corr = np.loadtxt(os.path.join(self.dataset.data_root, 'corres', 'cross_category_corres',
                                           f'{first_cat}_{second_cat}.vts'), dtype=np.int32) - 1
            
            # fix the cache problem because the corr entry is modified on the fly


            item['second']['corr'] = item['second']['corr'][corr]

        return item


@DATASET_REGISTRY.register()
class PairShrec20Dataset(PairShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=120):
        dataset = SingleShrec20Dataset(data_root, return_faces, return_evecs, num_evecs)
        super(PairShrec20Dataset, self).__init__(dataset=dataset)


@DATASET_REGISTRY.register()
class PairShrec16Dataset(Dataset):
    """
    Pair SHREC16 Dataset
    """
    categories = [
        'cat', 'centaur', 'david', 'dog', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(self,
                 data_root,
                 categories=None,
                 cut_type='cuts', return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=False, return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        assert cut_type in ['cuts', 'holes'], f'Unrecognized cut type: {cut_type}'

        categories = self.categories if categories is None else categories
        # sanity check
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = sorted(categories)
        self.cut_type = cut_type

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_evecs = return_evecs
        self.return_corr = return_corr
        self.return_dist = return_dist
        self.num_evecs = num_evecs
        self.return_elas_evecs = return_elas_evecs
        self.bending_weight = bending_weight

        # full shape files
        self.full_off_files = dict()
        self.full_dist_files = dict()

        # partial shape files
        self.partial_off_files = dict()
        self.partial_corr_files = dict()

        # load full shape files
        off_path = os.path.join(data_root, 'null', 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files'
        for cat in self.categories:
            off_file = os.path.join(off_path, f'{cat}.off')
            assert os.path.isfile(off_file)
            self.full_off_files[cat] = off_file

        if return_dist:
            dist_path = os.path.join(data_root, 'null', 'dist')
            assert os.path.isdir(dist_path), f'Invalid path {dist_path} without .mat files'
            for cat in self.categories:
                dist_file = os.path.join(dist_path, f'{cat}.mat')
                assert os.path.isfile(dist_file)
                self.full_dist_files[cat] = dist_file

        # load partial shape files
        self._size = 0
        off_path = os.path.join(data_root, cut_type, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files.'
        for cat in self.categories:
            partial_off_files = sorted(glob(os.path.join(off_path, f'*{cat}*.off')))
            assert len(partial_off_files) != 0
            self.partial_off_files[cat] = partial_off_files
            self._size += len(partial_off_files)

        if self.return_corr:
            # check the data path contains .vts files
            corr_path = os.path.join(data_root, cut_type, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} without .vts files.'
            for cat in self.categories:
                partial_corr_files = sorted(glob(os.path.join(corr_path, f'*{cat}*.vts')))
                assert len(partial_corr_files) == len(self.partial_off_files[cat])
                self.partial_corr_files[cat] = partial_corr_files

    def _get_category(self, index):
        assert index < len(self)
        size = 0
        for cat in self.categories:
            if index < size + len(self.partial_off_files[cat]):
                return cat, index - size
            else:
                size += len(self.partial_off_files[cat])

    def __getitem__(self, index):
        # get category
        cat, index = self._get_category(index)

        # get full shape
        full_data = dict()
        # get vertices
        off_file = self.full_off_files[cat]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        full_data['name'] = basename
        verts, faces = read_shape(off_file)
        full_data['verts'] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            full_data['faces'] = torch.from_numpy(faces).long().cpu()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            full_data = get_spectral_ops(full_data, self.num_evecs, cache_dir=os.path.join(self.data_root, 'null',
                                                                                           'diffusion'))
        
        if self.return_elas_evecs:
            full_data = get_elas_spectral_ops(full_data, num_evecs=self.num_evecs, bending_weight=self.bending_weight, cache_dir=os.path.join(self.data_root, 'null', 'elastic'))

        # get geodesic distance matrix
        if self.return_dist:
            dist_file = self.full_dist_files[cat]
            mat = sio.loadmat(dist_file)
            full_data['dist'] = torch.from_numpy(mat['dist']).float()

        # get partial shape
        partial_data = dict()
        # get vertices
        off_file = self.partial_off_files[cat][index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data['name'] = basename
        verts, faces = read_shape(off_file)
        partial_data['verts'] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            partial_data['faces'] = torch.from_numpy(faces).long().cpu()

        # get eigenfunctions/eigenvalues
        if self.return_evecs:
            partial_data = get_spectral_ops(partial_data, self.num_evecs,
                                            cache_dir=os.path.join(self.data_root, self.cut_type, 'diffusion'))
        
        if self.return_elas_evecs:
            partial_data = get_elas_spectral_ops(partial_data, num_evecs=self.num_evecs, bending_weight=self.bending_weight, cache_dir=os.path.join(self.data_root, self.cut_type, 'elastic'))

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.partial_corr_files[cat][index], dtype=np.int32) - 1
            full_data['corr'] = torch.from_numpy(corr).long()
            partial_data['corr'] = torch.arange(0, len(corr)).long()

        return {'first': full_data, 'second': partial_data}

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class PairTopKidsDataset(Dataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_dist=False, return_elas_evecs=False, bending_weight=1e-2, cache=True):
        assert phase in ['train', 'test'], f'Invalid phase: {phase}'
        self.dataset = SingleTopKidsDataset(data_root, phase, return_faces, return_evecs, num_evecs, return_dist, return_elas_evecs, bending_weight, cache)
        self.phase = phase
        if phase == 'test':
            corr_path = os.path.join(data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))
            self._size = len(self.corr_files)
        else:
            self.combinations = list(product(range(len(self.dataset)), repeat=2))
            self._size = len(self.combinations)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.phase == 'train':
            # get index
            first_index, second_index = self.combinations[index]
        else:
            # extract pair index
            first_index, second_index = 0, index + 1

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        if self.phase == 'test':
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['first']['corr'] = torch.from_numpy(corr).long()
            item['second']['corr'] = torch.arange(0, len(corr)).long()

        return item
