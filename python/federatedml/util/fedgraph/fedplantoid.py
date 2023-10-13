import os.path as osp
from torch_geometric.data import download_url
from torch_geometric.io import read_planetoid_data
from typing import Callable, List, Optional
from federatedml.util.fedgraph.splitters.graph.louvain_splitter import LouvainSplitter
from federatedml.util.fedgraph.fedgraphdataset import FedGraphDataset
import argparse
import torch
import numpy as np


class FedPlanetoid(FedGraphDataset):
    r"""
    The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"geom-gcn"`,
            :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
            `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Cora
          - 2,708
          - 10,556
          - 1,433
          - 7
        * - CiteSeer
          - 3,327
          - 9,104
          - 3,703
          - 6
        * - PubMed
          - 19,717
          - 88,648
          - 500
          - 3     
    """    
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'

    def __init__(self, root: str, name: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,  pre_filter: Optional[Callable] = None, log: bool = True, fed_num = 2):
    # def __init__(self,  root: str, name: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True, fed_num = 2):
        self.name = name
        self.fed_num = fed_num
        super().__init__(root, transform, pre_transform, pre_filter, log)

        # print(self.data.train_mask.sum())
        self.split = split
        assert self.split in ['public', 'full', 'random']

        if split == 'full':
            data = self.data
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data = data

        elif split == 'random':
            data = self.data
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True
            self.data = data       

        self.fed_split(self.data, self.fed_num)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)
        self.data = data if self.pre_transform is None else self.pre_transform(data)

    def __repr__(self) -> str:
        return f'{self.fed_dataset}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser("process the pytorch-geometric dataset for federation.")
    parser.add_argument("--name", type=str, help="name of the PYG dataset")
    parser.add_argument("--root", type=str, help="path to the processed data")
    parser.add_argument("--split", type=str, help="the way to split the dataset", default='public')
    parser.add_argument("--fed_num", type=int, help="the number of fed cluster", required=True)
    args = parser.parse_args()
    INF = np.iinfo(np.int64).max

    num_split = {
        'cora': [232, 542, INF],
        'citeseer': [332, 665, INF],
        'pubmed': [3943, 3943, INF],
    }    

    tmp = FedPlanetoid(args.root, args.name, 
            split=args.split, 
            num_train_per_class=num_split[args.name][0],
            num_val=num_split[args.name][1],
            num_test=num_split[args.name][2],
            )

    tmp.save()


