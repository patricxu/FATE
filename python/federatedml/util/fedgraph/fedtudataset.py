from genericpath import exists
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional
import argparse
import torch
from federatedml.util.fedgraph.fedgraphdataset import FedGraphDataset
from torch_geometric.data import download_url, extract_zip
from torch_geometric.io import read_tu_data
import copy
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.data.in_memory_dataset import nested_iter
from torch_geometric.data.dataset import Dataset
from random import shuffle
import pandas as pd
import numpy as np


class FedTUDataset(Dataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)

    Stats:
        .. list-table::
            :widths: 20 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #graphs
              - #nodes
              - #edges
              - #features
              - #classes
            * - MUTAG
              - 188
              - ~17.9
              - ~39.6
              - 7
              - 2
            * - ENZYMES
              - 600
              - ~32.6
              - ~124.3
              - 3
              - 6
            * - PROTEINS
              - 1,113
              - ~39.1
              - ~145.6
              - 3
              - 2
            * - COLLAB
              - 5,000
              - ~74.5
              - ~4914.4
              - 0
              - 3
            * - IMDB-BINARY
              - 1,000
              - ~19.8
              - ~193.1
              - 0
              - 2
            * - REDDIT-BINARY
              - 2,000
              - ~429.6
              - ~995.5
              - 0
              - 2
            * - ...
              -
              -
              -
              -
              -
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False, 
                 fed_num: int = 2,
                 split: list = [0.6, 0.2]):
        self.name = name.upper()
        self.cleaned = cleaned
        self.fed_num = fed_num
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        # if self.data.x is not None and not use_node_attr:
        #     num_node_attributes = self.num_node_attributes
        #     self.data.x = self.data.x[:, num_node_attributes:]
        # if self.data.edge_attr is not None and not use_edge_attr:
        #     num_edge_attributes = self.num_edge_attributes
        #     self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    # @property
    # def processed_dir(self) -> str:
    #     name = f'processed{"_cleaned" if self.cleaned else ""}'
    #     return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)
        return data

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        print(folder)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

    def save(self, ds, path):
        os.makedirs(path, exist_ok=True)
        for idx, subg in enumerate(ds):
            x = pd.DataFrame(subg.x.numpy(), columns=['x'+str(i) for i in range(subg.x.shape[1])])
            adj = pd.DataFrame(subg.edge_index.T.numpy(), columns=['src', 'dst'])
            y = pd.DataFrame(subg.y.numpy(), columns=['y'])
            if hasattr(subg, "edge_attr"):
                edge_attr = pd.DataFrame(subg.edge_attr.numpy(), columns=['f'+str(i) for i in range(subg.edge_attr.shape[1])])
                adj = pd.concat([adj, edge_attr], axis=1)

            x.to_csv(f'{path}/{idx}-x.csv')
            adj.to_csv(f'{path}/{idx}-adj.csv')
            y.to_csv(f'{path}/{idx}-y.csv')
                
    def save2(self, ds, path):
        os.makedirs(path, exist_ok=True)
        for idx, subg in enumerate(ds):
            if idx == 0:
                x = pd.DataFrame(subg.x.numpy(), columns=['x'+str(i) for i in range(subg.x.shape[1])])
                x['subg'] = [idx] * len(x)
                x['id'] = x.index
                adj = pd.DataFrame(subg.edge_index.T.numpy(), columns=['src', 'dst'])
                adj['subg'] = [idx] * len(adj)
                y = pd.DataFrame(subg.y.numpy(), columns=['y'])
                y['subg'] = [idx] * len(y)
                if hasattr(subg, "edge_attr"):
                    edge_attr = pd.DataFrame(subg.edge_attr.numpy(), columns=['f'+str(i) for i in range(subg.edge_attr.shape[1])])
                    adj = pd.concat([adj, edge_attr], axis=1)
            else:
                tmp_x = pd.DataFrame(subg.x.numpy(), columns=['x'+str(i) for i in range(subg.x.shape[1])])
                tmp_x['subg'] = [idx] * len(tmp_x)
                tmp_x['id'] = tmp_x.index
                tmp_adj = pd.DataFrame(subg.edge_index.T.numpy(), columns=['src', 'dst'])
                tmp_adj['subg'] = [idx] * len(tmp_adj)
                tmp_y = pd.DataFrame(subg.y.numpy(), columns=['y'])
                tmp_y['subg'] = [idx] * len(tmp_y)
                if hasattr(subg, "edge_attr"):
                    tmp_edge_attr = pd.DataFrame(subg.edge_attr.numpy(), columns=['f'+str(i) for i in range(subg.edge_attr.shape[1])])
                    tmp_adj = pd.concat([tmp_adj, tmp_edge_attr], axis=1)
                
                x = pd.concat([x, tmp_x], axis=0)
                adj = pd.concat([adj, tmp_adj], axis=0)
                y = pd.concat([y, tmp_y], axis=0)                   

        x.to_csv(f'{path}/x.csv', index=False)
        adj.to_csv(f'{path}/adj.csv')
        y.to_csv(f'{path}/y.csv')
                

    def process(self):
        self.data, self.slices, _ = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # fed split
        fed_dataset = [[] for _ in range(self.fed_num)]
        for idx in range(self.len()):
            fed_dataset[idx % self.fed_num].append(self.get(idx))

        # trn-val-test split
        trn_pct= self.split[0]
        val_pct = trn_pct + self.split[1]
        for cls_idx, cluster in enumerate(fed_dataset):
            shuffle(cluster)
            train = cluster[:int(len(cluster)*trn_pct)]
            vali = cluster[int(len(cluster)*trn_pct):int(len(cluster)*val_pct)]
            test = cluster[int(len(cluster)*val_pct):]

            #output
            path = "/".join([self.root, self.name, str(cls_idx), 'train'])
            self.save2(train, path)
            
            path = "/".join([self.root, self.name, str(cls_idx), 'vali'])
            self.save2(vali, path)

            path = "/".join([self.root, self.name, str(cls_idx), 'test'])
            self.save2(test, path)            


if __name__ == "__main__":
    parser = argparse.ArgumentParser("process the pytorch-geometric dataset for federation.")
    parser.add_argument("--name", type=str, help="name of the PYG dataset", required=True)
    parser.add_argument("--root", type=str, help="path to the processed data")
    parser.add_argument("--fed_num", type=int, help="the number of fed cluster", required=True)
    args = parser.parse_args()

    tmp = FedTUDataset(
        root = args.root,
        name = args.name,
        fed_num = args.fed_num,
    )

    # print(len(tmp))
    # print(tmp[0])
    # print([i for i  in tmp])