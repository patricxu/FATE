from collections import namedtuple
from grakel import Graph
import numpy as np
import pandas as pd
from federatedml.statistic.data_overview import with_weight
from federatedml.nn.dataset.base import Dataset
from torch_geometric.data import Data
import torch
from federatedml.util import LOGGER


class GraphClassification(Dataset):

    """
     A Graph Dataset includes feature table, edge table and input_nodes table. The data come from a given csv path, or transform from FATE DTable

     Parameters
     ----------
     label_col str, name of label column in csv, if None, will automatically take 'y' or 'label' or 'target' as label
     feature_dtype dtype of feature, supports int, long, float, double
     label_dtype: dtype of label, supports int, long, float, double
     label_shape: list or tuple, the shape of label
     flatten_label: bool, flatten extracted label column or not, default is False
     """

    def __init__(
            self,
            label_col=None,
            feature_dtype='float',
            label_dtype='long',
            label_shape=None,
            flatten_label=False):

        super(GraphClassification, self).__init__()
        self.key2idx: dict = {}
        self.f_dtype = self.check_dtype(feature_dtype)
        self.l_dtype = self.check_dtype(label_dtype)
        self.data: Data = Data()

        # ids, match ids is for FATE match id system
        self.sample_ids = None

    def __len__(self):
        return self.input_cnt

    # def __getitem__(self, item):
    #     return self.x[item]

    @staticmethod
    def check_dtype(dtype):

        if dtype is not None:
            avail = ['long', 'int', 'float', 'double']
            assert dtype in avail, 'available dtype is {}, but got {}'.format(
                avail, dtype)
            if dtype == 'long':
                return torch.int64
            if dtype == 'int':
                return torch.int32
            if dtype == 'float':
                return torch.float32
            if dtype == 'double':
                return torch.float64
        return dtype

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    def __process_feats(self, data_feats):
        LOGGER.info("processing feats")
        LOGGER.info(data_feats.__dict__['_schema']['header'])
        
        subg_column = data_feats.__dict__['schema']['header'].index("subg")
        subgdict = {}

        cnt = data_feats.count()
        if cnt <= 0:
            raise ValueError("empty data")        

        for k, inst in data_feats.collect():
            subgid = inst.features[subg_column]
            if (subgid not in subgdict):
                subgdict[subgid] = {'key2idx':{}, 'x': [], 'edge_index':[], 'edge_attr':[], 'y':None}
            subgdict[subgid]['key2idx'][k] = len(subgdict[subgid]['key2idx'])
            subgdict[subgid]['x'].append(inst.features)

        for subgid in subgdict.keys():
            subgdict[subgid]['x'] = torch.tensor(np.delete(np.array(subgdict[subgid]['x']), subg_column, axis=1), dtype=self.f_dtype)

        self.subgdict = subgdict


    def __process_adj(self, data_adj):
        LOGGER.info("processing edges")
        subgdict = self.subgdict

        edges = data_adj.collect()
        for _, v in edges:
            src, dst, subg = v.features[0:3]
            values = v.features[3:]
            if subg not in subgdict:
                raise ValueError("subg unmatch!")

            subgdict[subg]['edge_index'].append([int(src), int(dst)])
            subgdict[subg]['edge_attr'].append(values)
           
        for subgid in subgdict.keys():
            subgdict[subgid]['edge_index'] = torch.tensor(subgdict[subgid]['edge_index'], dtype=torch.long).T
            subgdict[subgid]['edge_attr'] = torch.tensor(subgdict[subgid]['edge_attr'], dtype=torch.float)
 
    def __process_graph_labels(self, data_y):
        LOGGER.info("processing graph labels")
        subgdict = self.subgdict
        for _, label, subg in data_y.collect():
            subgdict[subg]['y'] = torch.tensor(label)

        self.labels = {subg: label for _, label, subg in data_y.collect()}

    def load(self, data_inst):
        LOGGER.info("Loading graph data...")
        data_feats, data_adj, data_y = data_inst

        if isinstance(data_feats, str):
            raise ValueError("not supported data")
        else:
            # if is FATE DTable, collect data and transform to array format
            LOGGER.info('collecting FATE DTable')

            self.__process_feats(data_feats)
            self.__process_adj(data_adj)
            self.__process_graph_labels(data_y)

            self.data_list = [Data(x=self.subgdict[subg]['x'], edge_index=self.subgdict[subg]['edge_index'], edge_attr=self.subgdict[subg]['edge_attr'], y=self.subgdict[subg]['y']) for subg in self.subgdict.keys()]


    def get_sample_ids(self):
        return self.sample_ids
