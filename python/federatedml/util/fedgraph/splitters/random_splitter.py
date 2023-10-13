import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx

import networkx as nx
from federatedml.util.fedgraph.splitters.base_splitter import BaseSplitter


class RandomSplitter(BaseTransform, BaseSplitter):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        return super().__call__(data)