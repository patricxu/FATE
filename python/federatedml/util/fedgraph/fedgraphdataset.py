from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from typing import Callable, Optional
from federatedml.util.fedgraph.splitters.graph.louvain_splitter import LouvainSplitter
from pipeline.backend.pipeline import PipeLine
import pandas as pd


class FedGraphDataset(Dataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, log: bool = True):
        self.fed_dataset = None
        self.files = []
        super().__init__(root, transform, pre_transform, pre_filter, log)

    def len(self) -> int:
        return len(self.fed_dataset)

    def upload(self, party_id, name_space, head, partition, id_delimiter=",", extend_sid=False, role = "guest"):
        print(party_id)
        pipeline_upload = PipeLine().set_initiator(role=role, party_id=party_id).set_roles(guest=party_id)
        for ds_files in self.files:
            for file_path_name in ds_files:
                pipeline_upload.add_upload_data(file=file_path_name,
                                                table_name=file_path_name.split(".")[0].split("/")[-1],
                                                head=head, partition=partition,
                                                namespace=name_space,
                                                id_delimiter=id_delimiter,
                                                extend_sid=extend_sid)
        pipeline_upload.upload(drop=1)

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return self._infer_num_classes(self.data.y)    

    def get(self, idx: int) -> Data:
        return self.fed_dataset[idx]

    def fed_split(self, data, fed_num, splitter_name='louvain', **kwargs):
        if splitter_name=='louvain':
            splitter = LouvainSplitter(fed_num, **kwargs)
        self.fed_dataset = splitter(data)

    def save(self, path=None):
        path = self.root + "/" + self.name if path == None else path
        for idx, d in enumerate(self.fed_dataset):
            tmp = []
            feats = pd.concat([pd.DataFrame(d.y.view(-1, 1).numpy(), columns=['y']), pd.DataFrame(d.x.numpy(), columns=[f'x{i}' for i in range(d.x.shape[1])])], axis=1)
            feats.index.name = 'id'
            feats_file = f'{path}/{self.name}-feats-{idx}.csv'
            print(f"Saving {feats_file}")
            feats.to_csv(feats_file)
            tmp.append(feats_file)

            adj = pd.DataFrame(d.edge_index.T.numpy(), columns=['node1', 'node2'])
            adj.index.name = 'id'
            adj_file = f'{path}/{self.name}-adj-{idx}.csv'
            print(f"Saving {adj_file}")
            adj.to_csv(adj_file)
            tmp.append(adj_file)

            train = pd.DataFrame(d.train_mask.view(-1, 1).numpy(), columns=['mask'])
            train['id'] = train.index
            train = train[train['mask'] == True]
            train_file = f'{path}/{self.name}-train-{idx}.csv'
            print(f"Saving {train_file}")
            train['id'].to_csv(train_file, index=False)
            tmp.append(train_file)

            val = pd.DataFrame(d.val_mask.view(-1, 1).numpy(), columns=['mask'])
            val['id'] = val.index
            val = val[val['mask'] == True]
            val_file = f'{path}/{self.name}-val-{idx}.csv'
            print(f"Saving {val_file}")
            val['id'].to_csv(val_file, index=False)
            tmp.append(val_file)

            test = pd.DataFrame(d.test_mask.view(-1, 1).numpy(), columns=['mask'])
            test['id'] = test.index
            test = test[test['mask'] == True]
            test_file = f'{path}/{self.name}-test-{idx}.csv'
            print(f"Saving {test_file}")
            test['id'].to_csv(test_file, index=False)
            tmp.append(test_file)

            self.files.append(tmp)