import os
import numpy as np
import pandas as pd
import torch as th
from utils.function import *
from utils.settings import *
from utils.data.featurization import *
from itertools import repeat, product, chain
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch, InMemoryDataset
from rdkit.Chem import AllChem

def DDIgraph_preprocess(cf):

    dataset = cf.dataset
    filepath_dict = DATA_INFO[dataset]
    smiles2idx = load_vocab(filepath=filepath_dict['simles'])

    if smiles2idx is not None:
        idx2smiles = [''] * len(smiles2idx)
        for smiles, smiles_idx in smiles2idx.items():
            idx2smiles[smiles_idx] = smiles
    else:
        idx2smiles = None

    train, valid, test, num_ddi_type = load_data(filepath_dict, smiles2idx)

    return train, valid, test, idx2smiles, smiles2idx, num_ddi_type


def load_vocab(filepath):
    df = pd.read_csv(filepath, index_col=False)
    smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    return smiles2id

def load_data(filepath, smiles2idx):

    assert smiles2idx is not None
    train_edges, train_labels = load_csv_data(filepath['train'], smiles2idx, is_train_file=True)
    val_edges, val_labels = load_csv_data(filepath['valid'], smiles2idx, is_train_file=False)
    test_edges, test_labels = load_csv_data(filepath['test'], smiles2idx, is_train_file=False)
    labels = train_labels + val_labels + test_labels

    return train_edges, val_edges, test_edges, max(labels)

def load_csv_data(filepath, smiles2idx, is_train_file: bool = True):
    df = pd.read_csv(filepath, index_col=False)
    edges = []
    labels = []
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2idx.keys() and smiles_2 in smiles2idx.keys():
            idx_1 = smiles2idx[smiles_1]
            idx_2 = smiles2idx[smiles_2]
            label = int(row_dict['label'])
        else:
            continue

        if label > 0:
            edges.append((idx_1, idx_2))
            edges.append((idx_2, idx_1))
            labels += [label, label]
        else:
            continue

    edges = th.LongTensor(edges)

    return edges, labels

def load_triple_data(file_path, smiles2idx):
    triples = []
    heads = []
    tails = []
    relations = []
    df = pd.read_csv(file_path, index_col=False)
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2idx.keys() and smiles_2 in smiles2idx.keys():
            h = smiles2idx[smiles_1]
            t = smiles2idx[smiles_2]
            r = int(row_dict['label'])
        else:
            continue

        heads.append(h)
        tails.append(t)
        relations.append(r)
        triples.append((h, r, t))

    entitys = heads + tails
    return triples, np.max(entitys), np.max(relations)


def smiles2molgraph(filepath, smiles_list):

    processed_path = osp.join(filepath['root'], 'processed', filepath['processed'])
    if osp.exists(processed_path):
        data = torch.load(processed_path)
    else:
        data_list = []
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
        for i in range(len(smiles_list)):
            print(i)
            rdkit_mol = rdkit_mol_objs_list[i]
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            data.id = torch.tensor([i])
            data_list.append(data)
        data = Batch.from_data_list(data_list)
        torch.save(data, processed_path)
    return data


class DrugDrugDataset(Dataset):
    def __init__(self,
                 filepath,
                 smiles2idx,
                 dataset='DrugBank',
                 negative_sample_size=1,
                 is_train=False,
                 symmetric=True):
        self.dataset = dataset
        self.filepath = filepath
        self.symmetric = symmetric
        self.is_train = is_train
        self.triples, self.nentity, _ = load_triple_data(self.filepath['train'], smiles2idx)
        self.negative_sample_size = negative_sample_size
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.true_tail[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        if self.symmetric:
            positive_sample_inv = (tail, relation, head)
            negative_sample_inv_list = []
            negative_sample_inv_size = 0

            while negative_sample_inv_size < self.negative_sample_size:
                negative_sample_inv = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                inv_mask = np.in1d(
                    negative_sample_inv,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
                negative_sample_inv = negative_sample_inv[inv_mask]
                negative_sample_inv_list.append(negative_sample_inv)
                negative_sample_inv_size += negative_sample_inv.size

            negative_sample_inv = np.concatenate(negative_sample_inv_list)[:self.negative_sample_size]
            negative_sample_inv = torch.LongTensor(negative_sample_inv)
            positive_sample_inv = torch.LongTensor(positive_sample_inv)

            positive_sample = torch.stack([positive_sample, positive_sample_inv], dim=0)
            negative_sample = torch.stack([negative_sample, negative_sample_inv], dim=0)

        positive_label = th.ones(positive_sample.size(0), dtype=th.long)
        negative_label = th.zeros(negative_sample.size(0), dtype=th.long)

        return positive_sample, negative_sample, positive_label, negative_label

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.cat([_[1] for _ in data], dim=0)
        positive_label = torch.cat([_[2] for _ in data], dim=0)
        negative_label = torch.cat([_[3] for _ in data], dim=0)
        label = torch.cat([positive_label, negative_label], dim=0)
        return positive_sample, negative_sample, label

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)