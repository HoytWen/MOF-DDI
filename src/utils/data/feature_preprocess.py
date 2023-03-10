from typing import Callable, List, Union

import os
import numpy as np
import pickle
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Registers a features generator.

    :param features_generator_name: The name to call the FeaturesGenerator.
    :return: A decorator which will add a FeaturesGenerator to the registry using the specified name.
    """
    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered FeaturesGenerator by name.

    :param features_generator_name: The name of the FeaturesGenerator.
    :return: The desired FeaturesGenerator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns the names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1-D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


PRETRAINED_SMILES_PATH = '/home/wangyh/data/pretrained_smiles'
MAPPING = None

@register_features_generator('ecfp4')
def ecfp4_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    # mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    # features = np.array([0, 0, 1])
    mapping_filepath = os.path.join(PRETRAINED_SMILES_PATH, 'smiles2ecfp4.pkl')
    with open(mapping_filepath, 'rb') as reader:
        mapping = pickle.load(reader, encoding='latin-1')

    try:
        features = mapping[smiles]
        return features
    except KeyError:
        print('No ECFP4 features for smiles {}'.format(smiles))


@register_features_generator('molenc')
def molenc_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    # mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    # features = np.array([0, 0, 1])
    global MAPPING
    if MAPPING is None:
        mapping_filepath = os.path.join(PRETRAINED_SMILES_PATH, 'smiles2molenc.pkl')
        with open(mapping_filepath, 'rb') as reader:
            mapping = pickle.load(reader)
        MAPPING = mapping
    try:
        features = MAPPING[smiles]
        assert np.shape(features)[0] == 196
        return features
    except KeyError:
        print('No molenc features for smiles {}'.format(smiles))


@register_features_generator('mol2vec')
def mol2vec_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    # mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    # features = np.array([0, 0, 1])
    global MAPPING
    if MAPPING is None:
        mapping_filepath = os.path.join(PRETRAINED_SMILES_PATH, 'smiles2vec.pkl')
        with open(mapping_filepath, 'rb') as reader:
            mapping = pickle.load(reader)
        MAPPING = mapping
    try:
        features = MAPPING[smiles]
        assert features.shape[0] == 300
        return features
    except KeyError:
        print('No mol2vec features for smiles {}'.format(smiles))


@register_features_generator('ssp')
def ssp_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    # mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    # features = np.array([0, 0, 1])
    global MAPPING
    if MAPPING is None:
        mapping_filepath = os.path.join(PRETRAINED_SMILES_PATH, 'smiles2ssp.pkl')
        with open(mapping_filepath, 'rb') as reader:
            u = pickle._Unpickler(reader)
            u.encoding = 'latin1'
            mapping = u.load()
            # mapping = pickle.load(reader)
        MAPPING = mapping
    try:
        features = MAPPING[smiles]
        assert features.shape[0] == 50
        return features
    except KeyError:
        print('No ssp features for smiles {}'.format(smiles))


@register_features_generator('seq2seq')
def seq2seq_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    # mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    # features = np.array([0, 0, 1])
    global MAPPING
    if MAPPING is None:
        mapping_filepath = os.path.join(PRETRAINED_SMILES_PATH, 'smiles2seq.pkl')
        with open(mapping_filepath, 'rb') as reader:
            mapping = pickle.load(reader)
        MAPPING = mapping
    try:
        features = MAPPING[smiles]
        assert features.shape[0] == 512
        return features
    except KeyError:
        print('No seq2seq features for smiles {}'.format(smiles))


@register_features_generator('cddd')
def cddd_features_generator(mol: Molecule) -> np.ndarray:
    # If you want to use the SMILES string
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

    # If you want to use the RDKit molecule
    # mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    # Replace this with code which generates features from the molecule
    # features = np.array([0, 0, 1])
    global MAPPING
    if MAPPING is None:
        mapping_filepath = os.path.join(PRETRAINED_SMILES_PATH, 'smiles2cddd.pkl')
        with open(mapping_filepath, 'rb') as reader:
            mapping = pickle.load(reader)
        MAPPING = mapping
    try:
        features = MAPPING[smiles]
        assert features.shape[0] == 512
        return features
    except KeyError:
        print('No cddd features for smiles {}'.format(smiles))