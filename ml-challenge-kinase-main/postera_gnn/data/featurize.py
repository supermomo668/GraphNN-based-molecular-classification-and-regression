
# RDkit
from rdkit import Chem
from rdkit.Chem import GraphDescriptors
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import numpy as np, pandas as pd
import torch
from torch_geometric.data import Data

#from .util import onehot_encodings
from ..settings import FeaturesArgs

def get_atom_features(atom, 
                      available_atoms:list = FeaturesArgs.available_atoms,
                      atom_encode_lambdas: dict = FeaturesArgs.atom_encoding_lambdas,
                      atom_info_lambdas: dict = FeaturesArgs.atom_info_lambdas,
                      debug=False
                     ):
  """
  Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
  """  
  if 'hydrogens_implicit' in atom_encode_lambdas:
    available_atoms = ['H'] + available_atoms
  atom_feature_vector = []
  # compute atom features
  for name, atom_encoding in atom_encode_lambdas.items():
    encoding = atom_encoding.onehot_encodings(atom)
    atom_feature_vector += encoding
    if debug: print(f"atom encoding length ({name}): {len(encoding)}")
    # boolean features
  for name, info_func in atom_info_lambdas.items():
    atom_feature_vector += info_func(atom)
    if debug: print(f"atom info ({name}): {info_func(atom)}")
  if debug: print(f"full atom feature:{len(atom_feature_vector)}")
  return torch.Tensor(atom_feature_vector)

def get_bond_features(bond, 
                      bond_encoding_lambdas: dict = FeaturesArgs.bond_encoding_lambdas,
                      bond_info_lambdas: dict = FeaturesArgs.bond_info_lambas,
                      debug=False):
  """
  Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
  """
  bond_feature_vector = []
    # compute bond features
  for name, bond_encoding in bond_encoding_lambdas.items():
    encoding = bond_encoding.onehot_encodings(bond)
    bond_feature_vector += encoding
    if debug: print(f"bond encoding length ({name}): {len(bond_feature_vector)}")
    # boolean features
  for name, info_func in bond_info_lambdas.items():
    bond_feature_vector += info_func(bond)
    if debug: print(f"bond info ({name}): {info_func(bond)}")
    return torch.Tensor(bond_feature_vector)

def smile_to_data(smiles, y_val):
  """ smile to pyg Data components"""
  # convert SMILES to RDKit mol object
  mol = Chem.MolFromSmiles(smiles)
  # get feature dimensions
  n_nodes = mol.GetNumAtoms()
  n_edges = 2*mol.GetNumBonds()

  # construct node feature matrix X of shape (n_nodes, n_node_features)
  for n, atom in enumerate(mol.GetAtoms()):
    atom_features = get_atom_features(atom)
    if n==0: X = torch.zeros((n_nodes, len(atom_features)), dtype=torch.float)
    X[atom.GetIdx(), :] = atom_features
    
  # construct edge index array E of shape (2, n_edges)
  E_ij = torch.stack(
    list(map(
      lambda arr: torch.Tensor(arr).to(torch.long),
      np.nonzero(GetAdjacencyMatrix(mol))))
  )
  # construct edge feature array EF of shape (n_edges, n_edge_features)
  EF = torch.stack([
    get_bond_features(mol.GetBondBetweenAtoms(i.item(), j.item())) for i, j in zip(
      E_ij[0], E_ij[1])]
  )
  # construct label tensor
  y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
  return X, E_ij, EF, y_tensor

        
def graph_datalist_from_smiles_and_labels(
  x_smiles, y):
    """
    Inputs:
      x_smiles [list]: SMILES strings
      y [list]: numerial labels for the SMILES strings
    Outputs:
      data_list [list]: torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    data_list = []
    for (smiles, y_val) in zip(x_smiles, y):
      X, E, EF, y_tensor = smile_to_data(smiles, y_val)
      # construct Pytorch Geometric data object list
      data_list.append(
        Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list
