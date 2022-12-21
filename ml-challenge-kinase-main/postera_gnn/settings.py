from .data.util import onehot_encodings
from rdkit import Chem
from pathlib import Path
import torch

class FeaturesArgs:
  # encodings information
  available_atoms = [
        'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I',
        'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge',
        'Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
  ]
  chirality = ["CHI_UNSPECIFIED","CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CCW","CHI_OTHER"]
  num_hydrogens = [0, 1, 2, 3, 4, "MoreThanFour"]
  n_heavy_atoms = num_hydrogens
  formal_charges = [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
  hybridisation_type = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
  # Atoms
    # atom encodings
  atom_encoding_lambdas = {
    'available_atoms': onehot_encodings(
      lambda atom: str(atom.GetSymbol()), available_atoms),
    'chirality_type_enc': onehot_encodings(
      lambda atom: str(atom.GetChiralTag()), chirality),
    'hydrogens_implicit': onehot_encodings(
      lambda atom: int(atom.GetTotalNumHs()), num_hydrogens),
    'n_heavy_atoms': onehot_encodings(
      lambda atom: int(atom.GetDegree()), n_heavy_atoms),
    'formal_charge': onehot_encodings(
      lambda atom: int(atom.GetFormalCharge()), formal_charges),
    'hybridisation_type': onehot_encodings(
      lambda atom: str(atom.GetHybridization()), hybridisation_type)
  }
    # atom info
  atom_info_lambdas = {
    'is_in_a_ring_enc': lambda atom: [int(atom.IsInRing())],
    'is_aromatic_enc': lambda atom: [int(atom.GetIsAromatic())],
    'atomic_mass_scaled': lambda atom: [float((atom.GetMass() - 10.812)/116.092)],
    'vdw_radius_scaled': lambda atom: [float(
      (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)],
    'covalent_radius_scaled': lambda atom:[float(
      (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
  }
    # compute node feature length
  n_node_features = sum(map(len, atom_encoding_lambdas.values()))
  n_node_features += len(atom_info_lambdas)
  
  # Bonds encoding info
  bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
  stereo_types = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
    # bond encodings
  bond_encoding_lambdas = {
    'bond_types': onehot_encodings(
      lambda bond: bond.GetBondType(), bond_types),
    'stereo_types': onehot_encodings(
      lambda bond: str(bond.GetStereo()), stereo_types)
  }
    # bond quantity
  bond_info_lambas = {
    'bond_is_conj_enc': lambda bond: [int(bond.GetIsConjugated())],
    'bond_is_in_ring_enc': lambda bond: [int(bond.IsInRing())]
  }
  n_edge_features = sum(map(len, bond_encoding_lambdas.values()))
  n_edge_features += len(bond_info_lambas)
    #
  n_features = n_edge_features + n_node_features
  # Molecule
  # lambda mol: GraphDescriptors.BalabanJ(mol)

class ModelArgs:
  available_models = ["GIN"]
  model = "GIN"

class TrainArgs:
  batch_size=2**5
  lr=5e-3
  weight_decay=5e-4
  epochs = 20
  name = "default-GIN"
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model_save_pt = Path(f"./model/{name}.pth")
  
class InferArgs:
  batch_size=1
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  name = "default-GIN"
  output_path = Path('./output')
  model_save_pt = Path(f"./model/{name}.pth")
  
class DataArgs:
  data_source = Path('data/kinase_JAK.csv')
  num_workers=4
  device = TrainArgs.device