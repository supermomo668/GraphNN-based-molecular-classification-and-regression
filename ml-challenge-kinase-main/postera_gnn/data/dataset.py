import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader import DataLoader

from postera_gnn.data.featurize import *

class Molecule_pKa(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"Created dataset processed at:{self.processed_paths[0]}")
    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), 
                   self.processed_paths[0])
        
        
def Molecule_pKa_dataloader(
  df, batch_size, num_workers:int=1, shuffle=False, device='cpu', processed_dataset_path='./data'):
  smile_col, label_col = 'SMILES', 'measurement_value'
  x_smiles, y = df[smile_col].to_numpy(), df[label_col].to_numpy()

  # create list of molecular graph objects from list of SMILES x_smiles and list of labels y
  data_list = graph_datalist_from_smiles_and_labels(x_smiles, y)
  dataset = Molecule_pKa(processed_dataset_path, data_list)
  # create dataloader for training
  dataloader = DataLoader(
    dataset = data_list, batch_size = batch_size, num_workers=num_workers,
    collate_fn=lambda x: tuple(
      x_.to(device) for x_ in default_collate(x)), 
    shuffle=shuffle
  )
  return dataloader