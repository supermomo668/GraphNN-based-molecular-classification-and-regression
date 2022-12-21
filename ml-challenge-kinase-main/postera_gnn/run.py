import os, pandas as pd, numpy as np

from postera_gnn.data.dataset import Molecule_pKa_dataloader, graph_datalist_from_smiles_and_labels
#from postera_gnn.data.featurize import graph_datalist_from_smiles_and_labels
from postera_gnn.model.models import *
from postera_gnn.train.train import train_model
from postera_gnn.train.evaluate import eval_model
from postera_gnn.data.data_selection import make_splits

from postera_gnn.data.util import json_to_list, list_to_json

from .settings import (FeaturesArgs, TrainArgs, ModelArgs, DataArgs)

def train(df_data, DataArgs, TrainArgs, model_save_path=TrainArgs.model_save_pt):
  """train a model by defined settings in TrainArgs"""
  # get dataloader
  dataloader = Molecule_pKa_dataloader(
    df=df_data, batch_size=TrainArgs.batch_size, num_workers=DataArgs.num_workers, device=TrainArgs.device, shuffle=True)
  # Model
  if ModelArgs.model=="GIN":
    model = GIN
  elif ModelArgs.model not in ModelArgs.available_models:
    raise Exception(
      f"Model not one of:{ModelArgs.available_models}")
    
  gnn_model = model(FeaturesArgs.n_node_features, 1)
  gnn_model.apply(init_weights_xavier)
  # define loss function
  criterion = nn.MSELoss()
  # define optimiser
  optimizer = torch.optim.Adam(
      gnn_model.parameters(), lr = TrainArgs.lr, 
      weight_decay=TrainArgs.weight_decay) 
  # train model
  gnn_model, train_hist = train_model(
    gnn_model=gnn_model, dataloader=dataloader, criterion=criterion, optimizer=optimizer,
    epochs=TrainArgs.epochs, device=TrainArgs.device)
  if not model_save_path is None:
    torch.save(gnn_model.state_dict(), model_save_path)
    print(f"Model checkpoint saved to :{model_save_path}")
  return gnn_model, train_hist

def evaluate(model, df_test, model_path=None):
  test_dataloader = Molecule_pKa_dataloader(
    df=df_test, batch_size=TrainArgs.batch_size, 
    num_workers=DataArgs.num_workers, device=DataArgs.device, shuffle=False)
  if not model_path is None:
    print(f"Loading weights from path: {model_path}")
    model.load_state_dict(torch.load(model_path))
  model.eval()
  criterion = nn.MSELoss()
  perf = eval_model(
    gnn_model=model, dataloader=test_dataloader, criterion=criterion, device=TrainArgs.device)
  return perf

def KFoldCV(df_data,  DataArgs, TrainArgs, k_fold:int=1, target_test_size:float=0.2) -> list:
  df_splits = make_splits(df_data, test_size=target_test_size)
  cv_perf = []
  k = len(df_splits)
  for k in range(k_fold):
    print(f"Fold split no.: {k}")
    train_set, test_set = pd.concat([
      df_ for i, df_ in enumerate(df_splits) if i!=k]), df_splits[k]
    gnn_model, train_history = train(
      train_set, DataArgs, TrainArgs, device=TrainArgs.device, model_save_path=None)
    cv_perf.append(evaluate(gnn_model, test_set))
  print(f"Mean loss:{np.mean([d['loss'] for d in cv_perf])}")
  return cv_perf

def infer(input_smiles, model, save_to:str=None):
  model.to(TrainArgs.device)
  df_infer = pd.DataFrame(data={'SMILES':input_smiles, 
                                'measurement_value':[0]*len(input_smiles)})
  infer_dataloader = Molecule_pKa_dataloader(
    df=df_infer, batch_size=TrainArgs.batch_size, 
    num_workers=DataArgs.num_workers, device=DataArgs.device, shuffle=False)
  out_preds = []
  for data in infer_dataloader:
    data.to(TrainArgs.device)
    y_pred = model(data.x, data.edge_attr, data.edge_index, data.batch)
    out_preds+= y_pred.flatten().detach().tolist()
  if not save_to is None: 
    pass
  return out_preds


    