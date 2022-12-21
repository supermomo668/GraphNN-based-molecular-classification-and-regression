from pathlib import Path
import os
from postera_gnn.settings import (FeaturesArgs, TrainArgs, ModelArgs, DataArgs)

# high-level test
class TestClass:
  def test_data_settings(self):
    assert DataArgs.data_source.parent.exists()
  
  def test_package_present(self):
    assert Path('./postera_gnn').exists()
    for subpackage in ['data','model','train']:
      assert Path(f'postera_gnn/{subpackage}').exists()
    assert Path('postera_gnn/run.py').is_file()
    assert Path('postera_gnn/settings.py').is_file()
       
  def test_settings(self):
    assert ModelArgs.model in ModelArgs.available_models
    
  def test_run_modules(self):
    # try:
    from postera_gnn.run import train, evaluate, KFoldCV, infer, json_to_list, list_to_json 
    # except:
    #   assert False
    
  def test_model_modules(self):
    try:
      from postera_gnn.model.models import (SAGEConv, GAT, GCN, GIN, GNNStack)
    except:
      assert False
    
  def test_imports(self):
    try:
      from postera_gnn.data.dataset import Molecule_pKa, Molecule_pKa_dataloader

    except Exception as e:
      assert False
        