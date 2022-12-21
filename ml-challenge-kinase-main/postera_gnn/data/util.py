import json

def onehot_encode(x, features:list):
    """
    Maps input elements x not in features to the last element
    """
    if x not in features: x = features[-1]
    binary_encoding = [int(bool_val) for bool_val in list(
       map(lambda s: x == s, features))]
    return binary_encoding
  
class onehot_encodings:
  ''' encoding class for one hot features'''
  def __init__(self, atom_info_func, features):
    self.atom_info_func = atom_info_func
    self.features = features
  
  #@property
  def onehot_encodings(self, atom):
    return onehot_encode(self.atom_info_func(atom), self.features)
  
  def __len__(self): return len(self.features)

def json_to_list(json_file):
  with open(json_file, 'r') as f:
    input_data = list(json.load(f).values())
  print(f"Loaded json from: {json_file}")
  return input_data

def list_to_json(json_file, input_list):
  with open(json_file, 'w') as f:
    json.dump(input_list, f)
  print(f"Output json saved to: {json_file}")