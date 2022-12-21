import numpy as np


class EvalArgs:
  batch_size=2**5
  
def eval_model(gnn_model, dataloader, criterion, device):
  # set model to eval mode
  gnn_model.to(device)
  gnn_model.eval()
  losses = []
  # loop over minibatches for training
  for (k, data) in enumerate(dataloader):
    # compute current value of loss function via forward pass
    data.to(device)
    out_gnn = gnn_model(
      data.x, data.edge_attr, data.edge_index, data.batch)
    l = criterion(out_gnn.flatten(), data.y)
    losses.append(l.item())
  return {'loss': np.mean(losses)}