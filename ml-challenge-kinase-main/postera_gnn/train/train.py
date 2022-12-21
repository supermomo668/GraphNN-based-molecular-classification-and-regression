import numpy as np


class TrainArgs:
  batch_size=2**5
  lr=5e-3
  weight_decay=5e-4
  epochs = 20
    
def train_epoch(gnn_model, dataloader, criterion, optimizer,
                device):
  gnn_model.to(device)
  # set model to training mode
  gnn_model.train()
  losses = []
  # loop over minibatches for training
  for (k, data) in enumerate(dataloader):
    # compute current value of loss function via forward pass
    data.to(device)
    out_gnn = gnn_model(
      data.x, data.edge_attr, data.edge_index, data.batch)
    l = criterion(out_gnn.flatten(), data.y)
    losses.append(l.item())
    # set past gradient to zero
    optimizer.zero_grad()
    # compute current gradient via backward pass
    l.backward()
    # update model weights using gradient and optimisation method
    optimizer.step()
  return {'loss': np.mean(losses)}
    
def train_model(gnn_model, dataloader, criterion, optimizer, epochs:int, device):
  # loop over training epochs
  hist = {k:[] for k in ['loss']}
  for epoch in range(epochs):
    ep_hist = train_epoch(gnn_model, dataloader, criterion, optimizer, device)
    hist['loss'].append(ep_hist['loss'])
    print(f"Epoch {epoch} loss: {ep_hist['loss']:.2f}")
  return gnn_model, hist