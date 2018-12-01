import box_lib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from IPython.core.debugger import set_trace
import numpy as np



def main():
  np.random.seed(89)
  torch.manual_seed(43)

  train_ds = box_lib.BoxDataset("data/sample/train.txt")
  train_dl = DataLoader(train_ds, batch_size=18, shuffle=True, num_workers=4)

  model = box_lib.Boxes(6,4)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


  N_EPOCHS = 500

  for epoch in range(N_EPOCHS):
      
      # Train
      model.train()  # IMPORTANT
      
      running_loss, correct = 0.0, 0
      for X, y in train_dl:
        #set_trace()
        #X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
          y_ = model(X)
          loss = criterion(y_, y)
          print(loss)

        loss.backward()
        
        for param in model.parameters():
          print(param.grad.data.sum())
        print(model.boxes[0])
        optimizer.step()
        
        # Statistics
        print("    batch loss: "+str(loss.item()))
        running_loss += loss.item() * X.shape[0]

      print("  Train Loss: "+str(running_loss / len(train_dl.dataset)))
      print("  Train Acc:  "+str(correct / len(train_dl.dataset)))
      

if __name__ == "__main__":
  main()
