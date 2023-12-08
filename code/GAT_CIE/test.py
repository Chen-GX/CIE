import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data

feature = torch.ones((10, 3), dtype=torch.float)

data = Data(x=feature)
print(data.x)
data = T.NormalizeFeatures(data)
print()
print(data.x)
