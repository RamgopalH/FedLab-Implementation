import torch
from torch.utils.data import TensorDataset, DataLoader , Dataset

class SimpleDataset(Dataset):
    # defining values in the constructor
    def __init__(self, data_length = 20, transform = None):
        
        self.transform = transform
        self.len = data_length
    
    # Getting data size/length
    def __len__(self):
        return self.len
    
    def get_dataloader(self, id, batch_size):
        return DataLoader(TensorDataset(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 3], [-2, -6], [4, 5]]) ,
        torch.tensor([[0], [1], [1], [1], [1], [0], [0]])), batch_size, shuffle=True)

class BasicModel(torch.nn.Module):
    def __init__(self): 
        super(BasicModel, self).__init__()
        self.input = torch.nn.Linear(2, 1)
        self.hidden1 = torch.nn.Linear(2, 1)
        self.act_output = torch.nn.Sigmoid()
 
    def forward(self, x):
        x = x.to(torch.float32)

        x = self.input(x)
        # x = self.hidden1(x)
        x = self.act_output(x)
        
        return x
    
model = BasicModel()

# dataloader = SimpleDataset().get_dataloader(1, 2)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.BCELoss()
# model.train()
# print(list(model.parameters()))
# for _ in range(2):
#     for data, target in dataloader:
#         target = target.view(-1, 1)
#         output = model(data)
#         loss = criterion(output, target.to(torch.float32))
#         print("Target:",target,"Ouput:", output)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# print(list(model.parameters()))


