import torch

model = torch.load("linear_model.pth",weights_only=False)
model.eval()

with torch.no_grad():
    y_pred = model(torch.tensor([[3.0]]))
    print(y_pred)




############
