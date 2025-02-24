from model import model, loss_func, opt, lr_scheduler, device
from datasetPreparation import test_loader, test_data
from tqdm import tqdm
import torch

params = torch.load('convRegression-model-params.pt')
model.load_state_dict(params)

test_loop = tqdm(test_loader, leave=False)
right_answers = 0
test_acc = None
model.eval()
for x, target in test_loop:
    x = x.to(device)
    target = target.to(device)
    prediction = model(x)
    right_answers += (torch.round(prediction) == target).all(dim=1).sum().item()
    test_acc = right_answers / len(test_data)
    test_loop.set_description(f"Model testing test_acc={test_acc:.4f}")

print(f"Final accuracy: {test_acc:.4f}") # 0.9930
