from model import model, loss_func, opt, lr_scheduler, device
from datasetPreparation import train_loader, train_data, val_loader, val_data
from tqdm import tqdm
import torch

EPOCHS = 15
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
best_loss = None

# model test
# print(device)
# input = torch.rand([16, 1, 64, 64], dtype=torch.float32)
# input = input.to(device)
# out = model(input)
# print(out.shape)
# exit()

for epoch in range(EPOCHS):
    model.train()
    avg_train_loss = []
    right_answers = 0
    train_loop = tqdm(train_loader, leave=False)
    for x, target in train_loop:
        x = x.to(device)
        target = target.to(device)

        prediction = model(x)
        lossing = loss_func(prediction, target)

        opt.zero_grad()
        lossing.backward()

        opt.step()

        avg_train_loss.append(lossing.item())
        mean_train_loss = sum(avg_train_loss) / len(avg_train_loss)

        right_answers += (torch.round(prediction) == target).all(dim=1).sum().item()

        train_loop.set_description(f"Train Epoch #{epoch+1}/{EPOCHS}, train_loss={mean_train_loss:.3f}, train_acc={right_answers/len(train_data):.4f}")

    avg_train_acc = right_answers / len(train_data)
    train_loss.append(mean_train_loss)
    train_acc.append(avg_train_acc)

    model.eval()
    with torch.no_grad():
        avg_val_loss = []
        right_answers = 0
        val_loop = tqdm(val_loader, leave=False)
        for x, target in val_loop:
            x = x.to(device)
            target = target.to(device)

            prediction = model(x)
            lossing = loss_func(prediction, target)

            avg_val_loss.append(lossing.item())
            mean_val_loss = sum(avg_val_loss) / len(avg_val_loss)

            right_answers += (torch.round(prediction) == target).all(dim=1).sum().item()
            val_loop.set_description(f"Val Epoch #{epoch + 1}/{EPOCHS}, val_loss={mean_val_loss:.3f}, val_acc={right_answers/len(val_data):.4f}")

        avg_val_acc = right_answers / len(val_data)
        val_loss.append(mean_val_loss)
        val_acc.append(avg_val_acc)

    lr_scheduler.step(mean_val_loss)
    lr = lr_scheduler._last_lr[0]
    lr_list.append(lr)

    print(f"Epoch #{epoch+1}/{EPOCHS}, train_loss={mean_train_loss:.3f}, train_acc={avg_train_acc:.3f}. val_loss={mean_val_loss:.3f}, val_acc={avg_val_acc:.3f}, lr={lr:.3f}")
        
    if best_loss is None:
        best_loss = mean_val_loss
        torch.save(model.state_dict(), 'convRegression-model-params.pt')
        print(f'Model saved on epoch #{epoch+1}')

    if best_loss > mean_val_loss:
        torch.save(model.state_dict(), 'convRegression-model-params.pt')
        print(f'Model saved on epoch #{epoch+1}')
        best_loss = mean_val_loss