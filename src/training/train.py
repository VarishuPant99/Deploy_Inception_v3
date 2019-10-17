from ..model import getModel
from ..data import Cifar10Data
from tqdm import trange
import torch
from pathlib import Path
from functools import partial

def check_for_dir(*args_path):
    for path in args_path:
        if not path.exists():
            path.mkdir(parents=True)

def delete_file(path):
    if path.exists() and not path.is_dir():
        path.unlink()

def logger(filepath , *args,**kwargs):
    print(*args,**kwargs)
    with open(filepath,"a") as f:  # appends to file and closes it when finished
        print(file=f,*args,**kwargs)

def train_model(model,
          data_loader ,
          optimizer=None,
          criterion =None,
          num_epochs=5 ,
          save_model_filename="saved_weights.pt",
          log_filename="training_logs.txt"):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if criterion is None:
        criterion =torch.nn.CrossEntropyLoss()
    global logger
    log_filename_path = Path("./src/training_logs/")
    save_model_filename_path = Path("./src/saved_weights/")
    check_for_dir(log_filename_path,save_model_filename_path)
    save_model_filename_path = save_model_filename_path/save_model_filename
    log_filename_path = log_filename_path/log_filename
    delete_file(log_filename_path)
    logger = partial(logger,log_filename_path)    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float("inf")
    for epoch in trange(num_epochs,desc="Epochs"):
        result = [f"[ Epochs {epoch} | {num_epochs} ] : "]
        for phase in ['train', 'val']:
            if phase=="train":     # put the model in training mode
                model.train()
            else:     # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0
            for data , target in data_loader[phase]:
                #load the data and target to respective device
                data , target = data.to(device)  , target.to(device)

                with torch.set_grad_enabled(phase=="train"):
                    #feed the input
                    output = model(data)
                    #calculate the loss
                    loss = criterion(output,target)
                    preds = torch.argmax(output,1)

                if phase=="train"  :
                    # backward pass: compute gradient of the loss with respect to model parameters 
                    loss.backward()
                    # update the model parameters
                    optimizer.step()
                    # zero the grad to stop it from accumulating
                    optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == target.data).item()

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)
            if phase =="val":
                if epoch_loss < best_val_loss:
                    logger(f"Saving the current best model. Previous best loss = {best_val_loss} Current best loss = {epoch_loss}")
                    best_val_loss = epoch_loss
                    torch.save(model.module.state_dict(),save_model_filename_path)
            result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        logger(" ".join(result))


