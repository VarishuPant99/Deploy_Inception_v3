from ..model import getModel
from ..data import Cifar10Data
from tqdm import trange
import torch
from functools import partial

def train(model,
          data_loader ,
          criterion ,
          optimizer ,
          num_epochs=5 ,
          save_model_filename="saved_weights.pt",
          log_filename="training_logs.txt"):

    logger = partial(logger,log_filename)    

    best_val_loss = float("inf")
    for epoch in trange(num_epochs,desc="Epochs"):
        result = []
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
                    best_val_loss = epoch_loss
                    torch.save(model.module.state_dict(),f"./src/saved_weights/{save_model_filename}")
        result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    looger(result)

def logger(filename = "training_logs.txt" , *args,**kwargs):
    print(*args,**kwargs)
    with open(f"./src/training_logs/{filename}","a") as f:  # appends to file and closes it when finished
        print(file=f,*args,**kwargs)

