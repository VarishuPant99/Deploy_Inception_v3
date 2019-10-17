from .densenet import DenseNet
import torch

def getModel(training=False,**kwargs):
    model = DenseNet(**kwargs)
    model.eval()
    if training :
        model.train()
        model = torch.nn.DataParallel(model)
        print("The model is training model")
    print("No of params in model is " , sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model
