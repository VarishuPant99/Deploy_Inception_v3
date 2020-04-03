# Flask Image Classification

This repo comprises of training DenseNet from scratch on cifar10 and then using Flask for Model serving.

The directory is defined as 

![](Screenshot from 2020-04-03 22-17-17.png)

## For Training

[Run Training Notebook file](TrainingNotebook.ipynb)



## For Inference

 we need Cifar-10 data for inference.

Execute the below to generate 10 random images from cifar-10 test data.

```bash
cd data
python generate_valid_data.py
```



For Serving this model as a website. First generate  test images by executing above commands and then.


```bash
python flask_api/app.py
```
