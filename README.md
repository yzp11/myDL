# myDL
This is a PyTorch implementation for Vision Transformer (ViT), ResNet and versions of attention mechanism.

The test results are as follows:

| Model                 | Acc (%) |
| --------------------- | ------- |
| ViT-B/16              | 75.82   |
| ViT-B/16 + SparseAttn | 75.82   |
| ViT-B/16 + LocalAttn  | 63.46   |
| ResNet                | 75.00   |

### 0. Environment

```
python==3.8.10 
torch==2.0.0+cu118
torchvision==0.15.1
numpy==1.24.2    
```

### 1. Dataset

Use *Five Flowers* Dataset to train and test the model.

```
wget https://github.com/Runist/image-classifier-keras/releases/download/v0.2/dataset.zip
unzip dataset.zip
```

Make sure to put the files as the following structure:

```
$data
├── train/
|   ├── daisy
│   ├── ...
├── vallidation/
|   ├── daisy
|   ├── ...
```

### 2. Train

**Train ViT:**

```
cd myDL
bash scripts/train_ViT-B16.sh 
```

If you want to use Sparse Attention or Local Attention, please ***import*** modules in *./models/VisionTransformer/ViT.py*

```
from models.Transformer.mutli_head_sparse_attention import MutliHeadSparseAttention as MutliHeadAttention
#or
from models.Transformer.mutli_head_local_attention import MutliHeadLocalAttention as MutliHeadAttention
```



**Train RN:**

```
cd myDL
bash scripts/train_ResNet-50.sh 
```

### 3. Predict

```
bash scripts/predict_ViT-B16.sh 
#or
bash scripts/predict_ResNet-50.sh 
```

