## AnomalyCLIP复现

### step 1下载权重

下载ViT-L-14.pt到当前目录

### step1 验证环境可行性

```python
 ## 在预训练权重下验证可行性
 python test.py `
   --dataset custom `
   --data_path  ../all-lpt `
   --save_path .\results\private_zero_shot `
   --checkpoint_path .\checkpoints\9_12_4_multiscale\epoch_15.pth `
   --features_list 24 `
   --image_size 518 `
   --depth 9 `
   --n_ctx 12 `
   --t_n_ctx 4 `
   --metrics image-pixel-level
```

### step2 训练模型

```python
 ## 需要在visa数据上训练，然后在mvtecAD上验证
 python train.py `
   --dataset visa `
   --train_data_path ../visa `
   --save_path .\checkpoints\retrain_visa `
   --features_list 24 `
   --image_size 518 `
   --batch_size 8 `
   --epoch 15 `
   --learning_rate 0.001 `
   --print_freq 1 `
   --save_freq 1 `c
   --depth 9 `
   --n_ctx 12 `
   --t_n_ctx 4 `
   --seed 111
 
 ## 私有数据集训练
 python train_custom.py `
   --dataset custom `
   --train_data_path ../all-lpt `
   --save_path .\checkpoints\private `
   --features_list 24 `
   --image_size 518 `
   --batch_size 4 `
   --epoch 15 `
   --learning_rate 0.001 `
   --print_freq 1 `
   --save_freq 1 `
   --depth 9 `
   --n_ctx 12 `
   --t_n_ctx 4 `
   --seed 111
```

### step3评估模型

```python
 ## 私有数据集上测试指标，公有数据集测试命令同step1
 python test.py `
   --dataset custom `
   --data_path D:\datasets\private_test `
   --save_path .\results\private_finetuned `
   --checkpoint_path .\checkpoints\private\epoch_15.pth `
   --features_list 24 `
   --image_size 518 `
   --depth 9 `
   --n_ctx 12 `
   --t_n_ctx 4 `
   --metrics image-pixel-level
```

### step4 复现情况

公有数据集复现完成（达到论文的结果）
