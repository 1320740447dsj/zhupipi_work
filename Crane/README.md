## Crane代码复现

### step1 下载模型

下载dinov2-main到当前目录

### step2 训练模型

```python
 ###公有数据集
 python train.py `
   --datasets_root_dir ../visa `
   --dataset visa `
   --model_name my_cranep `
   --device 0 `
   --epoch 5 `
   --batch_size 8 `
   --features_list 24 `
   --dino_model dinov2 `
   --why "train on MVTec"
 
 ### 私有数据集
 python train.py `
   --datasets_root_dir ../all-lpt `
   --dataset private `
   --model_name my_cranep `
   --device 0 `
   --epoch 5 `
   --batch_size 8 `
   --features_list 24 `
   --dino_model dinov2 `
   --why "train on private dataset"
```

### step3 评估模型

```python
 python test.py `
   --datasets_root_dir ../public `
   --dataset private `
   --model_name trained_on_mvtec_my_cranep `
   --epoch 5 `
   --devices 0 `
   --features_list 24 `
   --dino_model dinov2 `
   --soft_mean True `
   --visualize False
```

### step4 复现情况

公有数据集复现完成（达到论文的结果）
