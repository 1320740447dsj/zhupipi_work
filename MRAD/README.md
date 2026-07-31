## MRAD代码复现

### step1 权重准备

将ViT-L-14-336px.pt权重放置checkpoints/clip下

### step2 数据处理

```python
 ## 生成数据的meta.json
 python .\generate_dataset_json\custom.py --root "../visa"
 
 python .\generate_dataset_json\custom.py --root "../public"
```

### step3 训练模型

```python
 ## 在visa上训练模型
 python .\test.py `
   --model_type mrad-clip `
   --dataset custom `
   --data_path "../visa" `
   --checkpoint_path ".\checkpoints\source_visa\mrad_clip_final.pth" `
   --cache_dir ".\cache\source_visa" `
   --cache_name visa `
   --save_path ".\results\visa_to_mvtec" `
   --metrics image-pixel-level `
   --image_size 518 `
   --seed 111 `
   --device "cuda:0"
  
 ## 在私有数据集训练模型
 python .\train.py `
   --model_type mrad-clip `
   --dataset custom `
   --data_path "../all-lpt" `
   --cache_dir "./cache/source_private" `
   --save_path ".\checkpoints\source_private" `
   --rebuild_cache `
   --batch_size 8 `
   --ft_epochs 1 `
   --clip_epochs 5 `
   --learning_rate 0.0004 `
   --image_size 518 `
   --seed 111 `
   --device "cuda:0"
```

### step4 评估模型

```python
 ### 评估mvtec指标
 python .\test.py `
   --model_type mrad-clip `
   --dataset mvtec `
   --data_path "../public" `
   --checkpoint_path ".\checkpoints\source_visa\mrad_clip_final.pth" `
   --cache_dir ".\cache\source_visa" `
   --cache_name visa `
   --save_path ".\results\visa_to_mvtec" `
   --metrics image-pixel-level `
   --image_size 518 `
   --seed 111 `
   --device "cuda:0"
 ### 私有数据集评估
 python .\test.py `
   --model_type mrad-clip `
   --dataset custom `
   --data_path "../public" `
   --checkpoint_path ".\checkpoints\source_mvtec\mrad_clip_final.pth" `
   --cache_dir ".\cache\source_public" `
   --cache_name mvtec `
   --save_path ".\results\mvtec_to_private" `
   --metrics image-pixel-level `
   --image_size 518 `
   --seed 111 `
   --device "cuda:0"
```

### step5 复现情况

公有数据集复现完成（未达到论文的结果）
