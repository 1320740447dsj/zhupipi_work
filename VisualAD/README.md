## VisualAD代码复现

训练说明，大概3个epoch即可，不需要训练太多

### step1 模型训练

```python
 ## 在visa数据上训练，mvtec上测试
 python train.py `
   --train_data_path "D:\sclead\visa" `
   --train_dataset visa `
   --save_path ".\checkpoints\visa_seed111" `
   --backbone "ViT-L/14@336px" `
   --epoch 3 `
   --batch_size 2 `
   --device cuda:0 `
   --seed 111
 ##  在私有数据集上训练
 python train.py `
   --train_data_path "D:\sclead\all-lpt" `
   --train_dataset private `
   --save_path ".\checkpoints\private_seed111" `
   --backbone "ViT-L/14@336px" `
   --epoch 3 `
   --batch_size 2 `
   --device cuda:0 `
   --seed 111
```



### step2 评估模型

```python
 python test.py `
   --test_data_path "D:\sclead\public" `
   --test_dataset mvtec `
   --checkpoint_path ".\checkpoints\visa_seed111\final_model.pth" `
   --save_path ".\test_results\visa_to_mvtec_seed111" `
   --device cuda:0 `
   --seed 42
```
