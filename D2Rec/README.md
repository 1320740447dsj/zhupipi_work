### step1 数据准备

```
 需要下载dtd数据集，然后训练模型的Mvtec数据集和私有数据集格式一致即可，dtd数据集地址为：
 通过网盘分享的文件：dtd
 链接: https://pan.baidu.com/s/1favigdAegOVEYLjRW0QSHw 提取码: gbsv 
 --来自百度网盘超级会员v6的分享
```

### step2 训练模型

```python
 ## 训练公有数据集
 python main.py `
   --data_path ../public `
   --dataset mvtec `
   --anomaly_path ../dtd/images `
   --save_path ./checkpoints/public `
   --image_size 224 `
   --batch_size 8 `
   --epoch 30 `
   --seed 111 `
   --dual_mask `
   --mask_head
 ## 训练私有数据集
 python main.py `
   --data_path ../all-lpt `
   --dataset Real-IAD-Variety `
   --anomaly_path ../dtd/images `
   --save_path ./checkpoints/private `
   --image_size 224 `
   --batch_size 8 `
   --epoch 30 `
   --seed 111 `
   --dual_mask `
   --mask_head
```

### step3 评估模型

```python
 ## 公有数据集 
 python main.py -e `
   --data_path  ../public `
   --dataset mvtec `
   --save_path ./checkpoints/public/mvtec-224-8-dualmask-True-maskhead-True `
   --image_size 224 `
   --batch_size 8 `
   --seed 111 `
   --dual_mask `
   --mask_head `
   --eval_metrics I-AUROC P-AUROC P-AUPRO I-F1max P-F1max
 
 ## 私有数据集  
 python main.py -e `
   --data_path  ../all-lpt `
   --dataset Real-IAD-Variety `
   --save_path ./checkpoints/privateReal-IAD-Variety-224-8-dualmask-True-maskhead-True `
   --image_size 224 `
   --batch_size 8 `
   --seed 111 `
   --dual_mask `
   --mask_head `
   --eval_metrics I-AUROC P-AUROC P-AUPRO I-F1max P-F1max
```

### step4 复现情况

公有数据集复现完成（达到论文的结果）
