## AdapClip复现

说明：需要在A数据集上训练,B数据集上测试，我这里选用的是Visa数据集上训练，Mvtec数据集上测试

### step1 数据集准备

```python
 ##私有数据和公有数据MVTec-AD，生成数据的meta.json文件
 ## zero-shot A数据
 python dataset\mvtec.py --root D:\datasets\mvtec
 
 python dataset\mvtec.py --root D:\datasets\private
```

### step2 训练模型

```python
 ## 训练共有数据集mvtecAD
 python train.py --dataset mvtec --train_data_path D:\datasets\mvtec --save_path .\checkpoint\mvtec_seed10 --pretrained_model "ViT-L/14@336px" --features_list 6 12 18 24 --image_size 518 --batch_size 8 --epoch 15 --save_freq 1 --seed 10 --k_shots 1 --n_ctx 12 --vl_reduction 4 --pq_mid_dim 128 --visual_learner --textual_learner --pq_learner --pq_context
 
 
 ## 训练私有数据集 （根据具体路径修改）
 python train.py --dataset private_mvtec --train_data_path D:/sclead/all-lpt --save_path .\checkpoint\private_seed10 --pretrained_model "ViT-L/14@336px" --features_list 6 12 18 24 --image_size 512 --batch_size 8 --epoch 30 --save_freq 1 --seed 10 --k_shots 0 --n_ctx 12 --vl_reduction 4 --pq_mid_dim 128 --visual_learner --textual_learner --pq_learner --pq_context
```

### step3 评估模型

```python
 ## 指标为auroc，aupro和f1 score3个指标
 python test.py --dataset mvtec --test_data_path D:\sclead\public --checkpoint_path .\checkpoint\mvtec_seed10\epoch_15.pth --save_path .\results\mvtec_seed10 --pretrained_model "ViT-L/14@336px" --features_list 6 12 18 24 --image_size 518 --batch_size 8 --seed 10 --k_shots 1 --n_ctx 12 --sigma 4 --vl_reduction 4 --pq_mid_dim 128 --visual_learner --textual_learner --pq_learner --pq_context --eval_metrics I-AUROC P-AUROC P-AUPRO I-F1max P-F1max
 
 ## 私有数据集
 python test.py --dataset private_mvtec --test_data_path D:\sclead\all-lpt --checkpoint_path .\epoch_11.pth --save_path .\results\private_seed10 --pretrained_model "ViT-L/14@336px" --features_list 6 12 18 24 --image_size 518 --batch_size 1 --seed 10 --k_shots 1 --n_ctx 12 --sigma 4 --vl_reduction 4 --pq_mid_dim 128 --visual_learner --textual_learner --pq_learner --pq_context --cpu_evaluation --cpu_prompt_memory --metric_image_size 256 --eval_metrics I-AUROC P-AUROC P-AUPRO I-F1max P-F1max
```

### step4 复现情况

公有数据集复现完成（达到论文的结果）
