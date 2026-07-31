## FoundAd复现

### step 1 下载权重

下载dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth到当前目录下

下载权重到dinov3-vitb16-pretrain-lvd1689m目录下

### step 2 生成训练数据集

```python
 #FoundAd是在正常样本训练，我这里选择4-shot去训练
 #公有数据集
 python .\foundad\src\sample.py `
   "source=../public" `
   "target=./fewshot/mvtec_4shot_s42" `
   seed=42 `
   num_samples=4
 
 #私有数据集(可以多一点)
 python .\foundad\src\sample.py `
   "source=../all-lpt" `
   "target=./fewshot/private_4shot_s42" `
   seed=42 `
   num_samples=4
```

### step 3训练模型

```python
 #公有数据集
 python ./foundad/main.py  mode=train  "devices=[cuda:0]"  dist.backend=gloo  data.dataset=mvtec  data.data_name=mvtec_4shot_s42  "data.data_path=./fewshot"   data.batch_size=4  app=train_dinov3  "app.meta.encoder_path=/data/sc_011/yw_seg/FoundAD/dinov3-vitb16-pretrain-lvd1689m"   diy_name=_s42  optimization.epochs=1000
 
 #私有数据集
 python .\foundad\main.py `
   mode=train `
   "devices=[cuda:0]" `
   dist.backend=gloo `
   data.dataset=private `
   data.data_name=private_4shot_s42 `
   "data.data_path=./fewshot" `
   data.batch_size=4 `
   app=train_dinov3 `
   "app.meta.encoder_path=D:/sclead/FoundAD/dinov3-vitb16-pretrain-lvd1689m"  `
   diy_name=_s42 `
   optimization.epochs=500
```

### step 4 评估模型

```python
 #测试公有数据集
 python .\foundad\main.py `
   mode=AD `
   "devices=[cuda:0]" `
   dist.backend=gloo `
   data.dataset=mvtec `
   data.data_name=mvtec_4shot_s42 `
   "data.test_root=../public" `
   data.normal_dir=good `
   diy_name=_s42 `
   app=test `
   app.ckpt_step=5300 `
   testing.K_top=10 `
   testing.segmentation_vis=false
 
 #测试私有数据集
 python .\foundad\main.py `
   mode=AD `
   "devices=[cuda:0]" `
   dist.backend=gloo `
   data.dataset=private `
   data.data_name=private_4shot_s42 `
   "data.test_root=../all-lpt" `
   data.normal_dir=good `
   diy_name=_s42 `
   app=test `
   "app.ckpt_step=25000" `
   testing.K_top=10 `
   testing.segmentation_vis=false
```

### step 5 复现情况

公有数据集复现完成（未达到论文的结果）
