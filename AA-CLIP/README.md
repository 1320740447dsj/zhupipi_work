## AAClip代码复现

说明，需要在A数据集上训练，B数据集上测试

### step 1 准备数据集

修改dataset/constants.py下面的数据集路径，换到本地路径（默认已经完成了meta.json的生成）

### step2准备模型权重

下载 ViT-L-14-336px.pt权重放到./model/目录下

### step2 训练评估模型

```python
 python train.py  --dataset Private  --training_mode full_shot  --save_path ./ckpt/private
 python train.py  --dataset MVTec  --training_mode full_shot  --save_path ./ckpt/MVTec
 
 python test.py --dataset MVTec   --save_path ./ckpt/mvtec
 python test.py --dataset Private --save_path ./ckpt/private
```