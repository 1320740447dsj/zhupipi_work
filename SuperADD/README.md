## SuperADD代码复现

### step1准备权重文件

下载dinov3-vitb16-pretrain-lvd1689m权重到当前目录下，

下载dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth到./weights目录下

### step1 准备数据集

```python
 ## 处理公共数据集
 python prepare_data.py ^
   --src "D:\sclead\public" ^
   --experiment-dir "D:\sclead\SuperADD\experiments\mvtec_ad" ^
   --update-config
 ## 处理私有数据集
 python prepare_data.py ^
   --src "D:\sclead\lpt-all" ^
   --experiment-dir "D:\sclead\SuperADD\experiments\private" ^
   --update-config
```

### step2 训练模型

```python
 ## 训练共有数据集  /  训练私有数据集
 python tracks\industrial\src\industrial\train.py
```

### step3 评估模型

```python
 ##  评估共有数据集  /  评估私有数据集
 python tracks\industrial\src\industrial\test.py
```

### step4 数据转换

```python
 ## 每次训练前后需要切换数据源
 python prepare_data.py ^
   --experiment-dir "D:\sclead\SuperADD\experiments\private" ^
   --activate-only
```

### step5 复现情况

公有数据集复现完成（达到论文的结果）
