

# yolov5_v6.0_object_detection



目标检测yolov5  v6.0版，pytorch实现，包含了目标检测数据标注，数据集增强，训练自定义数据集全流程。



### 一.环境

```text
    Python >= 3.7
    Pytorch >= 1.5.x
```

## 


## 二.标注工具

 pip install labelImg ==1.8.6

安装完毕后，键入命令：

> labelImg



或者直接打开目录下的  labelImg.exe



------

##### 半自动标注

如果数据集较多，可以先手动标注少量，然后训练出初版模型，然后用初版模型预测进行预标注，最后人工检查。

步骤：

1.将待标注图像放入auto_label/images

2.修改auto_label.py的第62至65行如下的内容：

```
path = r"auto_label/images"   #待标注图片路径
xml_path = r"auto_label/images"    #输出的xml标注文件保存路径
yolo_model_weight='./weight/IDCard_v6x_best.pt'  #模型文件路径
data_conf = './data/custom_data.yaml'  #数据集配置文件路径
```

3.运行auto_label.py 



------

##### 鱼苗目标检测标注数据集分享衔接：

关注公众号datanlp 然后回复关键词 **fry** 获取





### 三.数据集增强

步骤：

1.将标注数据集的标签（xml文件）放入./DataAugForObjectDetection/data/Annotations

2.将标注数据集的图片放入./DataAugForObjectDetection/data/images

3.修改./DataAugForObjectDetection/DataAugmentForObejctDetection.py/中的need_aug_num，即每张图片需要扩增的数量，然后运行./DataAugForObjectDetection/DataAugmentForObejctDetection.py



注意：DataAugmentForObejctDetection_pool.py 是多进程增强版本，耗时较少。代码中的process不宜设置过大否则可能会报错，默认即可。



### 四.数据集格式转换

将` VOC` 的数据集转换成 `YOLOv5` 训练需要用到的格式。

步骤：

1.将标注数据集的标签（xml文件）放入./datasets/Annotations

2.将标注数据集的图片放入./datasets/images

3.将voc_to_coco.py中的class_names改为数据集中标注的类别名称，运行 voc_to_coco.py



------

##### 额外说明

需要生成每个图片对应的 `.txt` 文件，其规范如下：

- 每一行都是一个目标
- 类别序号是零索引开始的（从0开始）
- 每一行的坐标 `class x_center y_center width height` 格式
- 框坐标必须采用**归一化的 xywh**格式（从0到1）。如果您的框以像素为单位，则将`x_center`和`width`除以图像宽度，将`y_center`和`height`除以图像高度。

生成的 `.txt` 例子：

```text
1 0.1830000086920336 0.1396396430209279 0.13400000636465847 0.15915916301310062
1 0.5240000248886645 0.29129129834473133 0.0800000037997961 0.16816817224025726
1 0.6060000287834555 0.29579580295830965 0.08400000398978591 0.1771771814674139
1 0.6760000321082771 0.25375375989824533 0.10000000474974513 0.21321321837604046
0 0.39300001866649836 0.2552552614361048 0.17800000845454633 0.2822822891175747
0 0.7200000341981649 0.5570570705458522 0.25200001196935773 0.4294294398277998
0 0.7720000366680324 0.2567567629739642 0.1520000072196126 0.23123123683035374
```



如果数据标签没生成正确，则会报错

```
mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
  File "D:\app\anaconda3\lib\site-packages\numpy\core\_methods.py", line 40, in _amax
    return umr_maximum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation maximum which has no identity
```



---


### 五.修改数据集配置文件

在 `data/目录下修改数据集配置文件 `custom_data.yaml ，文件内容如下

```yaml

# 目标类型数量，按自己的数据集来改
nc: 3

#目标类型名称，按自己的数据集来改
names: ['person', 'head', 'helmet']
```





### 六.聚类得出先验框（Yolov5 内部已做适配，可选）
步骤：

1.将`./gen_anchors/clauculate_anchors.py`的CLASS_NAMES改为数据集中标注的类别名称

2.运行 `./gen_anchors/clauculate_anchors.py` 





跑完会生成一个文件 `anchors.txt`，里面有得出的建议先验框：

```text
Best Anchors : 
[257, 114, 309, 75, 327, 243]
[439, 59, 469, 347, 488, 117]
[497, 460, 500, 240, 500, 172]
```



### 七.修改模型配置文件

在文件夹 `./models` 下选择一个你需要的模型然后复制一份出来（选择的预训练模型pt文件模型名称必须与模型配置文件yaml对应，否则加载模型会报错），将文件开头的 `nc = ` 修改为数据集的分类数，修改第六步获取的先验框anchors（可选）。

比如，预训练模型是yolov5s.pt,就需要复制一份`./models/yolov5s.yaml`，重命名为custom_yolov5.yaml。

然后修改custom_yolov5.yaml中的 nc和anchors（可选）。

```yaml
# parameters
nc: 1  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [257, 114, 309, 75, 327, 243]  # P3/8
  - [439, 59, 469, 347, 488, 117]  # P4/16
  - [497, 460, 500, 240, 500, 172]  # P5/32

```



### 八.开始训练

1.将预训练模型下载放置在weight目录下；

链接1: 

https://pan.baidu.com/s/18-ywfDdxuTxQ-ZLdvL_bYw  密码: vsp2



链接2:

https://github.com/ultralytics/yolov5/releases



2.修改train.py中的第454行weights预训练模型的路径；

3.修改train.py中的第455行cfg模型配置文件路径

4.修改train.py中的第455行batch-size

5.运行train.py



注意 workers=0 #必须为0

------

注意：

如果代码是 从github 重新clone下来的，需要

注释掉 utils/loggers/__init__.py 的wandb，不然程序会提示你需要注册wandb用户

修改如下：

```
# try:
#     import wandb
#
#     assert hasattr(wandb, '__version__')  # verify package import not local dir
#     if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in [0, -1]:
#         try:
#             wandb_login_success = wandb.login(timeout=30)
#         except wandb.errors.UsageError:  # known non-TTY terminal issue
#             wandb_login_success = False
#         if not wandb_login_success:
#             wandb = None
# except (ImportError, AssertionError):
#     wandb = None
wandb = None
```



train.py开头添加

```
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```



------



```
File "/home/lkz/.virtualenvs/lkztor/lib64/python3.6/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/data/disk2/pact/yolov5_v6.0/models/common.py", line 47, in forward
    return self.act(self.bn(self.conv(x)))
  File "/home/lkz/.virtualenvs/lkztor/lib64/python3.6/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/lkz/.virtualenvs/lkztor/lib64/python3.6/site-packages/torch/nn/modules/conv.py", line 446, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/home/lkz/.virtualenvs/lkztor/lib64/python3.6/site-packages/torch/nn/modules/conv.py", line 443, in _conv_forward
    self.padding, self.dilation, self.groups)
RuntimeError: Unable to find a valid cuDNN algorithm to run convolution

```

显存不足导致

解决办法：减小batchzise





### 训练模型保存

开启训练之后，权重会保存在 `./runs` 文件夹里面的每个 `exp` 文件里面的 weights






# 九. 预测

批量预测步骤：

1.修改predict.py 内第218行weights 模型文件.pt路径，219行source 待预测图像路径，220行data 数据集配置文件路径

2.其他参数可默认，运行predict.py



单张预测步骤：

1.修改detect_image_only.py内第97行至100行的模型路径，数据集配置文件路径等内容

2.运行detect_image_only.py



部署代码简化：

yolo5_inference目录下是清理掉无关代码后的模型部署推理代码，只需关注某几个参数即可。



鱼苗目标检测模型下载

链接: https://pan.baidu.com/s/18-ywfDdxuTxQ-ZLdvL_bYw  密码: vsp2



![348e6d2c1a4b05bbe803453744e56c7e](./blob/main/runs/detect/exp/348e6d2c1a4b05bbe803453744e56c7e.jpeg)







![5f77238d9f6fbb676a2fcd3ebee3dbad](./blob/main/runs/detect/exp/5f77238d9f6fbb676a2fcd3ebee3dbad.jpeg)






# 十. 生成 ONNX
## 安装 `onnx` 库

```shell script
pip install onnx==1.7.0
```

## 执行生成

1.修改export.py 中第477行的数据集配置文件路径和模型文件路径，如下：

```
parser.add_argument('--data', type=str, default='data/custom_data.yaml', help='dataset.yaml path')

parser.add_argument('--weights', nargs='+', type=str, default='weight/best.pt', help='model.pt path(s)')
```



2.运行export.py 

`onnx` 和 `torchscript` 文件会生成在 `./weights` 文件夹中





## 参考

https://github.com/ultralytics/yolov5





机器学习，深度学习算法学习，计算机视觉cv，自然语言处理NLP，人工智能AI资源分享，案例源码分享

关注微信公众号 ： 机器学习算法AI大数据技术

##### 公众号ID：datanlp

扫下方二维码关注公众号

![公众号_机器学习算法AI大数据技术_datanlp](./blob/main/机器学习算法AI大数据技术_datanlp.jpg)

