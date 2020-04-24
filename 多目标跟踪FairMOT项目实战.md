# 多目标跟踪FairMOT项目实战

### 环境准备


#### 基本开发环境
笔者使用的环境是 Ubuntu16.04 系统，Tesla K80 显卡，Python3.7 环境

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586936481449-f25432d4-5f79-4bde-9e6d-e0d28b80b3d7.png)


```bash
# 可以 Fork 原始项目
# git clone https://github.com/ifzhang/FairMOT.git

# 也可以 Fork 笔者的项目
git clone https://github.com/FLyingLSJ/FairMOT.git

# 创建并激活虚拟环境
conda create -n FairMOT
conda activate FairMOT
# 安装必要包
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd FairMOT
pip install -r requirements.txt
```
若出现下图这个错误，运行
```bash
pip install --upgrade cython
pip install -r requirements.txt # 再次运行
```


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586924731666-f2ec72f1-7e6f-43f5-adf3-a116a70c8b9f.png)

```bash
# 编译项目
cd src/lib/models/networks/DCNv2
sh make.sh
```
出现下图的结果，就代表编译好了。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586929836018-b486c843-93cc-46fd-b833-0d900c0b5074.png)


#### 附加环境


如果你想要将结果生成视频，那么就就必须安装 **FFmpeg**


> FFmpeg 是一个免费的开源命令行工具，用于对多媒体文件进行代码转换。它包含一组共享的音频和视频库，例如 libavcodec，libavformat 和 libavutil。使用 FFmpeg，可以在各种视频和音频格式之间转换，设置采样率以及调整视频大小。



在 `FairMOT/src/track.py` 文件中使用到了该命令。


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586998471777-94b66dd7-97b8-4227-851b-4703e3f2f89b.png)


```bash
apt update
apt install ffmpeg
# 输入 y

# 安装验证
ffmpeg -version 
```
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1587010711113-1bed6d99-ffd5-4851-8066-04456fc25d98.png)

[https://www.ffmpeg.org/](https://www.ffmpeg.org/)

[https://www.ffmpeg.org/ffmpeg.html](https://www.ffmpeg.org/ffmpeg.html)

[https://linuxize.com/post/how-to-install-ffmpeg-on-ubuntu-18-04/](https://linuxize.com/post/how-to-install-ffmpeg-on-ubuntu-18-04/)


### 准备数据集

（原始项目所用到的数据集包括 [Caltech Pedestrian](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), [CityPersons]( https://bitbucket.org/shanshanzhang/citypersons ), [CUHK-SYSU](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html), [PRW](http://www.liangzheng.com.cn/Project/project_prw.html), [ETHZ](https://data.vision.ee.ethz.ch/cvl/aess/dataset/), [MOT17](https://motchallenge.net/data/MOT17/), [MOT16](https://motchallenge.net/data/MOT16/).）[2DMOT15](https://motchallenge.net/data/2D_MOT_2015/) 和 [MOT20](https://motchallenge.net/data/MOT20/)，更多数据集[DATASET_ZOO]( https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md )


但是，为了测试，我们只使用到 [2DMOT15](https://motchallenge.net/data/2D_MOT_2015/) 

MOT15 数据集地址：[https://motchallenge.net/data/2D_MOT_2015/](https://motchallenge.net/data/2D_MOT_2015/)


我准备了一个 shell 脚本 `MOT15_dataset_down.sh` 可以帮你自动下载 MOT15 数据集，并且构造成所需要的目录结构，直接运行 `sh MOT15_dataset_down.sh` 即可。
```bash
# 下载并解压
cd ..
wget https://motchallenge.net/data/2DMOT2015.zip
mkdir dataset
cd dataset
mkdir MOT15
cd MOT15
mkdir labels_with_ids
mkdir labels_with_ids/train
unzip ../../2DMOT2015.zip 
mv 2DMOT2015 images
```


将数据集构建成以下结构， `train(empty)` 代表是一个空的文件夹，(empty只是注释，文件夹名称叫 train)
```bash
dataset  
|——————  MOT15
|            |——————images
|            |        └——————train
|            |        └——————test
|            └——————labels_with_ids
|                     └——————train(empty)
|——————  MOT20
|            |——————images
|            |        └——————train
|            |        └——————test
|            └——————labels_with_ids
|                     └——————train(empty)
```

将 seqinfo 文件夹中的 *.ini 复制到  MOT15 对应的文件夹中，用以下脚本完成复制

seqinfo 信息在本项目下面了 [seqinfo 文件](seqinfo-MOT15.zip)，也可以在这里下载  [[Google]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w)  [[Baidu],提取码:8o0w](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g). 

运行下面这个脚本，改变一下对应的路径即可：
```python
import shutil
import os
seqinfo_dir = "./seqinfo" # seqinfo 所在的路径
MOT15_dir = "./dataset/MOT15/images/train/" # 数据集所在的路径
seqs = ['ADL-Rundle-6', 'ETH-Bahnhof', 'KITTI-13', 'PETS09-S2L1', 'TUD-Stadtmitte', 'ADL-Rundle-8', 'KITTI-17',
        'ETH-Pedcross2', 'ETH-Sunnyday', 'TUD-Campus', 'Venice-2']

for seq in seqs:
    src = os.path.join(seqinfo_dir, seq, "seqinfo.ini")
    dst = os.path.join(MOT15_dir, seq, "seqinfo.ini")
    shutil.copy(src, dst)
```


修改  `src/python gen_labels_15.py` 文件中数据集的路径
```python
seq_root = '/dli/dataset/MOT15/images/train'
label_root = '/dli/dataset/MOT15/labels_with_ids/train'
```
然后运行下面程序
```bash
cd src
python gen_labels_15.py

# 没有下载 MOT20 数据集的话就不需要运行下面这句
python gen_labels_20.py  
```


会生成一些 txt 文件
```bash
# 在 MOT15/labels_with_ids/train 文件夹下生成一些 txt 文件
└── train
    ├── ADL-Rundle-6
    │   └── img1
    ├── ADL-Rundle-8
    │   └── img1
    ├── ETH-Bahnhof
    │   └── img1
    ├── ETH-Pedcross2
    │   └── img1
    ├── ETH-Sunnyday
    │   └── img1
    ├── KITTI-13
    │   └── img1
    ├── KITTI-17
    │   └── img1
    ├── PETS09-S2L1
    │   └── img1
    ├── TUD-Campus
    │   └── img1
    ├── TUD-Stadtmitte
    │   └── img1
    └── Venice-2
        └── img1


# 在 MOT20/labels_with_ids/train 文件夹下生成一些 txt 文件
└── train
    ├── MOT20-01
    │   └── img1
    ├── MOT20-02
    │   └── img1
    ├── MOT20-03
    │   └── img1
    └── MOT20-05
        └── img1
```
生成的 txt 内容如下：


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586931302189-bf9418d1-907a-4092-858f-91474ff21c98.png)



更多数据集的含义可以参考：https://mp.weixin.qq.com/s/lkuiGIpY5rjho1zfAdxYPQ

### 下载权重


权重所在的目录结构如下：
```bash
FairMOT
   └——————models
           └——————ctdet_coco_dla_2x.pth
           └——————hrnetv2_w32_imagenet_pretrained.pth
           └——————hrnetv2_w18_imagenet_pretrained.pth
```


下面我使用的是直接用命令从谷歌网盘下载权重，如果你访问不了谷歌的话，那么下面的代码可能无法运行，我也将其做成了一个脚本文件 `model_down_script.sh` 。
```bash
# 在 FairMOT 创建一个 models 目录，用于保存权重文件
mkdir models 
cd models
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT' -O ctdet_coco_dla_2x.pth

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu" \
-O all_dla34.pth && rm -rf /tmp/cookies.txt

```


### 训练
#### 修改部分文件的部分参数


- 修改 `src/lib/opts.py` 文件中的 `data_dir` （数据集路径）和 `data_cfg` （代表的意义就是你使用什么数据集进行训练，在 **src/lib/cfg** 文件夹下有 4 份 json 文件可供选择，其中 data.json 文件中包含的数据集更多）



![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586934564286-8b6c86be-f26e-4ec2-900f-79813e294182.png)


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586934663067-0562e8a2-7c0e-40db-aeb7-4d65dd012305.png)


-  修改 `src/lib/cfg/*.json` 文件中的部分参数（*.json 代表你使用了那份数据进行训练，我们这里选择的是 `mot15.json` ，所以就要修改 mot15.json 这份文件）



修改前后对比如下，其实就是修改 root 参数，改成我们的数据所在的路径就可以了。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586935048622-ddd9ac7a-ae78-4a43-a5b0-2e3bbe200515.png)


#### 训练


如果你只有单 GPU 的话，那么你需要修改 `experiments/all_dla34.sh` 中 GPU个数的参数，单 GPU 就是 0，多GPU 的话是 0 1 2 ... 等

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586935268649-e524cb62-97d2-447d-b12f-747c09cdc118.png)



然后在 FairMOT （也就是项目路径下）运行以下程序即可开始训练

```bash
sh experiments/all_dla34.sh
```
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586935346269-2991f1c5-735c-4c24-b3d4-24436db41858.png)

训练的时候可能会出现以下信息

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586939998152-a7db3511-2a63-4a96-9943-1e7977387a76.png)


这个可以忽略，我在原始项目提了一个 issue [https://github.com/ifzhang/FairMOT/issues/36](https://github.com/ifzhang/FairMOT/issues/36) 作者解答说，这是因为训练的时候，只加载了部分的模型。


### 测试demo


```bash
cd src
python demo.py mot --load_model ../models/all_dla34.pth --conf_thres 0.4
```
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1586931382784-6175c7b8-dee7-411f-b07b-c6133bc32fb0.png)


可以在  FairMOT/results/frame 文件夹下查看视频每一帧的运行结果。


### 跟踪检测测试


```bash
python track.py mot --load_model ../models/all_dla34.pth --conf_thres 0.6
```
可以修改 `src/track.py` 代码后面的几个参数，可以将对视频的预测每一帧保存起来，或者最终保存成一个视频，这个需要 **FFmpeg **的支持

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1587712127547-4888cc56-f7e7-4812-85d0-03b5a7d8bff0.png)



![](https://cdn.nlark.com/yuque/0/2020/png/653487/1587011291813-e538c6ca-e119-4b8b-bb31-ddfd9a4d2b5c.png#)



运行起来以后，可以在数据集下面生成一系列结果




![](https://cdn.nlark.com/yuque/0/2020/png/653487/1587011569800-d124cdc7-8436-4463-99dc-0a0f2e8075ae.png)
### 待完成的工作


笔者只是将项目运行起来，但是对于论文中的细节还没有细看，而且这也是笔者第一做多目标跟踪，所以对其中的原理、数据集的格式等问题还不是很理解。


 TO DO LIST：

- [ ]  多目标跟踪数据集的理解
- [ ]  论文阅读







