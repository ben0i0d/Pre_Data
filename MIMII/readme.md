## 流程

1. 解压数据集：根据`mimii_url.txt`将数据下载到`data`，运行`bash extract.sh`在`data`目录下解压，这一操作会自然的创建出三个子目录,也就是说，这一操作后，您的目录结构应该是
```
data
└───-6dB
    ├───fan
    ├───pump
├───0dB
└───6dB
```
2. 数据集处理为pkl格式：运行`python gen_mfcc.py`得到mfcc特征数据
3. 最终你会得到如下所展示的目录结构与文件
```
data
└───-6dB
    ├───fan
    ├───pump
└───data.pkl
```