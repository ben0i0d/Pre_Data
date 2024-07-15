## 依赖库

`numpy tqdm`

## 注意：

1. 完整流程可以直接运行子文件夹下的`ipynb`
2. 国内注意PIP换源，命令为：`pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple`

## 流程

1. 解压数据集：将`Skeleton.zip`在`data`目录下解压，这一操作会自然的创建出一个子目录`Skeleton`,也就是说，这一操作后，您的目录结构应该是
```
data
└───Skeleton
    ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
    ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
```
2. 数据划分: UAV-human动作识别评价标准分为CSv1,CSv2，运行`python split.py`得到划分好的数据集。
3. 数据集处理为npy格式(joint模态)：运行`python gen_joint.py`得到joint模态数据
4. 数据集处理出bone模态数据（可选）：运行`python gen_modal.py --modal bone`得到bone模态数据
5. 数据集处理出motion模态数据（可选）：运行`python gen_modal.py --modal motion`得到motion模态的数据
6. bone模态与joint模态合并（可选）：运行`python gen_modal.py --modal jmb`得到合并模态的数据
7. 最终你会得到如下所展示的目录结构与文件
```
└───data/v1
    ├───train
        ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
        ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
        └───...
    ├───test
        ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
        ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
        └───...
    ├── train_label.pkl
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── test_label.pkl
    ├── test_bone_motion.npy
    ├── test_bone.npy
    ├── test_joint_bone.npy
    ├── test_joint_motion.npy
    ├── test_joint.npy
```