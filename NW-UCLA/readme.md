## 流程

1. 解压数据集：将`all_sqe.zip`在`data`目录下解压，这一操作会自然的创建出一个子目录`all_sqe`,也就是说，这一操作后，您的目录结构应该是
```
data
└───all_sqe
    ├───a01_s01_e00_v01.json
    ├───a01_s01_e00_v02.json
```
2. 数据集处理为npy格式(joint模态)：运行`python gen_joint.py`得到joint模态数据
3. 数据集处理出bone模态数据（可选）：运行`python gen_modal.py --modal bone`得到bone模态数据
4. 数据集处理出motion模态数据（可选）：运行`python gen_modal.py --modal motion`得到motion模态的数据
5. bone模态与joint模态合并（可选）：运行`python gen_modal.py --modal jmb`得到合并模态的数据
6. 最终你会得到如下所展示的目录结构与文件
```
data
└───all_sqe
    ├───a01_s01_e00_v01.json
    ├───a01_s01_e00_v02.json
└───npy
    ├── train_label.npy
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── val_label.npy
    ├── val_bone_motion.npy
    ├── val_bone.npy
    ├── val_joint_bone.npy
    ├── val_joint_motion.npy
    ├── val_joint.npy
```