# UAV-human骨架数据预处理

## 依赖库

`numpy tqdm`

*国内注意PIP换源，命令为：`pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple`*

## 数据处理指导

1. 解压数据集：将`Skeleton.zip`在`data`目录下解压，这一操作会自然的创建出一个子目录`Skeleton`,也就是说，这一操作后，您的目录结构应该是
```
data
└───Skeleton
    ├───P000S00G10B10H10UC022000LC021000A000R0_08241716.txt
    ├───P000S00G10B10H10UC022000LC021000A001R0_08241716.txt
```
2. 数据划分: 以下是UAV-human动作识别评价标准，分为CSv1,CSv2。
```
Cross-Subject-v1
In cross-subject-v1 evaluation, we split 119 subjects into training and testing groups. The IDs of training subjects are 0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 106, 110, 111, 112, 114, 115, 116, 117, 118; the remaining subjects are for testing.

Cross-Subject-v2
In cross-subject-v2 evaluation, we split 119 subjects into training and testing groups. The IDs of training subjects are 0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118; the remaining subjects are for testing.
```
因此先运行
```
python split_v1.py 

python split_v2.py 
```
即可得到划分好的数据集。
3. 数据集处理为npy格式(joint模态)：
运行以下命令得到npy格式的数据
```
python generate_data.py --data_path data/v1 

python generate_data.py --data_path data/v2 
```
4. 数据集处理出bone模态数据（可选）：
骨架数据包括原始的关节点数据，还包括根据连通性将关节连接起来的骨骼数据
运行以下命令得到出bone模态npy格式的数据
```
python gen_bone_data.py
```
5. 数据集处理出motion模态数据（可选）：
运行以下命令得到出motion模态的数据
```
python gen_motion_data.py
```
6. bone模态与joint模态合并（可选）：
运行以下命令得到出bone模态的数据
```
python merge_joint_bone_data.py
```
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
    ├── train_data_bone_motion.npy
    ├── train_data_bone.npy
    ├── train_data_joint_bone.npy
    ├── train_data_joint_motion.npy
    ├── train_data_joint.npy
    └── train_label.pkl
    ├── test_label.pkl
    ├── test_data_bone_motion.npy
    ├── test_data_bone.npy
    ├── test_data_joint_bone.npy
    ├── test_data_joint_motion.npy
    ├── test_data_joint.npy
    └── test_label.pkl
```
## 完整流程演示
如果还有问题，请参考或直接运行我们的`data-process.ipynb`文件，该文件内包括代码，运行结果，注释等。

## 在线代码仓库

https://github.com/ben0i0d/Pre_Data