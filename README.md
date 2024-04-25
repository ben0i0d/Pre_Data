# Pre_Data

data prepare code

## dataset

1. UAV_Human: Skeleton (action recognition)
2. SHL_2024

## 注意

1. 数据预处理使用了大量的性能优化，优化目标是“尽量减少内存需求，尽量提高并发性能”，我们设法在速度与需求取得一个合理的平衡
2. 如果您愿意承担内存消耗，可以将`open_memmap`替换为`np.load`或者`np.save`来加速处理