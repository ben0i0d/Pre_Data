# Pre_Data

Data preparation code to provide consistent and high-performance processing

## dataset

1. UAV-Human: Skeleton (action recognition)
2. NTURGB-D：Skeleton (action recognition) [ST-GCN]

## 注意

1. Data preprocessing uses a lot of performance optimizations, the goal of which is to "minimize memory requirements and maximize concurrency performance", and we try to strike a reasonable balance between speed and demand
2. If you are willing to bear the memory consumption, you can replace 'open_memmap' with 'np.load' or 'np.save' to speed up the processing
3. This project has a high demand for I/O to ensure that it works on a medium with high I/O capability
4. Some datasets have different preprocessing patterns in different projects, and to avoid ambiguity, I have indicated the source in "[]".