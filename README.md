# Pre_Data

Data preparation code to provide consistent and high-performance processing

## dep
`numpy tqdm numba joblib scipy`

## dataset

1. UAV-Human: Skeleton  --> `N C T V M`
2. NTURGB-D：Skeleton [ST-GCN]  --> `N C T V M`
3. NTURGB-D：Skeleton [CTR-GCN]
4. NW-UCLA：Skeleton  --> `N C T V M`

## Target needs

1. Mem：Try to have 0 burden, and the maximum is not more than 2G(In order for it to run in memory-constrained situations)
2. Time：At least faster than the original code

## Be care

1. Data preprocessing uses a lot of performance optimizations, the goal of which is trying to strike a reasonable balance between speed and demand
2. If you are willing to bear the memory consumption, you can replace 'open_memmap' with 'np.load' or 'np.save' to speed up the processing
3. This project has a high demand for I/O to ensure that it works on a medium with high I/O capability
4. Some datasets have different preprocessing patterns in different projects, and to avoid ambiguity, I have indicated the source in "[]".
5. `N C T V M` is `Num Channel Frames Joint Body`