# Pre_Data

Data preparation code to provide consistent and high-performance processing

## dataset

1. UAV-Human: Skeleton (action recognition)
2. NTURGB-D：Skeleton (action recognition) [ST-GCN]
2. NTURGB-D：Skeleton (action recognition) [CTR-GCN]

## Target needs

1. Mem：Not greater 16G
2. Time：At least faster than the original code

*We will load data into memory for some operations to speed up, but ensure that the memory requirement is not greater than 16G.*

*16G is not just a project requirement, the test scenario is on a Debian system, and it also starts daily tasks such as browsers*
## 注意

1. Data preprocessing uses a lot of performance optimizations, the goal of which is trying to strike a reasonable balance between speed and demand
2. If you are willing to bear the memory consumption, you can replace 'open_memmap' with 'np.load' or 'np.save' to speed up the processing
3. This project has a high demand for I/O to ensure that it works on a medium with high I/O capability
4. Some datasets have different preprocessing patterns in different projects, and to avoid ambiguity, I have indicated the source in "[]".