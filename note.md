[toc]
# 论文部分
## Similarity Search Combining Query Relaxation and Diversification 
### abstract
contribution：找了个新目标函数来平衡similarity和diversity
### intruduction

#### motivation

#### contribution

## Approximate Nearest Neighbor Search on High Dimensional Data — Experiments, Analyses, and Improvement
### abstract
contribution: 对目前一些主流ANN search的SOTA做了一个比较全面的实验价值评估（comprehensive experimental evaluation）。
### intruduction

#### motivation

#### contribution



## Residual Vector Product Quantization for approximate nearest neighbor search 
### abstract
- 基于RVPQ提出的结构，改善了ANN性能，H-Variable Beam Search算法，nverted Multi-Index （for large scale）


motivation:these methods do not strike a satisfactory balance between accuracy and efficiency because of their defects in quantization structures.  VQ因为quatntization的数据结构缺陷不能达到性能、效率平衡
contribution: RVPQ的量化方法被提出，由有序的residual codebooks组成的residual structure 被构建，它提升了ANN search的性能。
除此之外，H-Variable Beam Search 被提供更高效的encoding效率，furthermore，invertied multi-index based on RVPQ

### intruduction

#### motivation

#### contribution






## CXL-ANNS: Software-Hardware Collaborative Memory Disaggregation and Computation for Billion-Scale Approximate Nearest Neighbor Search
### Abstract
提出*CXL-ANNS*，基于billion规模的ANNS的软硬件合作的内存分离、计算技术。
因为CXL’s far-memory-like characteristics会导致搜索性能显著下降，CXL-ANNS关注到node-relationship层，cache了一些访问最频繁的neibours；unchached的node预抓取了几个最可能访问的by understanding the graph traversing behaviors of ANNS。
然后这里文章说CXL-ANNS同样关注CXL网络内部的硬件协同情况，进行硬件层并行。它还用了relaxes the execution dependency of neighbor search tasks（暂不理解他在说什么）并且最大并行化的程度。
### Intruduction

#### Conclusion

#### details

#### Q&A
- 什么是memory disaggregation？
  分离式内存给予主机像访问本地DRAM一样，访问其他主机的空闲内存（或内存池） 的能力
  新型高带宽低延时网络硬件、协议（如 RDMA 网卡[7]）的发展，已经可以提供100Gbps带宽、纳秒级延迟的网络互联，这使得集群尺度下分离式内存池，对比SSD等承 载虚拟内存的介质，在访存延迟（<1μs | >10μs ）和带宽（>100Gbps | <5Gbps）上都有了明显优势。
- 什么是CXL（computer express link）？
  是一种高速串行协议，它允许在计算机系统内部的不同组件之间进行快速、可靠的数据传输。
- what is execution dependency for neighbor tasks
- what is contxt-aware
  似乎是一种找feature的方式，













# 方向
## 偷理解
### 高稠密向量处理（wmz）
主要分三点，向量索引管理，向量查询优化，向量安全防护
#### 索引管理
主流：树、**哈希**、**量化编码**、**图**
树：高维性能骤降
哈希：理论高维很好，实操精度不太行
量化：精度较低，扭曲原始分布
图：高维主流
**HNSW**
高性能ssd
#### 查询优化
近似计算、硬件加速计算、聚集计算

#### 安全
DP（differential privacy）差分隐私
HE（homomorphic encryption）同态加密

局限在稠密向量的加密，与特定索引和查询结合的向量隐私保护技术仍缺乏研究


#### ta的规划


## paper reading list
[label](Similarity%20Search%20Combining%20Query%20Relaxation%20and%20Diversification%20.pdf)新硬件之软硬协同加速： CXL-ANNS: Software-Hardware Collaborative Memory Disaggregation and Computation for Billion-Scale Approximate Nearest Neighbor Search, ATC 2023, KAIST
新硬件之多核处理器加速：iQAN: Fast and Accurate Vector Search with Efficient Intra-Query Parallelism on Multi-Core Architectures, PPoPP 2023, MSR
新硬件之计算型盘加速：VStore: in-storage graph based vector search accelerator, DAC 2022, CAS


新场景之向量动态更新：SPFresh: Incremental In-Place Update for Billion-Scale Vector Search, SOSP 2023, MSRA
新场景之多模查询处理：VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity, OSDI 2023, MSRA
新场景之向量查询引擎： Fast, Approximate Vector Queries on Very Large Unstructured Datasets, NSDI 2023, PKU
新场景之面向知识图谱：High-Throughput Vector Similarity Search in Knowledge Graphs, SIGMOD 2023, Apple

新型向量检索算法：Hierarchical Satellite System Graph for Approximate Nearest Neighbor Search on Big Data，ACM/IMS Transactions on Data Science,  SJTU




# 知识部分
k-NN: K nearest neighbor
### ANN
- Annoy（Approximate Nearest Neighbors Oh Yeah）；参考https://zhuanlan.zhihu.com/p/454511736，二叉树分割
- ScaNN（Scalable Nearest Neighbors）；
- Faiss（Billion-scale similarity search with GPUs）；
- Hnswlib（fast approximate nearest neighbor search）

 Quantization：mapping continuous infinite values to a smaller set of discrete finite values(mathwork这么解释)
Vector quantization: 常用压缩技术

