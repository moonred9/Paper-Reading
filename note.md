[toc]
# 论文部分

## Survey of Vector Database Management Systems
### Abstract
vec data management的五个主要障碍：语义相似性模糊、矢量规模大、相似性比较成本高、缺乏可用于索引的自然分区、高效的混合查询比较困难。(semantic similarity, large size of vectors,high cost of similarity comparison, lack of a natural partitioning that can be used to indexing, difficulty of efficiently answer- ing “hybrid” queries)
为克服上述困难，主要在**query processing**,**storage and indexing** and **query optimization and execution**提出了一些思路。
- query processing
  - similarity scores
  - query types <-- 这是什么
- storage and indexing
  - vector compression（量化quantization）
  - partitioning techiniques(基于randomization,learning partitioning, and navigable partitioning)
- query optimization and execution
  - new operators for hybrid queries
  - plan enumeration, plan selection, hardware acceleratied query execution

### Introduction
five key obstacles
- Vague Search Criteria: 模糊的标准。相比标量的布尔代数体系，vec query依赖于模糊的语义相似性。（不过VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity这篇给出了宽松的单调性定义并取得不错的效果，不知道是否可以根据这里的广义单调推导出宽松的体系？）
- Expensive Comparisons: for D-dimensional vec, a comparison typically requires O(D)
- Large size: vector search需要完整的特征向量。一个vec可能跨越多个data pages，磁盘重排代价可能非常高
- Lack of Structure: vec没有明确的“有序性”，sort困难，很难设计indexes
- Incompatibility with Attributes: hybrid search那边，与属性不兼容。缺少既高效又准确的混合查询方法。（Apple那篇似乎是针对特定情境下，特定数据集所附带的atrribute大概都是那几种）

后面的章节概括：
- Query Processing
  how to specify the search criteria in the first place and how to execute search queries
  对于前者，similarity scores, query types, query interfaces
  对于后者, similarity projection, index-supported operators
- Storage and Indexing
  how to organize and store the vec collection
  大部分系统采用vector search indexes(table-based, tree-based, graph-based)
- Optimization and Execution
  主要使用plan enumeration,plan selection and physical execution来优化
- Current Systems
  作者对现存的VDBMS进行分类，主要分为natiev systems; extended systems; search engines and libraries

### Query Processing
主要分为similarity score和query type
#### Similarity Scores
A similarity score $f: \mathbb{R}^D \times \mathbb{R}^D \rightarrow \mathbb{R}$
距离的设计需要遵循一些基本原则
identity $d(\mathbf{a}, \mathbf{a}) = 0$
positivity $d(\mathbf{a}, \mathbf{b}) > 0 \text{,if }\mathbf{a} \neq \mathbf{b} $
triangle inequality $d(\mathbf{a}, \mathbf{b}) \leq d(\mathbf{a}, \mathbf{c}) + d(\mathbf{c}, \mathbf{b})$
多种score方式
- basic scores(一些基本的score公示)
  海明距离，标准内积的距离，余弦相似度，Minkowski($d(a,b)=(\sum_{i=1}^n|a_i - b_i|p)^{1/p}$), Mahalanobis（标准内积空间下加权一个半正定矩阵）
- Aggregate Score
  如人脸识别，可能一个人有不同角度人脸照片，因此有多个feature vector
  对于这类数据需要综合考虑各个特征向量算出的值（比如均值等）
- Learned scores
  
#### Discussion
- Score Selection
  迄今为止没有能够什么定理、规律指导什么情形选择怎样的分数算法；
  对于高维向量而言，除了考量score，vec search还需要考虑查询的语义信息以及embedding本质。
- Curse of Dimensionality
  维度诅咒（貌似是高维空间，欧氏距离度量下向量趋向于稀疏
### Queries and Operators
基础假设
$S \subset \mathbb{R}^D$是vec collection with N members
$\mathbf{q}$是D维query vec，不知道是不是属于S
**Data Manipulation queries**目标是改善S。给定S，q和衡量相似度的方法f，**vector search querries**的目的是返回S的一个子集，要求这个子集的元素都满足一个基于和q相似度标准。
#### Data Manipulation Queries
在S上提供插入删除更新等操作
VDBMS有直接、间接两种操作模式。

#### basic search queries
可以转化为对q的最大相似度/最小距离问题

- **ANN** find a k-size subset $S' \subset S$ such that $d(x', q) \leq c(min_{x\in S}d(x, 1)) $ for all $x' \in S'$
  c是approximation degree
- **Range** Find $\{ x\in S | d(x, 1) \leq r \}$
#### Query Variants 变体
- Predicated Search Queries
  另一个名字似乎是hybrid query，带attribute可以筛选
- Batched Queries
  多个查询一起提交给系统
- Multi-Vector Queries
  对应aggreagate score
#### Basic Operators
(c, k)-search和range queries只需要Projection
- Projection S对g的projection描述如下(在q下)
  $\pi _g(S) = \{ g(x,q) | x \in S \}$

#### Discussion
- Query Accuracy and Performance
  主要使用precision 和 recall评估accuracy(pricision = k' / |S'|，recall = k'/k)  |S'|大小不就是k吗（recall分母理应S中所有满足要求的数量）
  主要使用latency和throughput评估performance
- Theoretical Results
  k-d tree $O(DN^{1 - \frac{1}{D}})$
  LSH(Locality Sensitice Hashing)被认为是最流行的ANN算法 时间$N^\rho$，空间$N^{1+\rho}$。启发式的方法最近也很流行，但是他们缺乏guarantees。基于图的方法最近也很火热，但是没有LSH成熟（会议里那几篇好像都是图的）


### Indexing
indexing本质是把S分区，然后将这些区安排进数据结构里以方便遍历
但因为向量unsortable，因此我们需要一点手段，如*randomization, learned partition- ing, and navigable partitioning*。
- Partitioning Techiniques（分区手段）
  - Randomization. 似乎是使用随机的一些事件帮助区分哪些向量是相似的
  - Learned Partitioning
  - Navigable Partitioning。不拘泥于固定分区
- Storage Techiniques
  - Quantization。节省空间，但是似乎是有损的
  - Disk-Resident Design。在降低比较次数的同时，也降低检索数

#### Tables
construction的复杂度约为$O(DN^{1 + \epsilon})$,查找一般$O(DN^\epsilon)$
- LSH
  通过提供error guarantee来实现可调整性能。但为提高准确性，会产生比较多的冗余度。
  主要思想：高维空间的两点若距离很近，那么设计一种哈希函数对这两点进行哈希值计算，使得他们哈希值有很大的概率是一样的。同时若两点之间的距离较远，他们哈希值相同的概率会很小。
- L2H（Learning to Hash）
  似乎并没有在VDBMS中广泛支持
#### Quantization
LSH很多使用k-means找质心来当hash key。但问题是Large K number使得直接使用k-means代价比较昂贵。
[Product quantization](https://zhuanlan.zhihu.com/p/140548922)，将D- dimension切割成m个小空间

some quantization-based indexes
- Flat Indexes
- IVFSQ
- IVFADC
#### Trees
最重要是如何设计splitting strategy
最自然的想法是based on distance。这些对low D比较有效，但受到维度灾难影响极大
High D，倾向于使用随机来进行node split。FLANN(Fast Library for ANN)将PCA和randomization结合。
一般来说更常用defeatist search，返回包含q的leaf中所有vec作为邻近。查找为$O(DlogN)$，插入平均为$O(DlogN)$，最坏为$O(DN)$，且多次插入后不能平衡节点。
- Non-Random Tree
  从中间切割
- Randome Tree
  当某些维度更容易表现差异性时，实际维度就会小于D，这时候比较容易产生维度诅咒，k-d tree不再使用
  - Principal Component Tree 构造树的时候将S旋转使得axis和主成分的轴对齐
  - Random Projection Tree
#### Graphs
在空间vec上叠加一个图，这样每个空间的点都对应一个图里的点，引导沿边去搜索。
需要考量边的选择
许多图构建依赖初始随机化、随机抽样。
基于图的搜索技术实际表现良好，但是最近的研究表明其渐进性能接近$N^\rho$和$N^{\rho + 1}$（LSH已经达到）
- KNNG
  每个node $v_i$与k个node连接（表示其为最接近的k个邻居）。KNNG可以通过"iterative refine"来得到精确或是近似的结果
  - 精确。通过N次暴搜达到，复杂度$O(DN^2)$且优化空间渺茫。（被认为复杂度有bound在$O(DN^{2-O(1)})$
  - Iterative Refine。
    - NN-Descent。从随机的KNNG开始，迭代的检查$v_i$邻居的邻居，并不断更新$v_i$的边。当数据集增长受到限制时，每次迭代都会使得每个节点最长边半径减半。时间复杂度在$O(N^{2-\epsilon})$
    - EFANNA。不采用随机KNNG，采用k-d tree的森林作为初始KNNG。
- Monotonic Search Networks
  KNNG并没有保证联通性。
  定义：a search path $v_1, v_2, ..., v_m$ is monotonic if $d(v_i,q) > d(v_{i+1},q)$ for all i from 1 to m-1。
  MSN图需要满足通过“best-first”得到的路径总是单调的。这意味着每一对节点都有一条单调路径，且图保证联通性。
  复杂度取决于节点的out degree
  构建图时，初始化可以随机、可以从空开始、可以用近似KNNG。如何选择source和target配对，两种方法，一种随机pair，一种固定source
  - Random Trial
  - Fixed Trial
- Small World Graphs
  定义：A graph is small-world if the length of its characteristic path grows in O(log N)。
  定义：A navigable graph is one where the length of the search path found by the best- first search algorithm scales logarithmically with N（best first的搜索路径是logN的）
  - NSW
  - HNSW 层次化。随机化分布层数
#### Discussion
- HNSW从各方面属性来说无疑比较优秀。reasonable storage，fast queries，canbe update
- 但其他index也有合适的地方。KNNG处理batch query比较优秀；KGraph比较容易构建，EFANNA更适合online query；当error guarantee需要时，LSH或RPTree是不错的选项；当内存受限时，SPANN或DiskANN比较合适

### Query Optimization and Execution
query optimizer：选择最优的查询方案，一般是latency最小的方案。
因此，步骤为 *plan enumeration*, *plan selection*, *query execution*。
#### Hybrid Operators
pre-filtering，post-filtering，single-stage filtering
pre-filtering可应用block-first scan
single-stage filtering可应用vist-first scan
post-first在那篇VBase里提到过，在那个广义单调性下可以持续吐出

- BLock-First Scan
  需要考虑block以后可能破坏联通性。
- Vist-First Scan
  对低选择性谓词效果较好，不需要在线阻塞。否则可能产生大量回溯。
  为避免回溯，一种做法是将过滤的attribute添加到best-first的算子里。
#### Plan Enumeration
因为矢量搜索的算子种类很少，所以很多时候预订计划是合理且高效的。但对于旨在支持更复杂查询的系统显然这样不行。因此对于关系系统扩展的VDBMS，可以使用关系代数来代表查询，从而允许枚举。
- Predefined
  单计划自然很好。但万一workload不适应预定的plan就G。
  Multiple Plan。不同索引有多种计划，可选择。
- Automatic
  利用底层的relational optimizer来执行美剧和选择
#### Plan Selection
目前的VDBMS主要根据 handcrafted rules或者cost model
- Rule Based
  plan数量较小，selection rules可被用来决定使用哪个。Qdrant和Vespa可见流程图。（似乎就是预定一个流程图，根据workload选择）
- Cost Based
  根据一些cost model计算预估的cost。
  有一些谓词查询估算的难题。对于pre-filtering，blocking带来较大的不确定性；而vist-first scan，扫描失败的概率很难事先知道。
#### Query Execution
硬件加速。processor caches,SIMD,GPU。分布式搜索降低单一机器负担。
对于write-heavy workloads,可采用out-of-place updates，牺牲一致性来获得吞吐量。
- 硬件加速
  - CPU Cache
  - SIMD （巧妙使用Shuffle指令）
  - GPU
- Distributed Search
  首先将vec collection切片，然后在每个分片建立本地索引。
  查询遵循scatter-gather pattern。分散到切片然后汇总。（这应该会造成损耗吧，类似于post-filtering）
- Out-Of-Place Updates
  就地更新index会导致这个索引一段时间不能使用，可能会导致严重的后果。一些应对策略
  - Replicas。类似于之前分片的方法，将向量集分为分片和副本，并建立本地索引。就地更新index、同时其他query来的时候，可以使用副本来完成查询。缺点是增大存储开销，且可能带来因为分散查询的额外开销。
  - Log-Structured-Merge Tree。stream updates into a separate structure，然后索引的时候对账。

### Current Systems
可分为native systems，特地为vec data management设计
extended system，将vec视为支持的额外数据类型
#### Native
- Mostly Vector
  针对fast search query。
  no need for a query parser,rewriter, or optimizer, 
  - EuclidesDB
    管理embedding modals
  - Vald
    非单一集中式。貌似是分布式的，有shard and replicas。每个分片采用NGT管理index。
  - Vearch
    基本目的是image-based search for e-commerce。
- Mostly Mixed
  支持更复杂的查询。
  支持更多种类的基本查询，包括对精确查询和范围查询的更多支持，以及更多查询变体。
  不需要查询重写器或解析器，但其中有部分系统优化了查询功能。
#### Extended
继承了非本地数据管理系统的所有功能，与本地系统相比更加复杂
都支持所有三种基本查询类型和多索引
主要分两类，底层NoSQL和关系型。
- NoSQL。比如schemaless storage, distributed architecture, and eventual consistency。
- Relational。功能大多来自关系型数据库固有功能。
- Search Engine, Libraries and Other Systems。这些引擎和库被嵌入到需要矢量搜索的应用程序中，但它们缺乏完整 VDBMS功能。
  - Search Engines
  - Libraries 比如知识图谱，Space Partition Tree and Graph library。SPANN和NGT等等

#### Discussion
Native mostly-vector systems一般性能较高，但针对的是特定的workloads，有时甚至是特定的查询
native mostly-mixed systems提供了更多功能，尤其是预设查询,期中一些也提供查询优化。这些系统以及扩展的 NoSQL 系统在高性能和搜索功能之间取得了良好的平衡
扩展的relational db提供更多功能，但是性能差了。

### Challenges and Open Problems
- Similarity Score Selection.前面有说，没有足够的理论支撑，很多时候选择哪个score合理似乎是玄学（语义和分数算法结合不起来）
- Operator Design.（主要对hybrid search）block-first 和 vist-first都有较大不确定性，和较大恢复成本。（但是那个广义单调性VBase这篇文章自称很大程度上解决了这点，不过性能提升多少不好说）
- Incremental Search。前面似乎没提这个
- Multi-Vector Search。较大空白，目前的q是单一向量而非向量集合。
- Security and Privacy。老生常谈。

### Question
- VDBMS怎么工作的？
  我们希望VDBMS不仅具有常规的查询优化、事务、可扩展性...，还希望它能处理非结构化数据
  因为这些非结构化数据很难用特定格式的属性来表达，因此我们需要使用相似性搜索而不能通过结构化查询来检索。为了支持similarity search，我们需要讲这些非结构化数据encode成*D-dimensional feature vectors*，然后存储到VDBMS。
- score-----discussion块
  高维向量我看好像也是算“分”？为啥说要考虑语义信息（难道这里指的是attribute filter？）
  Tagliabue, J., Greco, C.: (Vector) Space is not the final frontier: Product search as program synthesis. In: SIGIR(2023)他说这篇文章里会答


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
因为CXL’s far-memory-like characteristics会导致搜索性能显著下降，CXL-ANNS关注到node-relationship层，cache了一些访问最频繁的邻居；unchached的node预抓取了几个最可能访问的by understanding the graph traversing behaviors of ANNS。
然后这里文章说CXL-ANNS同样关注CXL网络内部的硬件协同情况，进行硬件层并行。它还用了relaxes the execution dependency of neighbor search tasks（暂不理解他在说什么）并且最大并行化的程度。
### Intruduction
- 一些motivation：
  ANNS是kNN的一个practical的实现方式，文章指出目前ANNS存在一个比较大的问题是内存的使用压力比较大，对大规模数据应用ANNS往往会造成10**TB**级的需求量。
  目前应对方法：**有损压缩**或者使用persistant的存储介质 **持久内存（PMEM）。**
- 文章的工作
  保持scalability和accuracy的基础上保证更好的performance。
  通过CXL实现DRAM的联通，联合了多台设备的内存，通过host的 root-complex映射到系统内存空间（这里的内存空间应该是指虚拟化），上层用起来和locally-attached的DRAM差不多
  上述方法解决了ANNS处理billion-point graph时的内存压力，但是会导致性能下降。造成这种现象的原因是因为CXL的特性，每次内存访问都会跑一次CXL protocal conversion。应对方式是提前cache一些访问最频繁的neighbors；unchached node，prefetch一写最可能be touched soon的节点。
  进一步提升性能，relax the execution dependency
- contribution
  Relationship-aware graph caching。核心：we observe that the graph data accesses, associated with the innermost edge hops, account for most of the point accesses 我们观察到，与最内层边缘跃点相关联的图形数据访问占了点访问的大部分。**基于数据特点找到的构建feature vector的方式**
  Hiding the latency of CXL memory pool.讲prefetch的，non-trival。提出了一个简单的预测技术。
  Collaborative kNN search design in CXL。一些硬件设计，设计EP controllers（endpoint device），更高程度利用CXL host
  Dependency relaxation and scheduling。把kNN query request分类成紧急/不紧急的子任务，然后重新调度

#### Conclusion

#### details

以往的设计

- DiskANN

  compress的在mem有什么用

- HM-ANN 层次化mem

  HNSW对应硬件存储

  （还是太慢了？）应对措施是尽量抬上去

CXL-ANN

- 计算放到内存池
- 经验发现entry附近访问次数高，提前做cache
- 读内存瓶颈，

#### Q&A
- 什么是memory disaggregation？
  分离式内存给予主机像访问本地DRAM一样，访问其他主机的空闲内存（或内存池） 的能力
  新型高带宽低延时网络硬件、协议（如 RDMA 网卡[7]）的发展，已经可以提供100Gbps带宽、纳秒级延迟的网络互联，这使得集群尺度下分离式内存池，对比SSD等承 载虚拟内存的介质，在访存延迟（<1μs | >10μs ）和带宽（>100Gbps | <5Gbps）上都有了明显优势。
- 什么是CXL（computer express link）？
  是一种高速串行协议，它允许在计算机系统内部的不同组件之间进行快速、可靠的数据传输。
- what is execution dependency for neighbor tasks
- what is contxt-aware
  似乎是一种找feature的方式，
- 压缩
- 



## High-Throughput Vector Similarity Search in Knowledge Graphs
### Abstract
探索了在知识图谱(Knowledge Graphs)背景下向量相似度搜索。他们focus on **hybrid vector similarity search**，一部分查询对应vector similarity search，一部分对应predicates over relational attributes associated with the underlying data vectors（关系上的谓词 与底层数据向量相关的属性）
什么是混合相似查询？以查歌曲为例。首先query 与给定歌曲的vecotr representaion相似的

- contribution：nabling efficient batch processing of past hybrid query workloads 对以往混合查询工作负载的批处理; 
  提出了一种workload-aware向量parittioning 方法，根据给定的workload定制向量索引布局
  多查询优化技术
- 应用效果：a 31× improvement in throughput for finding related KG queries compared to existing hybrid query processing approaches.
**没看懂上面这段**

### Instruction
背景：ADBV等新型数据管理系统通过矢量相似性搜索原语增强了查询处理，以支持查询处理需要搜索向量以查找与查询向量最相似的工作负载。
约束向量相似性搜索来给基于知识图谱的应用powering。应用有以下例子2:finding related entities, performing link prediction, and detecting erroneous facts。
motivation：当前的方法侧重于online query processing（载自知乎老哥，OLAP主要两类任务，即时响应的交互式查询和复杂耗时的批处理任务，当前方法focus on前者），缺少处理高吞吐批处理的优化。于是作者团队引入了一个系统优化，可以在工业规模的KG上实现高吞吐量的混合查询处理，且这种方式是general的，可以在vector db上适用。
他似乎在说要使用similarity search来查找过去用户查询的相关 KG 查询或实体，以及缺失事实插补的链接预测，似乎是在用link prediction的技术来补全知识图谱。

对于相关联的KG查询，他们的目标在于构建和预评估一个KG queries的集合，使得这个集合能与过去用户的查询有关联。

似乎主要意思是：根据KG对未来可能的查询做cache，用来加速

三个主要特征，上下文意思是查询KG的三个主要方法
- 混合查询：向量相似度+相关属性谓词筛选。文章里描述了一个How tall is Taylor Swift。
- 批处理：优先考虑吞吐量的查询评估方法。
- 先前工作负载的可用性：workload-aware，混合查询的谓词和用于计算相似性的向量有关。


接下来才是motivation？：1. 目前大部分vec db对关系属性谓词（主要是数字类谓词比较）支持的比较有限，多谓词也没有很多优化；2. 将向量相似度和谓词分开来单独处理，之后再合并查询结果；3. 目前基本上都在搞降低单个在线查询的延迟，加吞吐量的比较少。但现实的应用场景告诉我们，我们很多时候都会批量评估混合查询。*最后，过去的查询工作负载为利用数据和工作负载分布来设计针对特定用例的算法和数据结构提供了机会。*似乎是在说针对特定的workload设计对应的算法和数据结构可以提高性能，是一种先验的workload信息来设计db。

contribution可分两个纬度：1. workload-awareness in vector index design 2. query processing setting (batch vs. online).
具体贡献：1. Workload-aware vector index 2. Batch query optimization
- workload-aware vec index：利用过去的workload infromation来知道向量索引的底层分区，希望访问尽可能少的分区来回答query。因为hybird query有filter commnality和filter stability两个特性，基于此特点我们可以根据特定的workload来优化数据布局。
- batdh query optimization：(i) 对具有相似属性和向量相似性约束的查询进行批量处理；(ii) 针对从基于聚类的向量索引中获得的向量发布列表（posting list）执行批量向量距离计算。

### Detail
现有的做法主要就是两类，pre-filter 和 post-filter，pre-filter 首先对过滤条件进行处理得到 bitmap，然后再利用这个 bitmap 去 ANN 索引上查询，搜索过程中遇到被 mask 掉的点不放到结果中。而 post-filter 是首先进行向量查询，然后再去进行属性过滤，显然这种方法在属性过滤的过滤量较少时比较适用，否则需要 ANN 索引返回很大数量的结果才可以满足。第一种方法还有一种小优化，可以将数据对常用于过滤条件的属性进行分区，在有该分区键的过滤条件的查询就可以做到分区剪枝，但是过滤条件变了以后就失去剪枝作用了。文章采用 pre-filter，并在此基础上针对多属性做了一些分区上的改动，使得在不同属性过滤条件下仍然能够有分区剪枝能力。

### Question
- predicates over relational attributes associated with the underlying data vectors怎么理解
  关系上的谓词 与底层数据向量相关的属性
  
- So what is hybrid query?
  两部分组成。1. vector similarity search 2. evaluation of predicates over relational attributes
  知乎老哥解释是带上属性过滤的ANNS。
  
- 谓语是什么啊
  似乎是描述事物的状态
  
- 怎么优化的呢

- pre-filter的难题

  额外的向量计算，block了以后不能利用剪枝

### 偷理解

- Hybrid Query
  - Filter index --> ANN index
  - ANN --> Filter
  - multi-level index

## SPFresh: Incremental In-Place Update for Billion-Scale Vector Search
### Abstract
他们好像没把论文挂网上？似乎找不到pdf，只有摘要

对于高维向量而言，识别一个新向量的right neighbour成本较高，现有系统采用维护一个二级索引的方式来积累更新，并定期将其与主索引合并（显然这会造成准确性波动以及延迟）。于是作者提出SPFresh，就地更新向量的system。它的核心是LIRE（lightweight incremental rebalancing protocol）协议，LIRE only reassigning vectors at the boundary between partitions，而一般认为高质量的向量索引分区边界的点会比较少
什么是分界线上的点？（是HNSW的上层节点吗）


## VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity
### Abstract
需要向量查询系统和关系数据库一体化。然而高维向量缺少单调性（单调性是关系数据库索引的一个重要属性），于是现有系统依赖于*保留单调性的暂时索引（monotonicity-preserving tentative indices，为目标vec的topK neighbours）*。
VBase特点：有效支持approximate similarity search和relational operator。
发现了一个通用的性质：relational operator
### Introduction
对问题“查找 K 个与图片最相似但低于某一价格的产品”的做法是，先设置K'>K，应用TopK'的近似搜索并通过标量筛选，逐步调整K'使得适配。这显然会导致性能变差。
本文中他们提出了VBase，高效服务于复杂在线查询的新系统，支持标量和vec数据集上的相似搜索和关系运算。
**Relaxed Monotonicity**：作者团队观察到，向量索引遍历首先近似定位离目标矢量最近的区域，然后以近似方式逐步远离目标区域。于是据此定义Relaxed Monotonicity。它将是广义的单调性，因此数学上的单调性也适用于它，所以传统的数据库的索引也具有此特征。作者可通过这个共性统一vec search和传统db。作者团队也根据这一特性，提出了搜索终止条件。

contribution：1. 定义了Relaxed Monotonicity 2. 构建统一的基于松单调的数据库引擎 3. 证明引擎与TopK算法的等效性 4. 将VBase基于PostgreSQL实现，只增加了2k额外代码。

### Detail
#### Relaxed Monotonicity
Fig1不太懂了，这里所谓的索引是什么。貌似是通过算法，根据vectors算出来的某个特征？
Fig1:以FAISS和HNSW为例的vector indices，搜索会呈现两阶段的性质，先大体快速下降、接近目标邻域
于是我们希望在进入二阶段时及早终止（因为继续查也没东西了）
<img src="img/WX20231024-204606@2x.png">
Fig2显示，判断进入二阶段标识，query 需要了解$R_q$（邻域球的半径）$M_q^s$（当前索引遍历位置与q的距离）是否有$M_q^s > R_q$。然后$R_q$有如下定义
$R_q = Max(TopE(\{ Distance(q,v_j) | j\in [1, s-1] \}))$
$M_q^s = Median(\{Distance(q,v_i) | i \in [s- w+1, s] \}) $，w是最近w个步骤，遍历窗口

**Definition 1 Relaxed Monotonicity** $\exist s, M_q^t \ge R_q,\forall t \ge s $

应用Relaxed Monotonicity你需要确定一些东西
E的值（算$R_q$需要TopE）
w窗口大小（算$M_q^s$需要）

### Unified Query Execution Engine
为什么说VBase可以会比only K的问题
那个图第二阶段，VBase因为有了单调性支撑所以可以不停吐出信息？
对应only TopK的算法应该只会保留最少的，在最后输出的list里应该会不停将较远的vec给t了。


优点：索引遍历过程中执行过滤（没太懂，为啥topk不能这么做），且具有灵活的终止条件；支持join

### Conclusion

### Question
- introduction里说，根据临时索引的单调性，他能够实现复杂的关系运算，但对问题“查找 X 个与图片最相似但低于某一价格的产品”，找到topK近似后筛选低于价格时很可能结果远小于K，那为什么不全筛了再查找？反正是个标量，或许可以像那个苹果论文里面的，他说某一领域的query所附带的筛选条件具有规律性，那我按照特定的地方用不同的标量设定一些table有什么弊端吗
- 有点不太理解为什么说，因为VBase的查询引擎搜索结果与topK算法相当，于是VBase就规避了仅topK的接口。它是如何回避的；only-topK有什么不足的吗
- 那个tentasive indices是什么，难到不是上层的向量吗
  google的结果是：通过某个算法来indexing vecotors，such as PQ, LSH, or HNSW
- 他们是如何想到这个松弛单调性
  似乎一切的出发点在于提前终止阶段二。
- SELECT recipe_id FROM Recipe ORDER BY INNER_PRODUCT( images_embedding ,\${p_images_embedding}) LIMIT 50;
  总的意思是single-vector topk，但是\${p_images_embedding}就不知道什么意思了




## Fast, Approximate Vector Queries on Very Large Unstructured Datasets
### Abstract
处理高维向量查询是一个比较困难的任务，尤其对于严格服务级别目标（SLO，strict serveice level objectives）。
本文介绍Anucel，可以提供有界的查询延迟和查询谬误（bounded query errors and bounded query latencies，我觉得应该是Anucel查询的错误和延迟有一个理论上限）。
核心理念：利用单个向量局部几何特性构建error-latency profile for each query。

### Introduction
motivation: ANN的一个基础想法是，采样整个dataset的一个子集，在里面去寻找top-k；于是采样的大小就影响了query的准确性和延迟。对与应用ANNS的SLOs，performance guarantees是比较需要的，但是目前的系统并没有在这方面有所实现。以Faiss和AnalyticDB-V为例，他们没有提供任何的performance bound，他们提供了query error对sampling size的profile。给定数据集下，他们忽略了查询向量的特征，对所有查询都使用了固定的采样数量，这样需要采样数量基本上要拉满，使用所有查询向量中的最大采样大小才能满足误差约束，因此会造成很多的冗余计算。
Learned Adaptive Early Termination（LAET），缺点在于将ANN当作黑箱，在要求误差有界的情形下性能糟糕。
他们提出了Auncel，分布式向量搜索引擎，带有performance guarantee。核心是query-aware error-aware error-latency profile。
**2 challenges**：
1. 给定误差或延迟上限后确定合适的采样数量。解决方式：使用了一个whitebox approach来挖掘高维空间几何特性，然后利用更精确的ELP可以提前terminate
2. 如何扩展到多个worker来降低查询延时。解决方式：数据集分片给多个worker，分别处理后汇总；需要注意为每个worker设置正确local error bound。Anucel使用概率论校准local error bound。

### Detail


### Question
- worker是什么东西




# 知识部分
k-NN: K nearest neighbor
### ANN
- Annoy（Approximate Nearest Neighbors Oh Yeah）；参考https://zhuanlan.zhihu.com/p/454511736，二叉树分割
- ScaNN（Scalable Nearest Neighbors）；
- Faiss（Billion-scale similarity search with GPUs）；
- Hnswlib（fast approximate nearest neighbor search）

 Quantization：mapping continuous infinite values to a smaller set of discrete finite values(mathwork这么解释)
Vector quantization: 常用压缩技术


