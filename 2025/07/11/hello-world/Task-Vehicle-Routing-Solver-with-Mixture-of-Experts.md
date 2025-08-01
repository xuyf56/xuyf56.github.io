---
title: 'MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts'
date: 2025-07-13 01:28:23
tags: ICML 2024
mathjax: true
---
<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Title.png" alt="VRP变体" width="1000">
</figure>
</div>

## 总结


## 摘要
- 在**某个特定问题**上独立地建模和训练的神经网络求解器缺乏足够的泛化性和实用性。
- 本文提出使用**混合专家系统**的多任务车辆路由问题求解器（Multi-task Vehicle Routing Solver with Mixture-of-Experts, MVMoE）。旨在不增加计算量的同时提升模型性能。
- 在MVMoE中进一步加入**分层门控机制**，控制模型性能和计算复杂度之间的权衡。
- 实验上，1）模型在10个训练中未见过的VRP变体中展现出zero-shot泛化能力；2）在real-world benchmark实例的few-shot设置中也得到合适的结果。3）门控机制在面对分布外数据时也展现出一定的优越性。
- Github：[https://github.com/RoyalSkye/Routing-MVMoE.](https://github.com/RoyalSkye/Routing-MVMoE)

## 研究问题

#### 不同约束的VRP变体
<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Various_Constraints.png" alt="VRP变体" width="500">
<figcaption>图1. 不同约束的子路径示例。</figcaption>
</figure>
</div>

1. **开放路径约束 (Open Route - O)**：车辆服务完客户点后**无需返回仓库**（$v_0$）。

2. **回程约束 (Backhaul - B)**：允许linehauls与backhauls**无优先级混合访问**。<br>

  - **正需求 ($\delta_i>0$)**：linehauls（卸货）。<br>
  - **负需求 ($\delta_i<0$)**：backhauls（装货）。

3. **时长限制 (Duration Limit - L)**：单条路径总长度（成本）**不得超过预设阈值**。

4. **时间窗约束 (Time Window - TW)**：节点$v_i$必须在$[e_i, l_i]$内开始服务。

- 早到需等待至$e_i$。  
- 所有车辆必须在$l_0$前返回仓库。

5. **容量限制(Capacity Constraint - C)**：即CVRP。

> **开放路径耦合效应**：当与(O)组合时，**免除仓库返回时间约束**。  
  **约束组合特性**：多约束组合存在非线性交互（如O+TW），非简单叠加。5种基础约束可组合成16种VRP变体。


## 方法
<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Model_Structrue.png" alt="VRP变体" width="1000">
<figcaption>图2. MVMoe模型架构。具体来说，在编码器中用 MoE 替换了 FFN 层，<br>并在解码器的多头注意力的最终线性层中用 MoE 进行替换。</figcaption>
</figure>
</div>

#### 编码器 (Encoder)
- **输入**：每个节点 $v_i$ 的**静态特征** $\mathcal{S}_{i} = \{ y_i, \delta_i, e_i, l_i \}$  
  - $y_i$：坐标 
  - $\delta_i$：需求
  - $e_i$：时间窗开始时间  
  - $l_i$：时间窗结束时间  
  
- **输出**：$d$ 维**节点嵌入向量** $h_i$

#### 解码器 (Decoder) - 第 $t$ 步
- **输入**：  
  1. **节点嵌入**：编码器输出的所有 $h_i$  
  2. **上下文表示**：  
     - 上一个被选节点的嵌入 $h_{\text{last}}$  
     - **动态特征** $\mathcal{D}_{t} = \{ c_t, t_t, l_t, o_t \}$  
        - $c_t$：车辆剩余容量  
        - $t_t$：当前时间  
        - $l_t$：当前部分路径长度  
        - $o_t$：开放路径指示器（是否需要返回仓库）
- **输出**：  
  - **节点概率分布**：所有有效节点的选择概率向量  
  - **动作**：根据概率分布选择下一个节点，添加到当前部分解中
  
#### 不同变体随机训练
- 每个训练批次随机选择一种VRP变体
- 当前VRP变体未使用的特征**置零填充**
  - **静态特征示例**（CVRP）：$\mathcal{S}_{i}^{(C)} = \{y_i, \delta_i, 0, 0\}$  → 时间窗特征$(e_i,l_i)$填零
  - **动态特征示例**（CVRP）：$\mathcal{D}_{t}^{(C)} = \{c_t, 0, l_t, 0\}$  → 当前时间$t_t$和开放路径$o_t$填零  
- **损失函数**：$\min\limits _ {\Theta} \mathcal{L}=\mathcal{L}_{a}+\alpha\mathcal{L}_{b}$
  - $\mathcal{L}_{a}$：原始损失函数（例如REINFORCE损失）。
  - $\mathcal{L}_{b}$：与MoEs相关的损失函数（例如用于确保负载均衡的辅助损失）
  
#### 门控机制
门控网络增加了额外的计算开销，主要是1）门控网络的前向传播；2）节点到专家的分配。  
编码器层的门控次数是固定的，取决于编码器层数。解码器上门控次数由层数和解码次数决定。本文在解码器层提出分层门控机制来权衡模型性能和计算开销。

<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/node level gating.png" width="500">
<figcaption>图3. 得分矩阵和门控算法的示例。彩色点代表被选择的节点或专家。<br>
左边：Input-choice gating. 右边：Expert-choice gating.</figcaption>
</figure>
</div>

- 节点级门控：门控网络参数$W_G\in \mathbf{R}^{d\times m}$，输入批次$X\in \mathbf{R}^{I\times d}$，得分矩阵$H=(X\cdot W_G)\in \mathbf{R}^{I\times m}$。
  - Input-choice gating：每个节点基于$H$选取$TopK$个专家。不保证负载均衡，可能导致某些专家欠拟合。通常会引入辅助损失(Auxiliary Loss)。
  - Expert-choice gating：每个专家基于$H$选取$TopK$个节点，$K=\frac{I\times \beta}{m}$。能够确保负载均衡，但可能有节点不被任一专家选中。
  
<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Hierarchical Gating.png" width="500">
<figcaption>图4. 分层门控（右边）。</figcaption>
</figure>
</div>

- 分层门控机制：在每一解码步骤上应用MoEs十分消耗计算资源，1）解码步骤$T$随着问题规模$n$的增加而上升；2）解码过程中问题特有的可行性约束必须被满足。因此，本文仅在部分解码步骤上应用MoEs。
  - 分层门控（右边）通过门控网络$G_1$将输入引导进稀疏层或稠密层。稀疏层中门控网络$G_2$再引导节点与专家选取。$G_1$根据问题级表征$X_1$进行选择，$G_2$根据节点级表征进行选择。

## 实验
考虑5中约束组合的16中VRP变体。设备：NVIDIA A100-80G，AMD EPYC 7513 CPU @ 2.6GHz.

####
- 基线：
  - 传统求解器：节点数$n=50/100$，搜索时间限制为$20s/40s$。
    - HGS求解CVRP和VRPTW
    - LKH3求解CVRP、OVRP、VRPL、VRPTW
    - OR-Tools求解全部16种变体。
      - 采用平行最廉价插入法(Parallel Cheapest Insertion)求初始解
      - 使用引导式局部搜索(Guided Local Search)做局部搜索。

  - 神经求解器
    - POMO
    - POMO-MTL (1.25M)
- 训练：
  - 对于神经求解器：
    - 采用Adam优化器，学习率$1e^{-4}$，权重衰减$1e^{-6}$。最后10\%数据，学习率衰减10倍。
    - batch size=128，epochs=5000 with 20000 trainning instances。
  - 对于多任务求解器：
    - 训练集数据：CVRP、OVRP、VRPB、VRPL、VRPTW、OVRPTW.
  - 对于MVMoE：
    - 专家数$m=4, K=\beta=2$。
    - 辅助损失权重$\alpha=0.01$。
    - 采用node-level input-choice gating。

#### 实验结果
<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Result1.png" alt="VRP变体" width="1000">
<figcaption>表1. MVMoe模型性能对比。</figcaption>
</figure>
</div>

- POMO在每个单独问题上的表现优于多任务求解器，但平均性能（泛化能力）差。
- MVMoE优于POMO-MTL，表明MVMoE在多任务VRP求解方面具有优势。

<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Zero-shot generalization.png" alt="VRP变体" width="1000">
<figcaption>表2. Zero-shot 泛化表现。</figcaption>
</figure>
</div>

- 在10个未见过的VRP变体上的表现，说明MVMoE的Zero-shot 泛化能力优于POMO-MTL。

<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Few-shot generalization.png" alt="VRP变体" width="500">
<figcaption>图5. Few-shot 泛化表现对比。</figcaption>
</figure>
</div>

#### 消融实验
探索不同MoE设置对Zero-shot泛化能力的影响。epochs减少至2500，问题规模$n=50$，使用MVMoE/4E。
<div align=center>
<figure>
<img src="./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Ablation experiment.png" alt="VRP变体" width="1000">
<figcaption>图6. 消融实验结果。</figcaption>
</figure>
</div>

- MoEs位置：Encoder和Decoder处有利于zero-shot泛化。
  - 原始特征处理层
  - 编码器层
  - 解码器层
- 专家数量：先统一使用50M实例进行训练，再使8E额外训练50M实例（总100M），16E额外训练150M实例（总200M）。
  - 专家数量8
  - 专家数量16
- 门控机制：
  - 层次
    - 节点级
    - 实例级
    - 问题级
  - 算法
    - input-choice
    - expert-choice
    - random gating

## 结论
为建立一个解决VRP问题的更通用和强大的神经网络求解器，本文提出了MVMoE(Multi-task Vehicle Routing Solver with MoEs)以及适用于MVMoE的分层门控机制。第一次尝试建立一个大型的VRP模型。MVMoE也展现出了zero-shot、few-shot的强大泛化能力。但相比于大语言模型的参数规模，MVMoE仍然还是小得多。  
未来的研究方向：
1. 解决大规模VRPs问题的可扩展（scalable）基于MoE模型的发展
2. 针对不同问题的通用表征的探索
3. 门控机制可解释性的探索
4. 对MoEs的scaling laws的研究

## 笔记
* 跨规模泛化  

> Fu, Z.-H., Qiu, K.-B., and Zha, H. Generalize a small pretrained model to arbitrarily large tsp instances. In AAAI, volume 35, pp. 7474–7482, 2021.  
Hou, Q., Yang, J., Su, Y., Wang, X., and Deng, Y. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. In ICLR, 2023.<br>
Son, J., Kim, M., Kim, H., and Park, J. Meta-SAGE: Scale meta-learning scheduled adaptation with guided exploration for mitigating scale shift on combinatorial optimization. In ICML, 2023.<br>
Luo, F., Lin, X., Liu, F., Zhang, Q., and Wang, Z. Neural combinatorial optimization with heavy decoder: Toward large scale generalization. In NeurIPS, 2023.<br>
Drakulic, D., Michel, S., Mai, F., Sors, A., and Andreoli, J.M. BQ-NCO: Bisimulation quotienting for generalizable neural combinatorial optimization. In NeurIPS, 2023.<br>

* 跨分布泛化

> Zhang, Z., Zhang, Z., Wang, X., and Zhu, W. Learning to solve travelling salesman problem with hardness-adaptive curriculum. In AAAI, 2022.<br>
Geisler, S., Sommer, J., Schuchardt, J., Bojchevski, A., and Günnemann, S. Generalization of neural combinatorial solvers through the lens of adversarial robustness. In ICLR, 2022.<br>
Bi, J., Ma, Y., Wang, J., Cao, Z., Chen, J., Sun, Y., and Chee, Y. M. Learning generalizable models for vehicle routing problems via knowledge distillation. In NeurIPS, 2022.<br>
Jiang, Y., Cao, Z., Wu, Y., Song, W., and Zhang, J. Ensemble-based deep reinforcement learning for vehicle routing problems under distribution shift. In NeurIPS, 2023.

* 跨VRP变体泛化

> Wang, C. and Yu, T. Efficient training of multi-task neural solver with multi-armed bandits. arXiv preprint arXiv:2305.06361, 2023.  
Liu, F., Lin, X., Zhang, Q., Tong, X., and Yuan, M. Multi-task learning for routing problem with cross-problem zero-shot generalization. arXiv preprint arXiv:2402.16891, 2024.  
Lin, Z., Wu, Y., Zhou, B., Cao, Z., Song, W., Zhang, Y., and Senthilnath, J. Cross-problem learning for solving vehicle routing problems. In IJCAI, 2024.

### CVRP（容量约束车辆路径问题）  
- **图结构**  
  设图 $ g = [V, E] $，其中  
  - $V = \{v_0, v_1, \dots, v_n\}$：节点集合  
    - $ v_0 $：仓库（depot）  
    - $\{v_i\}_{i=1}^n$：顾客节点（共 $ n $ 个）  
  - $ E $：边集合，包含任意两节点 $v_i, v_j $（ $ i ≠ j $ ）之间的边 $ e(v_i, v_j)$。  

- **容量约束**  
  每个顾客节点 $ v_i $ 有需求 $ d_i ≥ 0 $，每辆车的最大容量为 $ Q $。  

- **路径结构**  
  解（即路径方案）$ T $ 由多个子路径（sub-tours）组成：  
  - 每个子路径表示一辆车从仓库 $ v_0 $ 出发，访问若干顾客节点后返回 $ v_0 $。  
  - 需满足：  
    1. **唯一性**：每个顾客节点被恰好访问一次。  
    2. **容量限制**：每个子路径中所有顾客节点的总需求 $ \sum d_i ≤ Q $。  

- **成本函数**  
  在欧几里得空间中，路径成本 $ c(T) $ 定义为所有子路径的总长度（即边的欧氏距离之和）。 

### MoE（混合专家层）
- 包含$m$个专家$\{E_1, E_2, \ldots, E_m\}$，每个专家是**线性层或FFN**，相互之间参数不共享。
- 门控网络$G$决定输入如何分配给专家。
> \[\text{MoE}(x) = \sum_{j=1}^{m} G(x)_j \cdot E_j(x)\\
  G(x)_j：第 j 个专家的权重  \\
  E_j(x)：第 j 个专家的输出\]


## 参考文献(后续阅读)
1. Wang C, Yu T. Efficient training of multi-task neural solver with multi-armed bandits. CoRR. 2023 Jan 1.
2. Lin Z, Wu Y, Zhou B, Cao Z, Song W, Zhang Y, Jayavelu S. Cross-problem learning for solving vehicle routing problems. arXiv preprint arXiv:2404.11677. 2024 Apr 17.
3. **Liu F, Lin X, Wang Z, Zhang Q, Xialiang T, Yuan M. Multi-task learning for routing problem with cross-problem zero-shot generalization. InProceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2024 Aug 25 (pp. 1898-1908).**
4. Kaplan J, McCandlish S, Henighan T, Brown TB, Chess B, Child R, Gray S, Radford A, Wu J, Amodei D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. 2020 Jan 23.
5. Floridi, L. and Chiriatti, M. Gpt-3: Its nature, scope, limits, and consequences. Minds and Machines, 30:681–694, 2020.
6. **Joshi, C. K., Cappart, Q., Rousseau, L.-M., and Laurent, T. Learning tsp requires rethinking generalization. In International Conference on Principles and Practice of Constraint Programming, 2021.**
7. **Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In ICLR, 2017.**
8. Puigcerver, J., Riquelme, C., Mustafa, B., and Houlsby, N. From sparse to soft mixtures of experts. In ICLR, 2024.
9. Xue, F., Zheng, Z., Fu, Y., Ni, J., Zheng, Z., Zhou, W., and You, Y. OpenMoE: An early effort on open mixture-of-experts language models. arXiv preprint arXiv:2402.01739, 2024.
10. Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232–5270, 2022b.
11. Yuksel, S. E., Wilson, J. N., and Gader, P. D. Twenty years of mixture of experts. IEEE transactions on neural networks and learning systems, 23(8):1177–1193, 2012.
12. Fedus, W., Dean, J., and Zoph, B. A review of sparse expert models in deep learning. arXiv preprint arXiv:2209.01667, 2022a.
13. **Luo, F., Lin, X., Liu, F., Zhang, Q., and Wang, Z. Neural combinatorial optimization with heavy decoder: Toward large scale generalization. In NeurIPS, 2023.**
