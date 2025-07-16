---
title: 'MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts'
date: 2025-07-13 01:28:23
tags:
mathjax: true
---
## Summary
写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。

## 摘要
- 在**某个特定问题**上独立地建模和训练的神经网络求解器缺乏足够的泛化性和实用性。
- 本文提出使用**混合专家系统**的多任务车辆路由问题求解器（Multi-task Vehicle Routing Solver with Mixture-of-Experts, MVMoE）。旨在不增加计算量的同时提升模型性能。
- 在MVMoE中进一步加入**分层门控机制**，控制模型性能和计算复杂度之间的权衡。
- 实验上，1）模型在10个训练中未见过的VRP变体中展现出zero-shot泛化能力；2）在real-world benchmark实例的few-shot设置中也得到合适的结果。3）门控机制在面对分布外数据时也展现出一定的优越性。
- Github：https://github.com/RoyalSkye/Routing-MVMoE.

## 问题描述  
### 不同约束的VRP变体
![VRP变体](./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/Various_Constraints.png)
1. **开放路径约束 (Open Route - O)**：车辆服务完客户点后**无需返回仓库**（\(v_0\)）。

2. **回程约束 (Backhaul - B)**：允许linehauls与backhauls**无优先级混合访问**。<br>
> **正需求 (\(\delta_i>0\))**：linehauls（卸货）。<br>
  **负需求 (\(\delta_i<0\))**：backhauls（装货）。

3. **时长限制 (Duration Limit - L)**：单条路径总长度（成本）**不得超过预设阈值**。

4. **时间窗约束 (Time Window - TW)**：节点\(v_i\)必须在\([e_i, l_i]\)内开始服务。
> 早到需等待至\(e_i\)。  
  所有车辆必须在\(l_0\)前返回仓库。
- **开放路径耦合效应**：当与(O)组合时，**免除仓库返回时间约束**。

> **约束组合特性**：多约束组合存在非线性交互（如O+TW），非简单叠加。5种基础约束可组合成16种VRP变体（详见表3）。


## 方法
![模型结构](./MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts/MVMoE-Multi-Task-Vehicle-Routing-Solver-with-Mixture-of-Experts.png "MVMoE模型架构图")

## Evaluation
作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？

## Conclusion
作者给出了哪些结论？哪些是strong conclusions, 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在discussion中提到；或实验的数据并没有给出充分的evidence）?

## 笔记
* 跨规模泛化  
> Fu, Z.-H., Qiu, K.-B., and Zha, H. Generalize a small pretrained model to arbitrarily large tsp instances. In AAAI, volume 35, pp. 7474–7482, 2021.  
Hou, Q., Yang, J., Su, Y., Wang, X., and Deng, Y. Generalize learned heuristics to solve large-scale vehicle routing problems in real-time. In ICLR, 2023.<br>
Son, J., Kim, M., Kim, H., and Park, J. Meta-SAGE: Scale meta-learning scheduled adaptation with guided exploration for mitigating scale shift on combinatorial optimization. In ICML, 2023.<br>
Luo, F., Lin, X., Liu, F., Zhang, Q., and Wang, Z. Neural combinatorial optimization with heavy decoder: Toward large scale generalization. In NeurIPS, 2023.<br>
Drakulic, D., Michel, S., Mai, F., Sors, A., and Andreoli, J.M. BQ-NCO: Bisimulation quotienting for generalizable neural combinatorial optimization. In NeurIPS, 2023.<br>
* 跨分布泛化
>Zhang, Z., Zhang, Z., Wang, X., and Zhu, W. Learning to solve travelling salesman problem with hardness-adaptive curriculum. In AAAI, 2022.<br>
Geisler, S., Sommer, J., Schuchardt, J., Bojchevski, A., and Günnemann, S. Generalization of neural combinatorial solvers through the lens of adversarial robustness. In ICLR, 2022.<br>
Bi, J., Ma, Y., Wang, J., Cao, Z., Chen, J., Sun, Y., and Chee, Y. M. Learning generalizable models for vehicle routing problems via knowledge distillation. In NeurIPS, 2022.<br>
Jiang, Y., Cao, Z., Wu, Y., Song, W., and Zhang, J. Ensemble-based deep reinforcement learning for vehicle routing problems under distribution shift. In NeurIPS, 2023.
* 跨VRP变体泛化
>Wang, C. and Yu, T. Efficient training of multi-task neural solver with multi-armed bandits. arXiv preprint arXiv:2305.06361, 2023.  
Liu, F., Lin, X., Zhang, Q., Tong, X., and Yuan, M. Multi-task learning for routing problem with cross-problem zero-shot generalization. arXiv preprint arXiv:2402.16891, 2024.  
Lin, Z., Wu, Y., Zhou, B., Cao, Z., Song, W., Zhang, Y., and Senthilnath, J. Cross-problem learning for solving vehicle routing problems. In IJCAI, 2024.

### CVRP（容量约束车辆路径问题）  
- **图结构**  
  设图 $ g = [V, E] $，其中  
  - $ V = \{v_0, v_1, \dots, v_n\} $：节点集合  
    - $ v_0 $：仓库（depot）  
    - $ \{v_i\}_{i=1}^n $：顾客节点（共 $ n $ 个）  
  - $ E $：边集合，包含任意两节点 $ v_i, v_j $（$ i \neq j $）之间的边 $ e(v_i, v_j) $。  

- **容量约束**  
  每个顾客节点 $ v_i $ 有需求 $ d_i \geq 0 $，每辆车的最大容量为 $ Q $。  

- **路径结构**  
  解（即路径方案）$ T $ 由多个子路径（sub-tours）组成：  
  - 每个子路径表示一辆车从仓库 $ v_0 $ 出发，访问若干顾客节点后返回 $ v_0 $。  
  - 需满足：  
    1. **唯一性**：每个顾客节点被恰好访问一次。  
    2. **容量限制**：每个子路径中所有顾客节点的总需求 $ \sum d_i \leq Q $。  

- **成本函数**  
  在欧几里得空间中，路径成本 $ c(T) $ 定义为所有子路径的总长度（即边的欧氏距离之和）。 



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
