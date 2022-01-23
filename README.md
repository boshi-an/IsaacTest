# Isaac Test

环境部分的搭建已经完成，实现了 `compute_observations()`, `compute_reward()`, `pre_physics_step()`, `post_physics_step()` 四个接口

todo:

手写一个 Runner 用于环境和算法的交互，具体需要实现的东西需要参考 `torch_runner`

确定好算法的接口，参考 `torch_runner`

流程大约是：

Runner 通过 cfg 的参数构建环境和模型，然后轮流调用环境和模型的接口进行训练。

## 算法

接口：

 - get_actions(batch)
 - train()

## Runner

没有给算法的接口

没有给环境的接口

不停调用算法和环境

## 环境

接口：


