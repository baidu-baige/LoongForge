# X-VLA 开源权重 LIBERO 复现问题分析与解决报告

- 日期：2026-07-17
- 背景：将 X-VLA 接入 LoongForge 评测系统，加载官方开源权重（`/ssd1/sunyuehang/xvla-libero`）
  在 LIBERO（libero_object）上复现。评测链路已连通，但初次评测成功率为 0
  （280 步内无任务完成）。
- 排查方法：以 X-VLA 官方评测客户端（`evaluation/libero/libero_client.py`，
  github.com/2toinf/X-VLA）为基准，逐项对比本体感知输入构造、控制器配置、
  动作执行方式与评测参数，定位差异。

## 一、失败现象

对失败评测的 trace（`reports/xvla/libero/libero_weight_object_300step/`）分析显示：

- 模型预测的绝对目标位姿与末端实际位姿之间始终保持近乎恒定的偏移
  （z 方向约 +0.22 m），机械臂始终无法到达目标；
- 夹爪 sigmoid 输出全程处于 0.007 至 0.045 区间，从未超过二值化阈值 0.5，
  即整个 episode 内从未发出闭合指令。

上述现象表明模型输入（本体感知）与动作执行方式（控制模式）和训练协议不一致。

## 二、与官方脚本对比发现的问题

按影响程度排序，共定位四项差异。

### 问题 1：控制模式不一致（影响最大）

官方脚本在环境初始化后将控制器切换为绝对位姿模式，并将模型输出的绝对位置与
axis-angle 姿态直接下发：

```python
for robot in env.env.robots:
    robot.controller.use_delta = False
```

本系统实现保留了增量（delta）控制器，并自行做"绝对位姿减当前位姿"的转换。
该转换存在两处错误：

1. 旋转分量 `rotvec_target - rotvec_current` 在数学上不构成相对旋转
   （rotation vector 不满足线性加减），姿态控制方向错误；
2. robosuite 的增量 OSC 控制器将输入视为 [-1, 1] 归一化量并缩放至
   `output_max`（位置约 0.05 m/步），以米为单位的原始差值直接输入导致尺度完全失配。

该问题解释了目标位姿恒定偏移的现象。

### 问题 2：本体感知输入中 rot6d 排布不一致

官方实现的 6D 旋转定义为旋转矩阵第一列与第二列的拼接（列主序）：

```python
def Mat_to_Rotate6D(self, R):
    return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)   # [R00,R10,R20, R01,R11,R21]
```

本系统 `adapters/libero.py` 中使用 `ee_ori_mat[:, :2].flatten()`，
得到的是行主序交错排布 `[R00,R01,R10,R11,R20,R21]`，模型接收到的姿态输入分布错误。
`quat_to_rot6d` 存在同样问题。动作解码侧（`predict_action_interface.py`）的排布正确，
仅输入侧存在该缺陷。

### 问题 3：闭环本体感知语义不一致

官方脚本在每个动作 chunk 消费完毕后，将该 chunk 最后一帧的预测动作回填作为
下一次推理的本体感知输入，仅在 episode 起始使用真实测量位姿：

```python
self.proprio[:9] = action[-1, :9].copy()
```

本系统每步均使用控制器实测状态。该差异单独不构成致命问题，列为观察项。

### 问题 4：评测参数低于官方协议

- horizon：官方 800，本系统配置 300 且被套件上限进一步截断至 280；
- 环境稳定等待步数：官方 10，本系统配置 1。

该项不解释成功率为 0，但会压低复现成功率。

## 三、解决方案

1. 问题 1：`orchestrator/runners/libero_runner.py` 在环境初始化完成后设置
   `robot.controller.use_delta = False`（依据 `action_postprocess` 是否配置判断），
   删除绝对转增量的减法逻辑，直接下发绝对动作。
2. 问题 2：`adapters/libero.py` 中 `_build_model_state` 与 `quat_to_rot6d`
   改为列主序排布（`[R[:,0], R[:,1]]`）。
3. 问题 4：评测配置调整为 `max_steps: 800`、`num_steps_wait: 10`，
   并将 `max_steps` 语义改为显式配置值优先于套件默认上限；同步调大
   `per_episode_sec` 与 `start_timeout_sec`（模型冷启动超过 900 秒）。
4. 问题 3：未做修改。修复问题 1、2、4 后复现即成功，说明使用环境实测状态
   构造本体感知输入在 LIBERO 上可行，该项保留为后续消融验证内容。

## 四、验证结果

### Smoke 验证（libero_object task0 episode0，seed 7）

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| success | 0（280 步超时） | 1（144 步完成） |
| episode 用时 | 56 s（跑满上限） | 32.6 s |
| 平均推理延迟 | 297 ms | 196 ms |

成功回放见 `reports/xvla/libero/libero_weight_object_fix800/artifacts/libero_object/task0/episode0/replay_success.gif`。

### 全量验证（libero_object 10 任务 × 10 episode）

总成功率 94/100 = 94.0%，平均完成步数 176，平均推理延迟 262 ms，总耗时约 64 分钟。
结果文件：`reports/xvla/libero/libero_weight_object_full/results.jsonl`。

| 任务 | 成功率 |
| --- | --- |
| task 0 / 1 / 4 / 5 / 6 / 9 | 10/10 |
| task 2 / 7 | 9/10 |
| task 3 / 8 | 8/10 |

该结果与 X-VLA 论文报告的 LIBERO 水平相当，确认开源权重复现成功。

## 五、结论

开源权重本身无问题，复现失败源于评测系统与官方部署协议的四项差异：
控制模式（绝对位姿被误转为增量）、本体感知 rot6d 排布（行主序误用）、
闭环本体感知语义（观察项，未修改）、评测参数（horizon 与稳定等待步数偏小）。
修复前两项并对齐评测参数后，成功率由 0 提升至 94.0%。

相关通用经验已整理至 `loongforge/embodied/eval/model_integration_guide.md`，
供后续新模型接入参考。
