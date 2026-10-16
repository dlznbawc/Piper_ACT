# pick_place：抓取放置

手腕相机看到物体，机械臂抓起物体放到目标位置。

纯端到端：手腕 DaBai 图像 + 7 维状态（6 关节角 + 夹爪开度 m）→ ACT → 7 维目标，25Hz。

| 文件 | 作用 |
|------|------|
| `task.py` | `TaskSpec`：任务描述、带夹爪、夹持力 1 N、终点停留 3 s、停稳判定 2 s |
| `record.py` | 拖拽示教采集 → `datasets/pick_place/`（流程在 `piper_act/record.py`） |
| `record_phone.py` | iPhone 遥操作采集 → `datasets/pick_place/`（流程在 `piper_act/phone_record.py`） |
| `deploy.py` | 真机运行训练好的策略（流程在 `piper_act/deploy.py`） |
| `start_pose.json` | 起始位姿（第一次运行 record.py 时拖拽设定，或第一次运行 record_phone.py 时用默认值；采集和部署共用） |

以下命令都在仓库根目录运行，环境见[根目录 README](../../README.md)。

## 1. 采集

```bash
python3 tasks/pick_place/record.py
```

每条示范：物体放到要抓的位置 → 托住机械臂、回车进入拖拽 → 拖着完成抓取和放置，**手动开合夹爪**，放完把机械臂抬离物体、停稳 → 回车结束 → 回车：机械臂沿示教轨迹倒着回起点（夹爪也倒放，**顺手把物体从放置处搬回抓取处**，不用手动复位），再正向回放并录制 → s 保存 / r 重来 / q 退出。

采集要点：
- **任务描述**：开录前把 `task.py` 里的 `task` 改成实际物体，如 `"pick up the red cube and place it on the blue square"`。
- **夹持力**：回放和部署都用 `task.py` 里固定的 `grip_force`（默认 1 N）。回放时物体滑落就调大，同一数据集录制中途别改。
- **覆盖**：每条之前（物体已被搬回原处）可以把物体挪到新位置，覆盖部署时会出现的范围。先录 50 条左右。
- **示教节奏**：抓之前、放之前的停顿别太长（> 2 s 会被部署时当成结束）。

### iPhone 遥操作采集（另一种采集方式，二选一）

```bash
python3 tasks/pick_place/record_phone.py
```

手机准备、标定、按键和仿真 [sim_pick_place](../sim_pick_place/README.md) 相同：启动时手机屏幕朝上平放、点一下 B1 标定。每次点「跟随 开/关」开始跟随的那一刻，手机顶边指向哪里，哪里就是机械臂的前方，所以先站到机械臂后方、顶边对准机械臂伸出去的方向再点。之后每条示范：机械臂自动慢速回起始位姿 → 摆好物体 → 点「跟随 开/关」开始（同时开始录制）→ 移动手机控制指尖、A3 滑块开合夹爪 → 放好、抬起后点「保存」（再原地录 3 秒）/「丢弃重来」/「退出」。

- 夹爪始终竖直朝下。起始位姿夹爪必须朝下：`start_pose.json` 是拖拽设定的歪姿态时会提示删掉，改用默认起始位姿。
- 安全：每帧关节最多变化 `--max-step`（默认 0.03 rad ≈ 43°/s），指尖不低于 `--z-min`（默认 1 cm，相对底座底面），目标够不到时停在原地。Ctrl+C 停在当前位置。首次运行手放在急停旁。
- 灵敏度 `--scale`（默认 0.5：手机移 10 cm，指尖移 5 cm）。

## 2. 训练

```bash
scripts/train.sh datasets/pick_place
```

## 3. 真机运行

```bash
python3 tasks/pick_place/deploy.py --policy outputs/train/act_pick_place/checkpoints/last/pretrained_model
```

每轮：物体放好 → 回车开跑 → 动作停稳（2 秒内关节指令变化 < 0.005 rad、夹爪 < 2 mm）或 30 秒超时即停 → 回车沿原路退回起始位姿（夹爪也倒放，会把物体搬回去）。Ctrl+C 随时停在原地。参数同 [charge_align](../charge_align/README.md#3-真机运行)；夹爪开度不限速，`--max-step` 只限关节。
