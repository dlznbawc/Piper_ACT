# Piper_ACT

PiPER X + ACT（Action Chunking Transformer）的真机任务集合。共享库负责硬件，每个任务一个文件夹。

```
Piper_ACT/
├── pyproject.toml              pip install -e . 装上 piper_act
├── src/piper_act/              共享库
│   ├── arm.py                  PiPER X 真机封装（含夹爪）
│   ├── camera.py               找到 DaBai DCW2，用 LeRobot 的 OpenCVCamera 取帧
│   ├── record.py · deploy.py   通用采集 / 部署流程
│   ├── task.py                 TaskSpec：一个任务的全部设定
│   ├── config.py               FPS、图像尺寸、关节限位
│   └── sim/                    MuJoCo 仿真：env.py（抓放场景 + IK）· phone.py（iPhone 遥操作）
├── tasks/
│   ├── charge_align/           充电口对准：task.py（TaskSpec）· record.py · deploy.py · start_pose.json
│   ├── pick_place/             抓取放置：同上
│   └── sim_pick_place/         仿真抓放：task.py · record.py（iPhone 遥操作采集）
├── assets/piper_x/             PiPER X 的 MuJoCo 模型（官方 URDF 转换，MIT；相机支架网格需自备，见下）
├── scripts/train.sh            共享训练脚本
├── third_party/lerobot/        LeRobot 0.4.4 官方源码（纯 Python，不用编译；可直接阅读/修改）
├── third_party/pyAgxArm/       松灵机械臂 SDK 1.0.0 官方源码（上游提交见 UPSTREAM_COMMIT）
├── third_party/hebi-py/        iPhone 遥操作通信库来源说明（SOURCE；安装包不随仓库分发，从 PyPI 装）
├── datasets/<任务名>/          LeRobot 数据集（采集后生成）
└── outputs/train/act_<任务名>/  训练出的模型（训练后生成）
```

| 任务 | 说明 |
|------|------|
| [charge_align](tasks/charge_align/README.md) | 手腕相机看到充电座，机械臂运动到充电座正前方的对准位姿并停住（纯端到端 ACT，6 关节，无夹爪） |
| [pick_place](tasks/pick_place/README.md) | 抓起物体放到目标位置（纯端到端 ACT，6 关节 + 夹爪开度） |
| [sim_pick_place](tasks/sim_pick_place/README.md) | MuJoCo 仿真里抓放方块，iPhone 遥操作采集；不需要真机 |

## 环境

```bash
conda create -n piper_act python=3.10 -y
conda activate piper_act
pip install hebi-py==2.11.0          # iPhone 遥操作通信库
pip install -e "third_party/lerobot[phone]"   # 工程自带的 LeRobot 源码 + 手机遥操作依赖（会顺带从网上装 torch、mujoco 等）
pip install -e third_party/pyAgxArm  # 工程自带的机械臂 SDK
pip install -e .                     # 最后装 piper_act（上面几个已满足版本要求，不会再从网上下载）
export PYTHONNOUSERSITE=1      # 必须：~/.local 里的 transformers 等会串进 conda 环境，lerobot 会 import 失败
```

仿真用到的手腕相机支架网格 `assets/piper_x/meshes/realsense_mid_stand.stl` 不随仓库分发（上游未声明许可证）：
从 [agilexrobotics/agx_arm_sim](https://github.com/agilexrobotics/agx_arm_sim) 的 `agx_arm_description/meshes/realsense_mid_stand.dae` 转成 STL 放到该路径。

每次上电后拉起 CAN：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

## 训练（各任务通用）

```bash
scripts/train.sh datasets/charge_align                  # 默认 cuda, batch 8, 100k 步
DEVICE=cpu STEPS=1000 BATCH=2 scripts/train.sh ...       # 覆盖参数
```

输出在 `outputs/train/act_<数据集名>/checkpoints/<步数>/pretrained_model`，已存在会报错（换名或删掉旧的）。

⚠️ 训练需要像样的 GPU：CPU 上约 1.5 s/步（batch 2），跑完 100k 步不现实。采集机没有 GPU 的话，把数据集目录整个拷到 GPU 机器上训练，再把 `pretrained_model` 目录拷回来。

## 新增任务

1. 复制 `tasks/pick_place/`（带夹爪）或 `tasks/charge_align/`（不带）为 `tasks/<任务名>/`，删掉里面的 `start_pose.json`
2. 改 `task.py` 里的 `TaskSpec`：`name`、任务描述、`gripper`、提示文字等。`record.py` / `deploy.py` 不用改
3. 采集 `python3 tasks/<任务名>/record.py`，训练 `scripts/train.sh datasets/<任务名>`，部署 `python3 tasks/<任务名>/deploy.py --policy ...`

采集和部署的流程本身在 `piper_act/record.py`、`piper_act/deploy.py`，改一处所有任务都生效。

## 真机注意事项

- 拖拽示教用 CAN 0x470 Leader-Follower 零力矩方式（`linkage_config=0xFA`）。MIT kp=kd=0、CAN 0x150 示教指令在真机上都实测无效。
- 同一个 CAN 口同时只能一个进程控制，脚本启动时会加锁 `/tmp/piperx_sync_<口>.lock`。
