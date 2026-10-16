"""iPhone（HEBI Mobile I/O App）遥操作 → 仿真里的末端目标（指尖位置 + 夹爪开度）。

手机位姿和按键用 LeRobot 的 IOSPhone 读取。方向：默认每次开始跟随时按手机顶边朝向定"前"；
calibrate_directions + PhoneTeleop.set_frame 可改为按实际动作（往前推、往上抬）定方向。夹爪保持竖直朝下、转角不变。

B1（跟随 开/关）是开关模式，其余按键是普通按钮（点一下弹起），程序看按下沿。

手机上的操作：
  点一下 B1  开始跟随手机移动（相对移动：开始那一刻的位置为起点）；再点一下 = 暂停，可以挪手再点
  A3 滑块  往上推张开、往下拉闭合，松手回中 = 夹爪停住
  B2 保存本条   B3 丢弃重来   B8 退出
"""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np

from .env import GRIP_OPEN

GRIP_SPEED = 0.2             # m/s，A3 推满时夹爪开合速度
MAX_SPEED = 0.5              # m/s，末端目标最大速度（防 AR 跟踪跳变）
WORKSPACE = (np.array([0.15, -0.35, 0.005]), np.array([0.55, 0.35, 0.25]))   # 指尖目标范围（m）
LABELS = {1: "跟随 开/关", 2: "保存", 3: "丢弃重来", 8: "退出"}


def lan_broadcasts() -> list[str]:
    """本机各局域网网卡的广播地址（如 10.168.1.255）。

    HEBI 靠 UDP 广播找手机。默认发往 255.255.255.255，而代理软件（sing-box/Clash 的 TUN 模式）
    会把这个地址路由进它的虚拟网卡、发不到 WiFi 上；按网段广播地址发就走真实网卡。
    """
    out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True).stdout
    addrs = []
    for line in out.splitlines():   # "3: wlp0s20f3    inet 10.168.1.102/24 brd 10.168.1.255 scope global ..."
        f = line.split()
        if "brd" in f and not f[1].startswith(("lo", "docker", "br-", "veth", "virbr")) and "tun" not in f[1]:
            addrs.append(f[f.index("brd") + 1])
    return addrs


class _WaitingGroup:
    """包一层 HEBI group：get_next_feedback 偶尔超时返回 None 时接着等（刚连上时第一包还没到就会这样），
    LeRobot 的 IOSPhone 没处理 None 会直接报错。连续 3 秒收不到才算手机断开。"""

    def __init__(self, group, max_wait: float = 3.0):
        self._group, self._max_wait = group, max_wait

    def __getattr__(self, name):
        return getattr(self._group, name)

    def get_next_feedback(self, *args, **kwargs):
        t0 = time.monotonic()
        while (fbk := self._group.get_next_feedback(*args, **kwargs)) is None:
            if time.monotonic() - t0 > self._max_wait:
                raise RuntimeError(f"{self._max_wait:.0f} 秒没收到手机数据：HEBI Mobile I/O 是否还在前台、WiFi 是否断开")
        # 留住整包：LeRobot 的 _read_current_pose 只传出 fbk[0]，fbk 随即被回收，fbk[0] 里的按键就读成空的
        # （实测 393 包里只有 3 包读得到 B2；之前以为是手机只在少数包里带按键状态，其实是这个原因）
        self._last = fbk
        return fbk


def connect_iphone(timeout: float = 60.0):
    """连接 HEBI Mobile I/O，按键全部设成开关模式、写中文标签，完成标定（屏幕朝上平放时点一下 B1）。返回 IOSPhone。"""
    import hebi
    from hebi._internal.mobile_io import MobileIO
    from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
    from lerobot.teleoperators.phone.teleop_phone import IOSPhone

    addrs = lan_broadcasts()
    print("连接 iPhone：打开 HEBI Mobile I/O（保持在前台），Family = HEBI、Name = mobileIO，手机和电脑连同一个 WiFi。\n"
          f"在 {addrs} 上查找…")
    # 不用 IOSPhone.connect()：它用默认 hebi.Lookup()（发往 255.255.255.255），有代理时找不到手机
    lookup = hebi.Lookup(addrs)
    group, t0 = None, time.monotonic()
    while group is None and time.monotonic() - t0 < timeout:
        time.sleep(1.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
    if group is None:
        seen = [f"{e.family}/{e.name}" for e in lookup.entrylist]
        sys.exit(f"❌ {timeout:.0f} 秒内没找到 HEBI/mobileIO。看到的 HEBI 设备：{seen or '无'}\n"
                 "   一个都没有 → 网络不通：手机和电脑是否同一个 WiFi（看手机 IP 是否同网段）、"
                 "路由器是否开了 AP 隔离、防火墙（sudo ufw status）是否拦了 UDP\n"
                 "   有设备但名字不对 → 改 App 设置里的 Family / Name")
    phone = IOSPhone(PhoneConfig(phone_os=PhoneOS.IOS))
    phone._group = _WaitingGroup(group)
    io = MobileIO(group)
    for b, label in LABELS.items():   # B1 开关：点一下开始跟随、再点一下暂停；其余普通按钮
        io.set_button_mode(b, "toggle" if b == 1 else "momentary")
        io.set_button_label(b, label, blocking=False)
    io.set_axis_label(3, "夹爪 ↑开 ↓合", blocking=False)
    io.set_snap(3, 0.0)   # A3 松手回中

    def b1_state():
        """读到带 B1 的反馈包为止，返回 (B1, 手机位置, 手机姿态)。"""
        while True:
            ok, pos, rot, fb = phone._read_current_pose()
            if ok and fb.io.b.has_int(1):
                return fb.io.b.get_int(1), pos, rot

    # 不用 IOSPhone.calibrate()：它要"按住 B1"，这里 B1 是开关，点一下就行
    print("✓ 已连上。标定：手机屏幕朝上平放，点一下「跟随 开/关」…")
    s0 = b1_state()[0]
    while (r := b1_state())[0] == s0:
        pass
    phone._calib_pos, phone._calib_rot_inv = r[1].copy(), r[2].inv()
    print("✓ 标定完成")
    if r[0]:
        print("  再点一下「跟随 开/关」关掉…")
        while b1_state()[0]:
            pass
    return phone


class PhoneTeleop:
    """把手机动作累积成末端目标：pos（指尖位置 m）、grip（夹爪开度 m）、enabled（是否在跟随）。

    不用 IOSPhone.get_action()：它读按键时整包已被回收、大多读成空的（见 _WaitingGroup），把"没读到 B1"
    当成 B1 松开，跟随状态来回跳、起点不停被重置。这里记住每个按键/滑块最后一次读到的值，自己判断跟随开关。

    方向：每次开始跟随的那一刻，手机顶边的水平朝向 = 机械臂 +x（前），左 = +y，上 = +z。"上"取自启动标定
    （标定时屏幕朝上平放）。启动标定时手机朝向不对（比如人还在电脑前）也没关系，开始跟随前对准就行。
    """

    def __init__(self, phone, scale: float = 1.0):
        self.phone = phone            # 已标定的 LeRobot IOSPhone（用它的 _read_current_pose / _calib_rot_inv）
        self.scale = scale            # 手机移动 1 m → 指尖移动 scale m
        self.pos, self.grip, self.enabled = np.zeros(3), GRIP_OPEN, False
        self._anchor, self._raw_anchor = np.zeros(3), np.zeros(3)
        self._inputs: dict[str, float] = {}           # 各按键/滑块最后一次的值
        calib = phone._calib_rot_inv.inv()            # 标定时手机的姿态（AR 坐标系下）
        self._up = calib.apply([0.0, 0.0, 1.0])       # 标定时屏幕朝上 → 屏幕法向 = 竖直向上
        self._frame = phone._calib_rot_inv.as_matrix()  # AR 坐标 → 机械臂坐标（行 = 前/左/上），开始跟随时重算
        self._fixed = False                            # True：方向已用 set_frame 定死，开始跟随时不再按顶边重算
        self.rot = self._rot_ref = None                # 手机当前姿态 / 开始跟随那一刻的姿态（只用于显示）
        self._buttons: dict[str, float] | None = None  # 上一帧按键状态；None = 还没读过

    def reset(self, pos: np.ndarray, grip: float) -> None:
        self.pos, self.grip, self.enabled = np.array(pos, float), grip, False

    def hold_at(self, pos: np.ndarray) -> None:
        """目标够不到时调用：目标退回 pos，起点跟着平移。目标不会在够不到的地方越跑越远，
        手机一往回移机械臂就跟着回来（像碰到墙），不会攒着一段距离、之后朝意外的方向突然动。"""
        pos = np.asarray(pos, float)
        self._anchor += pos - self.pos
        self.pos = pos.copy()

    def set_frame(self, frame: np.ndarray) -> None:
        """用 calibrate_directions 的结果定死方向。"""
        self._frame, self._fixed = np.asarray(frame, float), True

    def _align(self, rot) -> None:
        """手机顶边（机身 x 轴）投影到水平面 = 前。顶边几乎竖直时朝向不可靠，沿用上一次的。"""
        head = rot.apply([1.0, 0.0, 0.0])
        head -= head.dot(self._up) * self._up
        n = np.linalg.norm(head)
        if n > 0.3:
            fwd = head / n
            self._frame = np.array([fwd, np.cross(self._up, fwd), self._up])

    def axes(self) -> np.ndarray:
        """手机相对开始跟随那一刻转了多少，换算到机械臂坐标：3x3，列 = 跟着手机转过的前/左/上方向。
        开始跟随那一刻是单位阵。只用于显示，控制只用手机位置。"""
        if self.rot is None or self._rot_ref is None:
            return np.eye(3)
        return self._frame @ (self.rot * self._rot_ref.inv()).as_matrix() @ self._frame.T

    def update(self, dt: float) -> set[str]:
        """读一次手机，更新目标；返回这一帧刚按下的按键（如 {"b2"}）。手机 AR 跟踪丢失时目标不动。"""
        ok, raw_pos, raw_rot, fb = self.phone._read_current_pose()
        if not ok:
            return set()
        self.rot = raw_rot
        io = fb.io
        for ch in range(1, 9):
            if io.b.has_int(ch):
                self._inputs[f"b{ch}"] = int(io.b.get_int(ch))
            if io.a.has_float(ch):
                self._inputs[f"a{ch}"] = float(io.a.get_float(ch))
        enabled = bool(self._inputs.get("b1", 0))
        if enabled and not self.enabled:   # 开始跟随：以当前目标和手机当前位置为起点，手机顶边朝向为前
            self._anchor, self._raw_anchor = self.pos.copy(), raw_pos.copy()
            self._rot_ref = raw_rot
            if not self._fixed:
                self._align(raw_rot)
        if enabled:
            delta = self._frame @ (raw_pos - self._raw_anchor)
            target = self._anchor + self.scale * delta
            step = target - self.pos
            n = np.linalg.norm(step)
            if n > MAX_SPEED * dt:
                step *= MAX_SPEED * dt / n
            new = self.pos + step
            self.pos = np.clip(new, *WORKSPACE)
            self._anchor += self.pos - new   # 截到范围边界时起点跟着平移，目标不会在范围外越跑越远
        self.enabled = enabled
        grip_vel = self._inputs.get("a3", 0.0)   # A3 滑块 -1..1：往上推张开、往下拉闭合
        self.grip = float(np.clip(self.grip + grip_vel * GRIP_SPEED * dt, 0.0, GRIP_OPEN))
        buttons = {k: v for k, v in self._inputs.items() if k.startswith("b")}
        prev, self._buttons = self._buttons, buttons
        if prev is None:   # 第一帧只记状态：开机时就是按下状态的键不算"刚按下"
            return set()
        return {k for k, v in buttons.items() if k != "b1" and v and not prev.get(k)}


def calibrate_directions(phone) -> np.ndarray:
    """按实际动作定方向：人站在机械臂后方，手机往机械臂前方水平推一段、再竖直往上抬一段。

    返回 AR 坐标 → 机械臂坐标的 3x3 矩阵（行 = 前/左/上）。只用两段位移的方向，不依赖手机怎么拿、
    也不依赖手机机身坐标轴的定义。每段用「保存」点一下开始、点一下结束。
    """
    b2 = [0]

    def read() -> np.ndarray:
        while True:
            ok, pos, _, fb = phone._read_current_pose()
            if fb is not None and fb.io.b.has_int(2):
                b2[0] = fb.io.b.get_int(2)
            if ok:
                return pos

    def tap() -> np.ndarray:
        """等「保存」点一下，返回按下那一刻的手机位置。"""
        while b2[0]:
            read()
        while True:
            p = read()
            if b2[0]:
                return p

    def move(what: str) -> np.ndarray:
        while True:
            print(f"  手机拿稳，点一下「保存」；然后{what}约 20 cm，停住再点一下「保存」…", flush=True)
            p0 = tap()
            d = tap() - p0
            if np.linalg.norm(d) > 0.08:
                return d
            print(f"  只移动了 {np.linalg.norm(d) * 100:.0f} cm，太短，重来")

    print("\n方向标定（机械臂不动）：人站在机械臂后方，面朝机械臂伸出去的方向。")
    while True:
        f = move("往机械臂前方水平推")
        u = move("竖直往上抬")
        up = u / np.linalg.norm(u)
        axes = np.vstack([np.eye(3), -np.eye(3)])   # 手机 AR 坐标系按重力对齐，必有一根轴竖直向上：
        k = int(np.argmax(axes @ up))               # 实测的上抬方向和它差不到 25° 就用它，消掉手抬歪的误差
        if axes[k] @ up > np.cos(np.radians(25)):
            up = axes[k]
        ang = np.degrees(np.arccos(min(1.0, abs(f.dot(up)) / np.linalg.norm(f))))
        if ang > 60:
            break
        print(f"  前推和上抬的方向夹角只有 {ang:.0f}°（应接近 90°），重来")
    fwd = f - f.dot(up) * up
    fwd /= np.linalg.norm(fwd)
    print("✓ 方向标定完成")
    return np.array([fwd, np.cross(up, fwd), up])
