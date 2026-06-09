# BCI 面试补强学习计划

> **目标**：7 天内完成面试核心知识补强，每天 2-3 小时
> **前置**：已完成 BCI 全链路项目 + 3 篇博客 + 网站改版

---

## Day 1：LSL 协议 + 10-20 电极系统

### 上午：LSL 协议（2h）

**学习目标**：
- 理解 LSL 架构：Outlet（发送端）→ 局域网 → Inlet（接收端）
- 了解时间同步机制（时钟偏移补偿）

**资源**：
- 官方文档：https://labstreaminglayer.readthedocs.io/
- pylsl 安装：`pip install pylsl`

**实践**（1h）：
```python
"""模拟 EEG Outlet + Inlet"""
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

# Outlet：模拟 8 通道 EEG，250Hz
info = StreamInfo('MockEEG', 'EEG', 8, 250, 'float32', 'myid123')
outlet = StreamOutlet(info)

print("发送数据 5 秒...")
for _ in range(1250):  # 5s * 250Hz
    chunk = np.random.randn(8) * 1e-6
    outlet.push_sample(chunk)
    time.sleep(0.004)

# Inlet：接收
streams = resolve_stream('name', 'MockEEG')
inlet = StreamInlet(streams[0])
sample, timestamp = inlet.pull_sample()
print(f"收到: {sample[:3]}..., timestamp={timestamp}")
```

**关键记住**：
| 概念 | 一句话 |
|------|--------|
| Outlet | 设备端，发布数据流 |
| Inlet | 接收端，订阅数据流 |
| LSL 时钟同步 | 每个数据包带时间戳，接收端本地守时 + 持续校准偏移 |

### 下午：10-20 电极系统（1h）

**记忆要点**：

```
        Nasion（鼻根）
         |
    Fp1 — Fpz — Fp2
      |    |    |
F3 ———— Fz ———— F4
      |    |    |
C3 ———— Cz ———— C4    ← 运动想象核心区
      |    |    |
P3 ———— Pz ———— P4
      |    |    |
O1 ———— Oz ———— O2    ← SSVEP 核心区（枕叶视觉皮层）

        Inion（枕外隆凸）
```

- **字母 = 脑区**：F=额叶、C=中央（运动皮层）、P=顶叶、O=枕叶（视觉皮层）、T=颞叶
- **数字 = 左右**：奇数=左、偶数=右、z=中线
- **运动想象**：C3（右手运动区）、C4（左手运动区）、Cz（足部）
- **SSVEP**：O1/O2/Oz（枕叶视觉皮层）
- 标准 10-20 系统共 21 个电极

### 验收
- [ ] 能默画 10-20 电极分布图，标出 C3/C4/Cz/O1/O2/Oz
- [ ] 能口述 LSL Outlet/Inlet 通信流程
- [ ] 跑通 pylsl demo 脚本

---

## Day 2：常见伪迹 + EEG 硬件设备

### 上午：常见伪迹（2h）

| 伪迹类型 | 来源 | 频率特征 | 去除方法 |
|---------|------|---------|---------|
| 眼电（EOG） | 眨眼/眼球运动 | 低频 <4Hz，额区电极幅度最大 | ICA 识别 EOG 成分后去除 |
| 肌电（EMG） | 面部/颈部肌肉 | 高频 >30Hz，宽带 | 带通 <40Hz + ICA |
| 工频干扰 | 电力线（50/60Hz） | 50Hz 窄峰 | Notch 滤波 |
| 电极漂移 | 出汗/松动 | 缓慢低频漂移 | 高通滤波 (>0.5Hz)、坏道插值 |
| 心电（ECG） | 心跳 | ~1Hz 周期性，QRS 复合波 | ICA 或 ECG 通道回归 |
| 运动伪迹 | 头部/身体运动 | 不确定，随运动方式变化 | 实验范式控制、幅值剔除 |

**MNE ICA 去伪迹流程**：

```python
import mne

# 1. 加载并预处理
raw = mne.io.read_raw_edf('data.edf', preload=True)
raw.filter(1, 40)  # 先滤波排除无用频段

# 2. 拟合 ICA
ica = mne.preprocessing.ICA(n_components=15, random_state=42)
ica.fit(raw)

# 3. 自动识别 EOG 成分（需要 EOG 通道标注）
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])

# 4. 观察并排除
ica.plot_components()  # 看各成分的地形图
ica.plot_scores(eog_scores)  # 看各成分与 EOG 的相关性

# 5. 应用
ica.apply(raw, exclude=eog_indices)
```

**关键记住**：
- ICA 需要足够数据（建议 >30 秒），在线场景不能用
- 去伪迹的顺序：**先滤波 → 后 ICA**
- EOG 伪迹的特征：额区幅度大、地形图前额分布、成分时间序列低频

### 下午：EEG 硬件设备（1h）

| 设备 | 类型 | 通道数 | 数据格式 | 接口 |
|------|------|--------|---------|------|
| OpenBCI | 开源 DIY | 8-16 | CSV/TXT | LSL / Bluetooth / USB |
| g.tec | 科研级 | 16-256 | HDF5/EDF | LSL / SDK / USB |
| BrainProducts | 科研级 | 32-256 | **.vhdr/.eeg/.vmrk** | USB / LSL |
| Neuroscan | 临床/科研 | 32-256 | **.cnt/.eeg** | USB / SDK |
| Neuracle（博睿康） | 国产科研 | 32-64 | 自有格式 | LSL / SDK |
| Emotiv EPOC/Insight | 消费级 | 5-14 | EDF/CSV | SDK / LSL |
| ANT Neuro | 科研级 | 32-256 | .edf/.cnt | USB / LSL |

**关键记住**：
- 你的 `bci/loader/` 已支持 EDF / FIF / EEGLAB(.set) / BrainVision(.vhdr) 四种格式，覆盖了 BrainProducts、Neuroscan（部分）、MNE 标准
- 面试可以说：**"loader 层做了格式抽象，支持 4 种主流 EEG 格式，新的后端只需实现 DataSource 协议"**

### 验收
- [ ] 能口述 EOG/EMG/工频/漂移四种伪迹的频率特征和去除方法
- [ ] 能写出 MNE ICA 去伪迹的核心代码
- [ ] 能说出 3 种以上 EEG 设备名称及格式

---

## Day 3：CSP 解码器实现

### CSP 原理（30 分钟）

**Common Spatial Pattern（共空间模式）**：找到空间滤波器 W，使得当 A 类时滤波后方差大、B 类时方差小（或相反），最大化两类在空间投影上的区分度。

```
CSP 核心思想：
  1. 计算两类协方差矩阵 R_A, R_B
  2. 求解广义特征值问题：R_A w = λ R_B w
  3. 取 λ 最大的 k 个特征向量 → W1（滤波后 A 类方差最大 / B 类最小）
  4. 取 λ 最小的 k 个特征向量 → W2（滤波后 B 类方差最大 / A 类最小）
  5. 拼接 [W1; W2] 得到完整的 CSP 投影矩阵
  6. 投影后提取 log-方差作为特征 → 输入 LDA 分类器
```

**关键记住**：CSP 本质是能找到两类在空间上的方差差异。CSP + LDA 是 MI-BCI 的经典 pipeline，2010-2015 年代的 SOTA。

### 实现（1.5h）

在 `bci/decoder/` 下新增 `csp.py`：

<｜DSML｜tool_calls>
<｜DSML｜invoke name="write">
<｜DSML｜parameter name="content" string="true">"""
CSP Decoder
===========
Common Spatial Pattern + LDA for Motor Imagery classification.

Pipeline: CSP(log-var features) → StandardScaler → LDA
Reference: Koles et al., 1990; Ramoser et al., 2000
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from bci.decoder.base import Decoder


class CSPDecoder(Decoder):
    """CSP + log-var + LDA decoder for Motor Imagery.

    Uses mne.decoding.CSP under the hood.
    Expects (n_epochs, n_channels, n_samples).
    """

    def __init__(self, n_components: int = 4, reg: str | None = None,
                 log: bool = True, random_state: int = 42):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.random_state = random_state
        self.pipeline: SkPipeline | None = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CSPDecoder':
        from mne.decoding import CSP
        classes = np.unique(y)
        self.classes_ = classes
        self.pipeline = SkPipeline([
            ('csp', CSP(n_components=self.n_components, reg=self.reg,
                        log=self.log, random_state=self.random_state)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis()),
        ])
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Must call fit() before predict_proba()")
        return self.pipeline.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'classes_': self.classes_,
            'n_components': self.n_components,
            'reg': self.reg,
            'log': self.log,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> 'CSPDecoder':
        import joblib
        state = joblib.load(path)
        obj = cls(n_components=state['n_components'],
                  reg=state['reg'], log=state['log'])
        obj.pipeline = state['pipeline']
        obj.classes_ = state['classes_']
        return obj