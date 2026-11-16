# 光滑粒子流体动力学（SPH）

**三维 SPH 流固耦合求解器：CPU 参考实现 + 经过验证的 CUDA 移植版。**

中文 · [English](README.md)

---

本仓库包含一个弱可压缩 SPH 求解器，采用 Riemann 解算器形式的粒子对相互作用、
总拉格朗日格式的固体（含 Kelvin–Voigt 摩擦项），以及镜像 Riemann 流固耦合。
代码有两种形态：

* **`cpu/`** —— 原始的单编译单元 C++/OpenMP 参考实现（`1.cpp`，双精度），
  是正确性基准；
* **`cuda/`** —— 同一套物理模型的 CUDA 移植版；标量类型做成模板，同一个
  可执行文件可在运行时选择 **fp64**（默认）或 **fp32**。

两者读取同一份 `input.dat`（237 298 个粒子：101 250 流体、60 320 固体、
75 728 墙面；t = 0 时约 1 111 万个邻居对），每 50 步输出一个 `result<N>.dat`，
时间步长固定为 `dt = 1e-5`。

## 性能

在 NVIDIA RTX 5080（`sm_120`）、CUDA 13.3、237 298 个粒子上实测：

| 实现 | 精度 | ms/步 | 相对 32 线程 CPU | 精度表现 |
|---|---|---:|---:|---|
| `cpu/` | fp64 | 211 | 1x | 基准 |
| `cuda/` | fp64 | **12.42** | **17.0x** | 与参考结果在打印精度内一致 |
| `cuda/` | fp32 | **2.43** | **87x** | 固体 `p` 中位数 0.01 Pa，`x` 偏差在 1e-6 m 内 |

CPU 基准为 16 核 Ryzen 9 9950X、32 个 OpenMP 线程。相同参数跑两遍，
两种精度的 `result*.dat` 都是**逐字节一致**的。

## 实现要点

移植没有改动参考格式的物理，只针对 GPU 重新组织了计算：

* 所有粒子对相互作用都改写成 **gather** 形式（每个粒子一个线程），
  每步只需少数几个 kernel，且不需要原子操作；
* 均匀网格给出邻居候选，先用 float 预筛，只有通过预筛的候选才做精确距离
  判定（fp32 版本里精确判定本身就是 float，预筛用 `if constexpr` 编译掉）；
* 固体/固体粒子对在参考构型下**只算一次**（总拉格朗日格式使它们不随时间
  变化），两个固体 kernel 因此退化成扁平的 CSR 循环；
* 结果文件用 `std::to_chars` 多线程分段格式化后顺序写出。

完整推导、以及必须逐字复刻的参考实现“怪癖”，见
[`cuda/docs/DESIGN.md`](cuda/docs/DESIGN.md)。

## 验证

对参考结果 `result0..9.dat`（500 步）：

* `x0 x1 x2`、`rho`、`mass`、`itype`、`X0`、`rho0` —— 打印精度内**零差异**；
* `p` —— 最大绝对误差 **1e-3**；`vx` —— 最大绝对误差 **1e-9**。

对纯净 CPU 跑出的 3000 步结果（第 566 步发生流固接触）：
`p` 最大绝对误差 **1e-2**，`vx` **1e-9**，与 CPU 自身两次运行的离散度同量级。
单步后的中间数组相对误差在 2.2e-15 … 2.5e-8 之间；
`compute-sanitizer` 的 `memcheck`、`initcheck`、`racecheck`、`synccheck`
均未报错。

## 快速开始

### CUDA 版

环境要求：NVIDIA GPU（`sm_70` 及以上）、CUDA 13.x、带 C++ 工作负载的
MSVC 2019/2022/2026、CMake ≥ 3.24。

```bat
cd cuda
build.bat
```

然后从**仓库根目录**运行，因为求解器从当前目录读取 `input.dat`：

```bat
set SPH_MAXSTEP=500
cuda\build\sph_cuda.exe          :: fp64
cuda\build\sph_cuda.exe --fp32   :: 单精度
```

运行会在当前目录写出 `result0.dat` … `result9.dat`（每 50 步一个）。
其他选项见 `cuda\build\sph_cuda.exe --help`。

### CPU 参考版

环境要求：Visual Studio 2019（v142）或更新版本（含 C++ 工作负载）、
CMake ≥ 3.20。

```bat
cd cpu
build.bat
cd ..
set SPH_MAXSTEP=500
cpu\build\Release\SPH.exe
```

CPU 版慢得多（32 线程跑 500 步约 105 秒）。可用 `SPH_THREADS` 限制线程数，
用 `SPH_MAXSTEP` 提前停止。

## 目录结构

```
input.dat                共用输入文件（237 298 个粒子，约 21 MB）
cpu/                     参考实现：1.cpp、CMake 工程、构建脚本
cuda/                    CUDA 移植：src/、CMakeLists.txt、build.bat
  docs/DESIGN.md         设计、移植笔记、验证与性能分析
docs/
  CUDA_PORT_ANALYSIS.md  源码分析与移植方案（中文）
```

## 文档

| 文档 | 内容 |
|---|---|
| [`cuda/docs/DESIGN.md`](cuda/docs/DESIGN.md) | 算法映射、精度模板、验证、实测性能、后续工作 |
| [`docs/CUDA_PORT_ANALYSIS.md`](docs/CUDA_PORT_ANALYSIS.md) | `1.cpp` 源码分析与初始移植方案 |
| [`cuda/README.md`](cuda/README.md) | CUDA 版构建、运行选项与诊断开关 |
| [`cpu/README.md`](cpu/README.md) | CPU 工程、构建矩阵、常见问题 |
| [`cpu/QUICKSTART.txt`](cpu/QUICKSTART.txt) | 一页版 CPU 快速上手 |

## 说明

* 500 步的参考结果（`result0..9.dat`，约 200 MB）**没有**放进仓库。
  可用 CPU 版重新生成，再用 `cpu\tools\CompareResults.exe <运行目录> <参考目录>`
  对比。
* 构建脚本面向 Windows/MSVC；CUDA 源码本身是标准 CUDA C++。
* `input.dat` 是 14 列的文本输入，求解器按工作目录的相对路径读写。

## 许可证

MIT，见 [LICENSE](LICENSE)。
