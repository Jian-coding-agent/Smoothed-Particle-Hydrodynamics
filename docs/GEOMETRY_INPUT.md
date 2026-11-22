# 从 STL / CAD 直接生成输入文件（几何驱动的粒子布置）

**English summary:** `tools/geom2deck.py` builds the 14-column SPH deck (or the
`.sphbin` container) from a domain box, a particle spacing, a fluid region and a
set of CAD parts -- STL triangles or analytic boxes -- instead of a hand-written
loop nest per block.  It reproduces the hand-written 8-hole gate deck particle
for particle; see "验证" below.

---

## 1. 结论

原来的 `1.cpp` 是一个**前处理器**，不是求解器：它按硬编码的 `nx×ny×nz = 70×330×95`
摆出水体粒子，再用 20 多段 `for(i=151;i<=160;i++)` 这类循环叠加两块闸门、一块挡板和四层
虚粒子，最后写出 `input.dat`。换一个案例就要重写这些下标。

现在这件事可以像 FLOW-3D 那样做：给它**计算域 + 粒子间距 + 水体范围 + CAD 实体**，
它把实体"体素化"到同一套格点上，再按角色分配 `itype`，写出求解器能读的 deck。

```
tools/geom2deck.py    生成器（约 870 行，Python + numpy，无第三方依赖）
cases/8hole_gate.json 8 孔闸门案例的几何描述（56 个 box）
cases/make_8hole_gate.py  用参数重新生成上面那个 json
tools/test_geom2deck.py   自检（6 项，约 25 s）
```

## 2. 和 FLOW-3D 的对应关系

FLOW-3D 的流程是：导入 STL → 定义组件（component）→ 划网格 → FAVOR 把每个网格单元
标成流体/固体/部分占据（记录面积、体积分数）→ 在流体单元里布置粒子。

这里做的是同一件事，只有一处本质差别：

| FLOW-3D / FAVOR | 本工具 |
|---|---|
| 笛卡尔网格，单元可被部分占据，用**面积/体积分数**精确表示固体边界 | 只在格点上放粒子，固体边界是**半格宽的台阶** |
| 粒子数由"流体单元内铺几层"决定 | 粒子位置 = 格点，间距 = hsml |
| 读 STL，支持组件变换 | 读 STL（二进制/ASCII）+ 轴对齐 box，支持缩放/旋转/平移 |
| 图形界面、网格无关性检查 | 命令行 + 统计输出（每块计数、体积比、流道尺寸） |

体积分数这一条在本求解器里补不上：`mass[i]` 是统一的 `ρ·dx³`，没有"部分质量"的概念。
所以几何精度就停在半个粒子间距上，这是方法本身的限制，不是实现问题。

## 3. 角色 → itype

求解器只按 `itype` 分派，所以几何部件的"角色"直接映射到这三个值：

| 角色 | itype | 求解器里的含义 | 需要的粒子范围 |
|---|---|---|---|
| `fluid` | `2`（>0） | Riemann 流体对，密度由连续方程更新 | 水体内部全部格点 |
| `elastic` | `-20`（<-10） | 全拉格朗日固体，自带应力更新 | 实体**全部体积**（它是被离散的固体本身） |
| `rigid` | `-2`（[-10,0]） | Monaghan 边界粒子，位置固定 | 只需贴着可变形区的**若干层** |

最后一条是粒子数的大头：一块混凝土不用几百万个粒子，只要贴着流体的
`ceil(3·hsml/dx)+1` 层。`SPH_CS = 3·hsml`（`cuda/src/sph.h`）是核支撑半径，
所以 `hsml = dx = 0.01` 时 3 层正好在支撑边缘，4 层比较稳妥 —— 原 `1.cpp` 用的就是 4 层。

case 文件里 `"layers": N` 就是这件事：`N=0` 表示填满整个实体（适合挡板、地板这种
本来就薄的东西），`N>0` 表示只保留贴着流体/弹性体的那 N 层。

## 4. 用法

### 4.1 写一个 case 文件

```json
{
  "spacing": 0.01,
  "domain": { "min": [-0.03, -0.03, -0.03], "max": [3.60, 3.34, 1.00] },
  "parts": [
    { "name": "water", "role": "fluid",
      "geom": { "type": "box", "min": [0.01, 0.01, 0.01], "max": [0.70, 3.30, 0.95] } },

    { "name": "gate", "role": "elastic",
      "geom": { "type": "stl", "file": "gate.stl", "scale": 0.001 } },

    { "name": "tank", "role": "rigid", "layers": 4,
      "geom": { "type": "stl", "file": "tank.stl" } }
  ]
}
```

`geom` 有四种写法：

| `type` | 字段 | 说明 |
|---|---|---|
| `box` | `min`, `max` | 轴对齐长方体，**闭区间**（两端都含），精确对齐格点 |
| `boxes` | `boxes`: `[[x0,y0,z0,x1,y1,z1], ...]` 或 `[{"min":…,"max":…}, …]` | 多个 box 取并集；实体有重叠也没关系 |
| `stl` | `file`，可选 `scale`, `rotate:{axis,angle_deg}`, `translate`, `seal` | 封闭三角面片；建议一个部件一个文件。`seal: N` 先把表面向外膨胀 N 格再灌水，能封住约 2N 格宽的小洞 |

其余可选字段：`itype`（覆盖默认值）、`rho`（默认 1000）、`velocity: [vx,vy,vz]`（默认 0）。

### 4.2 生成

```bat
python tools\geom2deck.py cases\8hole_gate.json --sphbin input.sphbin
python tools\geom2deck.py cases\8hole_gate.json --out input.dat
python tools\geom2deck.py cases\8hole_gate.json --check 老deck.sphbin
python tools\geom2deck.py cases\8hole_gate.json --dump-stl tmp\stl
```

- `--sphbin` 写二进制容器（CUDA 求解器直接读，397 MB / 370 万粒子，约 0.4 s）；
- `--out` 写 14 列文本 deck（CPU 参考求解器用，约 211 MB，制表符分隔、CRLF、`%.6g`，
  与原 `1.cpp` 的输出格式逐字符一致）；
- `--check` 和已有 deck 对比粒子集合，逐块报"格点完全相同 / 差多少"；
- `--dump-stl` 把每个 box 部件写成 ASCII STL，并生成一个指向这些 STL 的
  `case_stl.json` —— 用它可以把同一个案例改走 STL 路径，验证两条路径等价。

典型输出：

```
lattice 364 x 338 x 104 = 12,795,328 cells (12.2 MiB as bool), spacing 0.01 m
    water            box x1          ->    2,194,500 cells  (0.01 s)
    gate_upstream    box x25         ->      138,000 cells  (0.01 s)
    ...
  fluid    itype 2        2,194,500 particles  x[0.010,0.700] ...
  elastic  itype -20        606,000 particles  x[1.510,3.600] ...
  rigid    itype -2         915,328 particles  x[-0.030,3.600] ...
total 3,715,828 particles, 2194.5 L of water, 396.9 MiB of deck
```

### 4.3 跑

```bat
cd cuda && build.bat && cd ..
cuda\build\sph_cuda.exe _local\geom\input_8hole.sphbin --fp32
```

注意 `.sphbin` 要**运行时分配 + 二进制 deck** 的那个构建（提交 `a106b7b` 及之后）。
更早编译出来的 exe（比如本仓库 `cuda\build\` 里 11-18 那个）只认 237k 固定算例，
读 `.sphbin` 会报 `input parse error at particle 0 col 0`；那种情况先用 `--out` 写文本 deck。

## 5. 8 孔闸门案例：1300 行 → 56 个 box

`1.cpp` 里这块几何是十段几乎一样的循环。在 case 文件里它是参数：

```
水体    x[0.01,0.70] y[0.01,3.30] z[0.01,0.95]          70×330×95
闸门×2  x = 1.51 / 2.41，厚 0.10 m
        9 根墩，宽 0.10 m、间隔 0.40 m  → 8 个孔
        8 个孔上方各两块梁，z[0.61,0.70] 和 [0.91,1.00]
挡板    x[3.51,3.60] 通高
池壁    地板 + 上游面 + 下游面 + 左面，各 4 层，位于水体之外
```

`cases/make_8hole_gate.py` 用 30 行参数生成这 56 个 box；`geom2deck.py` 把它们
展开成格点。想改孔数、墩宽、间距，改参数重跑即可，不用碰 C++。

## 6. 验证

三组独立证据，全部在本机复现：

**(1) 与手写生成器逐粒子相同**

```
python tools\geom2deck.py cases\8hole_gate.json --check _local\case8hole\input3.sphbin
  fluid    ok  2,194,500 particles, same lattice cells, max |dx| = 4.44e-16 m
  elastic  ok    606,000 particles, same lattice cells, max |dx| = 4.44e-16 m
  rigid    ok    915,328 particles, same lattice cells, max |dx| = 4.44e-16 m
check: identical particle sets
```

4.44e-16 m 是 1 个 ulp：原程序写 `i*dx0+dx0`，本工具写 `x0+i*dx`，同一个格点、末位差 1。
块内粒子顺序不同（求解器与顺序无关：邻域表、块划分都只看坐标和 itype）。

更强的结论在**文本 deck** 上：`--out` 写出的 211,709,020 字节与 `1.cpp` 的
`input.dat` **字节数相同**，两边的行做哈希排序后**逐行完全相同** —— 也就是
"同样的行、只换了块内顺序"。`%.6g` 那点末位差异被 6 位有效数字吃掉了。

**(2) STL 路径与 box 路径等价**

把上面 56 个 box 导出成 STL（672 个三角片），再走 STL 路径重新生成，
三个块的粒子集合与 box 路径**完全相同**（`tools/test_geom2deck.py` 第 3 项）。

**(3) 求解器端到端**

```bat
set SPH_MAXSTEP=550
cuda\build\sph_cuda.exe _local\geom\input_8hole.sphbin --fp32     :: 几何生成
cuda\build\sph_cuda.exe _local\case8hole\input3.sphbin  --fp32    :: 原 1.cpp
```

两份 deck 各跑 500 步（39.9 vs 42.8 ms/step），按初始位置配对后：

| 量 | 差值 |
|---|---|
| 单个粒子位移 | ≤ 2.4e-7 m（域长 3.63 m） |
| 单个粒子压力 | ≤ 1.7 Pa（压力幅值 0–2363 Pa），中位数 0 |

t=0 时邻居对数 187,250,782 完全相同 —— 两个 deck 是同一个算例。

**(4) 曲面精度**

格点对齐的 box 是精确的（两端都是格点）。曲面只能近似，实测（dx = 0.01 m）：

| 几何 | 体积误差 |
|---|---|
| r = 0.30 m、h = 1.0 m 圆柱，轴线与格子对齐 | +2.1% |
| 同上，绕 x 轴转 30° | +0.33% |
| 同上，绕 (1,1,1) 转 20° | −0.04% |

对齐的圆柱偏大是因为**边界被算作实体内部**：几何面正好落在格点上时，在面上的
格点算"在实体内"，于是每个方向多出半层 —— 这正是 `1.cpp` 的约定（`x[0.01,0.70]`
取 70 个点，两端都含）。斜面上这个效应被格点取样平均掉，所以误差反而小。

生成器会为每个 STL 部件打印 `vol x1.03` 这样的**体积比**（格子体 / 网格体积），
薄板按定义会偏大（3 层厚的板算 4 层 → ×1.33），**偏小才是出问题**：
低于 0.9 会警告"曲面可能不封闭，试试 `seal`"。

## 7. 局限与注意事项

**必须知道的**

1. **`hsml` 和 `dt` 是编译期常量**（`cuda/src/sph.h` 的 `SPH_HSML` / `SPH_DT`）。
   deck 里没有它们。所以 `spacing` 必须等于 `SPH_HSML`，改间距要重新编译求解器；
   `spacing`（或 `--hsml`）不等于 0.01 时生成器会警告。
2. **CPU 参考求解器 `cpu/1.cpp` 的 `num` 是编译期常量**（`const int num=237298`，
   静态数组）。换粒子数要改这一行重编。CUDA 版是运行时分配，不用动。
3. **格点规模 = 域体积 / dx³**。本案例 12.8 M 个格子（单个 bool 数组 12 MB，
   体素化峰值约 200 MB）；`dx` 减半就是 8 倍。
   100 m × 100 m × 10 m 的域在 0.01 m 下是 10¹² 个格子，跑不动 —— 这类问题得改
   SPH 的变间距布置，不是这个工具的范畴。
4. **STL 必须是封闭、法向朝外的实体**。水面开口的"板"可以（就得到一层粒子），
   但实体的破洞会让 flood fill 灌进去，实体变成空壳。`"seal": 1` 先把表面膨胀
   一格再灌，能封住约 2N 格宽的小洞；更大的洞封不住 —— **它是权宜手段，不是
   mesh 修复**，能修 CAD 就修 CAD。体积比警告（< 0.9）是同一件事的自动检查。
5. **空腔**：实体内部封闭的空腔会被填实。要空腔就把空腔本身作为 `fluid` 部件导出来。
6. 一个部件一个 STL。多个实体叠在一起导出成一个 STL 也能用（并集），
   但内部面会落在格点上，不如分开写干净。

**从 CAD 导出时**

- **单位**：SolidWorks 默认导出毫米，用 `"scale": 0.001`；或者导出前就换成米。
  忘了缩放的典型症状是"部件产生 0 个粒子"（跑到域外去了）。
- **坐标**：本求解器的 z 是竖直方向，域 `min`/`max` 都要用米、且与 CAD 同一原点。
- **只导出实体表面**，不要带装配体的基准面、草图、曲面片。
- 三角化精度（弦高）取 `dx/5` 以下就够了，再细只是拖慢体素化。

## 8. 还能往哪走

按"离 FLOW-3D 还差什么"排优先级：

1. **面积/体积分数**：给边界粒子按"格子被实体占据的体积比"缩质量，或引入
   `V_i` 数组，能把台阶误差从半格降到 ~0.1 格。要改求解器（`mass[i]` 现在是常量
   `ρdx³`），工作量中等。
2. **直接读 STEP/IGES**：需要 OpenCASCADE，重；现实中先让 CAD 导出 STL 更划算。
3. **图形化预览**：生成后把每块粒子的包围盒/切片画成 PNG，比看数字快。可以用
   `--dump-stl` 的几何 + matplotlib 切片。
4. **变间距（分级）布置**：远离关注区用 2dx、3dx 的粒子，粒子数能降一个量级；
   求解器要支持变 hsml，改动大。
