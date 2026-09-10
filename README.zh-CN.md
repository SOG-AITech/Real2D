# Real2D 固态电池模型

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库包含一个基于 FEniCSx 实现的 Real2D（R2D）固态全电池模型。该模型在恒电流充放电条件下，耦合求解锂金属负极、多晶固态电解质、碳/固态电解质复合层以及多颗粒正极中的电化学、相场和力学行为。

模型主要基于以下文献：

- Z.-T. Sun *等*，《Real 2D Galvanostatic Model: Encoding Physicochemical Heterogeneity into a Full Battery》，**Physical Review Letters** 135, 068001 (2025)，DOI：[10.1103/4783-dkt8](https://doi.org/10.1103/4783-dkt8)。
- 配套补充材料 `SunXuZhouBo_PRLSI25.pdf`，其中进一步介绍了几何结构、材料参数、自由能表达式、边界条件、Butler–Volmer 动力学以及数值实现。

代码实现了上述文献中的 R2D 建模方法，可用于二维数值模拟。若要复现论文中的特定图表，需要同时匹配对应的网格、参数、求解器配置和后处理流程。

## 仓库内容

| 文件 | 说明 |
| --- | --- |
| `r2d_irregular_3_34.msh` | Gmsh 二维计算网格。物理标签用于标识材料区域：`Omega1`（锂金属/多晶固态电解质相场区域）、`Omega2`（碳/固态电解质复合层）和 `Omega3`（正极颗粒）。包含标签 `101` 和 `102` 的网格分别将其用于仅初始化阶段的锂帽区域和晶粒区域。 |
| `generate_grain_boundaries.py` | 晶粒预处理与退火程序。该程序读取 `.msh` 网格，初始化多个固态电解质晶粒相场 `eta_i`，执行 Allen–Cahn 松弛，并计算晶界指示函数 `B` 及辅助场。最终场数据连同 PNG 预览图一起写入 `.npz` 文件。 |
| `r2d.py` | R2D 主求解器。该程序读取 `.msh` 网格和晶粒 `.npz` 文件，在恒电流充放电循环过程中求解锂相场 `xi`、晶粒场、锂浓度、电势、位移/应力及正极颗粒反应。 |

晶粒 `.npz` 文件是预处理结果，不能替代网格。主求解器通过 `--gb` 参数读取该文件，并将晶界指示函数用于有效电导率、反应和力学耦合计算。

## 环境要求

请使用与所安装 FEniCSx/DOLFINx 版本兼容的 Python 环境。需要以下软件包：

- FEniCSx / DOLFINx
- PETSc、`petsc4py` 和 `mpi4py`（推荐使用 MUMPS）
- UFL、Basix 和 NumPy
- Matplotlib，用于生成 PNG 预览图（运行求解器时可选）
- Gmsh，用于创建或修改网格

工作流程基于 MPI。开始正式计算前，请确认目标环境能够导入并运行 `python`、`mpirun`/`mpiexec` 和 DOLFINx。

## 使用流程

### 1. 生成晶界场

在仓库根目录运行预处理脚本：

```bash
mpirun -np 8 python generate_grain_boundaries.py
```

当前 `__main__` 入口使用默认网格名和输出名。它读取 `r2d_irregular_3_34.msh`，并生成：

- `r2d_120um_r3_comsol_layout1.npz`：供 `r2d.py` 使用的晶粒相场及晶界指示函数；
- `grain_boundary_previews/`：晶粒标签、`B` 场、辅助场及其他 PNG 诊断图。

若要使用其他网格或输出文件名，可以修改末尾对 `main(msh_file="your_mesh.msh", out_file="your_grains.npz")` 的调用，或为预处理脚本增加命令行参数解析。

`GrainParams` 数据类用于控制晶粒数量、随机种子、初始化方法（`partition_random`、`comsol_random` 或 `columnar`）以及退火参数。修改这些设置后，需要重新生成 `.npz` 文件。

### 2. 运行 R2D 模拟

显式指定网格和生成的晶粒文件：

```bash
mpirun -np 16 python r2d.py \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_120um_r3_comsol_layout1.npz \
  --cycles 7 \
  --current-density 10.0
```

也可以将生成的文件重命名或复制为 `r2d_irregular_3_34.npz`，然后使用求解器默认参数。正式计算可能需要较多内存和运行时间，请根据可用内存以及 PETSc/MUMPS 配置选择 MPI 进程数。

常用命令示例：

```bash
python r2d.py --help
python r2d.py --cycles 1 --t-end 100 --discharge-time 100 --progress-only
python r2d.py --linear-solver mumps --dt-max 5
```

`--cycles` 用于设置充放电循环次数。默认情况下，求解器使用自适应时间步、Newton/SNES 非线性求解、晶粒场后处理裁剪以及生命周期停止条件；当锂界面到达预设的正极侧目标位置时，计算会终止。

## 输出

默认情况下，`r2d.py` 将结果写入 `r2d_comsol_lr_periodic_tagged_side_120um_preview/`，其中包括：

- `diagnostics.csv`：已接受时间步的诊断信息，如电压、荷电状态、残差、时间步长和界面指标；
- `final_state.npz`：NumPy 格式的最终解和派生场；
- 锂相场、荷电状态、电势、晶界场、应力和位移的 PNG 快照；
- 可选的 XDMF 和采样场输出，可通过 `--xdmf-interval` 和 `--field-output-interval` 启用。

输出目录和输出间隔可在 `Params` 中修改，也可通过命令行覆盖。PNG 文件适合快速目视检查；定量后处理应使用 `final_state.npz`、CSV 或 XDMF 数据。

## 模型概述

该实现遵循参考文献中的混合电极自适应策略：

1. `Omega1` 使用多晶粒相场方法描述锂金属形貌、锂/固态电解质界面以及固态电解质晶粒结构。
2. `Omega2` 使用连续介质电化学模型描述碳/固态电解质复合层中的输运。
3. `Omega3` 使用锐界面/颗粒模型表示各个正极颗粒中的锂嵌入、Butler–Volmer 反应动力学和化学膨胀。
4. 线弹性模型计算各材料区域中的应力，静水应力通过力学过电位影响反应动力学。
5. 全局恒电流约束耦合负极和正极电流，并一致地确定电势。

## 重要说明

- `.msh` 文件中的物理标签必须与源代码约定一致。替换网格时，请先检查预处理程序或求解器输出的单元/面标签报告。
- 晶粒 `.npz` 文件必须由拓扑和尺寸兼容的网格生成。请勿复用由其他网格生成的晶粒文件。
- 正式计算会产生大量 PNG 和诊断文件。发布仓库前，建议将运行输出目录加入 `.gitignore`。
- PRL 论文及其补充材料定义了模型基础；本仓库实际采用的默认值以 `GrainParams`、`Params` 及源文件命令行帮助中的设置为准。

## 引用

如果在学术工作中使用本代码或其衍生版本，请引用上述 PRL 论文及补充材料，并注明模拟所采用的网格、参数集、FEniCSx/PETSc 版本、线性求解器和 MPI 配置。

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 许可。
