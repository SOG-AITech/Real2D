# R2D 多晶电化学–相场–力学耦合模拟

[English](README.md) | [简体中文](README.zh-CN.md)

本项目使用 FEniCSx/DOLFINx、PETSc 和 MPI，模拟多晶材料中的晶界演化以及电化学、相场和力学耦合行为。

仓库包含两个独立的工作流程：

1. `generate_grain_boundaries.py` 从 Gmsh 网格生成晶界场和 eta 场。
2. `r2d.py` 读取网格及生成的晶界场，运行 R2D 耦合模拟。

这两个程序相互独立，需要分别运行。

## 工作流程

```text
Gmsh 网格 (.msh)
        |
        v
generate_grain_boundaries.py
        |
        +--> 晶界场 / eta 场 (.npz)
                         |
                         v
              r2d.py --msh mesh.msh --gb grain_boundary.npz
                         |
                         +--> PNG / XDMF / NPZ / CSV
```

## 环境要求

请在包含以下依赖的 FEniCSx 环境中运行本项目：

- Python 3.10 或更新版本
- FEniCSx / DOLFINx、UFL 和 Basix
- PETSc 和 petsc4py
- MPI 和 mpi4py
- NumPy
- Matplotlib，用于可视化输出

请在仓库根目录运行命令。

## 生成晶界场

```bash
mpirun -np 2 python generate_grain_boundaries.py
```

等效的包入口：

```bash
mpirun -np 2 python -m grain_generation.main
```

MPI 进程数可根据可用资源调整。该流程读取 `.msh` 文件，创建 eta 晶粒场，施加周期性约束，求解 Allen–Cahn 演化方程，计算派生晶界场，并输出 `.npz` 文件和预览文件。

## 运行 R2D 模拟

```bash
mpirun -np 2 python r2d.py \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_irregular_3_34.npz
```

等效的包入口：

```bash
mpirun -np 2 python -m r2d_simulation.main \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_irregular_3_34.npz
```

R2D 流程创建混合有限元状态，初始化物理场，组装耦合方程，施加边界条件和周期性约束，使用 Restricted Newton 求解器推进充放电阶段，并输出诊断信息和场数据。

## 仓库结构

```text
.
├── generate_grain_boundaries.py    晶粒生成入口
├── r2d.py                          R2D 模拟入口
├── grain_generation/               晶粒生成包
└── r2d_simulation/                 R2D 模拟包
```

### `grain_generation/`

| 模块 | 职责 |
|---|---|
| `main.py` | 命令行入口 |
| `workflow.py` | 晶粒生成流程编排 |
| `config.py` | 参数、区域常量和默认值 |
| `mesh.py` | 网格加载、标签和几何信息 |
| `initialization.py` | 晶粒和 eta 初始场 |
| `fields.py` | 有限元场和 eta 场操作 |
| `constraints.py` | 周期性约束 |
| `equations.py` | Allen–Cahn 弱形式和雅可比矩阵 |
| `solver.py` | Newton/PETSc 求解器配置 |
| `evolution.py` | 时间推进和收敛处理 |
| `derived.py` | 晶界指示函数和派生场 |
| `output.py` | PNG、SVG 和 NPZ 输出 |

### `r2d_simulation/`

| 模块 | 职责 |
|---|---|
| `main.py` | 命令行入口 |
| `workflow.py` | 顶层流程编排 |
| `config.py` | `Params` 和命令行参数覆盖 |
| `state.py` | 共享的 `R2DContext` 和状态容器 |
| `setup.py` | 网格、标量空间和混合空间构建 |
| `simulation_setup.py` | 将配置结果绑定到 `R2DContext` |
| `state_builder.py` | 混合状态、测试函数和试探函数视图 |
| `mesh.py` | 网格和区域处理 |
| `regions.py` | 子空间和自由度映射 |
| `fields.py` | 有限元场、晶界场和输出场 |
| `initialization.py` | 初始物理场 |
| `constitutive.py` | 材料本构关系 |
| `equations.py` | 辅助场、残差和雅可比矩阵 |
| `constraints.py` | 边界条件、周期性约束和非活跃自由度 |
| `solver.py` | Restricted Newton、PETSc 和性能分析 |
| `evolution.py` | 时间误差估计、自适应时间步和重试 |
| `evolution_loop.py` | 阶段和已接受时间步的调度 |
| `evolution_callbacks.py` | 单步诊断、输出和截止条件处理 |
| `cycling.py` | 充放电阶段构建 |
| `diagnostics.py` | 电压、荷电状态（SOC）、浓度和截止条件诊断 |
| `output.py` | PNG、XDMF、NPZ、CSV 和阶段输出 |
| `electrochemistry.py` | 电化学辅助计算 |

## 输出

典型输出包括晶界 `.npz` 场文件、PNG/SVG 预览图、R2D 最终状态 `.npz` 文件、XDMF/HDF5 场文件、诊断 CSV 文件、差分场、阶段快照，以及 Newton 求解和时间步自适应的性能分析记录。

输出目录和文件名由配置参数控制。

## 模块化架构

```text
main.py
  -> workflow.run()
       -> config / setup
       -> fields / initialization
       -> equations / constraints
       -> solver
       -> evolution_loop
            -> evolution_callbacks
            -> diagnostics
            -> output
```

`workflow.py` 连接各模块并构建运行时上下文。数值方程、材料关系、约束、求解器行为、时间积分、诊断和文件输出分别在对应模块中实现。R2D 共享对象通过 `R2DContext` 和 `EvolutionRuntimeContext` 传递。

## 开发与验证

```bash
python -m py_compile r2d.py r2d_simulation/*.py grain_generation/*.py
```

```bash
mpirun -np 2 python generate_grain_boundaries.py
mpirun -np 2 python r2d.py \
  --msh r2d_irregular_3_34.msh \
  --gb r2d_irregular_3_34.npz
```

回归测试时，请比较初始电势、周期性自由度数量、活跃及消除的自由度、Newton 迭代次数、时间步长、残差、运行时间和生成的输出文件。

## 注意事项

- `generate` 和 `r2d` 是独立的工作流程。
- `.msh` 文件中的物理区域标签必须与配置模块一致。
- `--msh` 和 `--gb` 接受绝对路径或相对于仓库根目录的路径。
- 请根据网格规模和可用内存选择 MPI 进程数。
- 默认参数值定义在对应的 `config.py` 文件中。

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 许可。
