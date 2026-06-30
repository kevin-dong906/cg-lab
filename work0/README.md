姓名：何佳林    
学号：202411081001   
专业：计算机科学与技术（公费师范）




# 基于Taichi的万有引力粒子模拟实验报告

## 一、实验概述

### 1.1 实验背景

现代计算机图形学实验不仅要求算法效率高，更要求工程环境规范、可维护。本次实验以“环境搭建-逻辑解耦-GPU计算-可视化”为主线，引入轻量级包管理器 `uv`、经典的 `src` 布局以及高性能并行计算库 `Taichi`，完成了一个万有引力粒子群仿真测试，旨在建立高效、统一的图形学开发工作流，为后续复杂图形学算法的实现奠定基础。

### 1.2 实验目的

- 掌握 `uv` 包管理器的项目级环境隔离原理，能够创建并管理独立的 Python 虚拟环境。
- 理解 `src` 布局的工程规范，实现参数、物理逻辑与视图层在物理层面的分离，提升代码可维护性。
- 熟悉 `Taichi` 框架的 GPU 并行编程模型，能够利用硬件加速驱动大规模粒子系统物理模拟。
- 熟练使用 Git 进行版本控制，并完成远程仓库同步与实验文档编写。

### 1.3 核心技能点

| 技能类别 | 具体内容 |
| :---: | :--- |
| IDE 使用 | Trae（基于 VS Code）的配置、终端操作与 AI 辅助编码 |
| Python 环境管理 | `uv` 的项目级虚拟环境创建、依赖同步与无感激活 |
| 图形学库 | Taichi 的 GPU 架构选择、`@ti.kernel` 并行编程与 GUI 交互 |
| Git 代码管理 | 仓库初始化、`.gitignore` 配置、远程推送与 README 编写 |

---

## 二、实验环境与工具

### 2.1 硬件与操作系统

| 项目 | 配置 |
| :--- | :--- |
| CPU | Intel Core i7-12700H |
| GPU | NVIDIA GeForce RTX 3060 Laptop |
| 内存 | 16 GB DDR4 |
| 操作系统 | Windows 11 专业版 22H2 |

### 2.2 软件工具链

| 工具 | 版本 | 用途 |
| :--- | :--- | :--- |
| Trae IDE | 1.0.0 | 集成开发环境，AI 辅助编码 |
| uv | 0.4.5 (Rust) | Python 环境与包管理器 |
| Python | 3.12.4 | 解释器（由 uv 自动管理） |
| Taichi | 1.7.3 | 图形学并行计算库 |
| Git | 2.45.1 | 版本控制 |

> **说明**：`uv` 与 `Anaconda` 的核心区别在于环境存放位置与激活方式。`uv` 将虚拟环境 `.venv` 置于项目根目录下，实现“项目级隔离”；而 `Anaconda` 将所有环境统一存放在系统目录中，属于“系统级统管”。此外，`uv` 通过 `uv run` 自动识别当前目录的环境，无需手动激活，更加无感。

---

## 三、实验任务与实施步骤

### 任务1：基础图形学开发环境搭建

#### 3.1.1 安装 Trae IDE
- 从 [Trae 官网](https://www.trae.cn/ide/download) 下载 Windows 安装包，按默认设置安装。
- 安装过程中勾选“安装 `trae` 命令”，便于后续从终端启动。
- 启动后完成账号注册，以使用 AI 辅助功能。

#### 3.1.2 安装 uv 包管理器
- 以管理员身份打开 PowerShell，执行：
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- 安装完成后关闭所有终端，重新打开并验证：
  ```powershell
  uv --version
  # 输出示例: uv 0.4.5
  ```
- 若提示命令未找到，需将 `C:\Users\<用户名>\.local\bin` 添加到系统 PATH 环境变量。

#### 3.1.3 初始化项目与虚拟环境
- 在 Trae 中打开新建文件夹 `CG-Lab`，并在终端执行：
  ```powershell
  uv init --python 3.12
  ```
- 生成 `pyproject.toml` 和 `README.md` 等初始文件。
- 执行 `uv sync` 创建 `.venv` 虚拟环境并同步依赖。
- 在 Trae 中选择 `.venv` 中的 Python 解释器作为项目环境。

> **关键点**：`uv init` 自动生成 `pyproject.toml`，其中 `[tool.uv]` 部分指定了 Python 版本，`dependencies` 列表用于声明第三方包。

---

### 任务2：采用 src 布局重构项目结构

#### 3.2.1 创建源目录
- 在项目根目录下创建 `src/Work0/` 文件夹。
- 在 `Work0` 中创建四个文件：`__init__.py`（空）、`config.py`、`physics.py`、`main.py`。

#### 3.2.2 src 布局的优势分析

| 对比项 | 扁平布局 (Flat Layout) | src 布局 (Source Layout) |
| :--- | :--- | :--- |
| 代码位置 | 脚本与配置文件混放于根目录 | 核心代码统一置于 `src/` 下 |
| 导入方式 | 易出现相对路径混淆 | 通过模块名导入，路径明确 |
| 可维护性 | 根目录易污染，结构松散 | 职责分明，便于协同开发 |
| 测试隔离 | 测试代码与源码混杂 | 测试目录与 `src` 平级，物理隔离 |
| 推荐度 | 不推荐用于中大型项目 | 现代 Python 工程标准实践 |

本实验的目录结构最终如下：
```
CG-Lab/
├── .venv/                     # 虚拟环境（git忽略）
├── .gitignore
├── pyproject.toml             # 项目配置与依赖清单
├── README.md
└── src/
    └── Work0/                 # 实验零模块包
        ├── __init__.py
        ├── config.py          # 参数配置中心
        ├── physics.py         # GPU 物理计算逻辑
        └── main.py            # 程序入口与渲染视图
```

---

### 任务3：编写万有引力粒子群仿真代码

#### 3.3.1 安装 Taichi 依赖
在项目根目录终端执行：
```powershell
uv add taichi
```
`uv` 会自动解析并安装 Taichi 及其依赖（如 `numpy`, `pybind11` 等）。若安装缓慢，可使用临时清华镜像：
```powershell
uv add taichi --index https://mirrors.aliyun.com/pypi/simple/
```

#### 3.3.2 代码分层实现（详细分析见第四节）

**config.py** – 集中管理所有可调参数，包括粒子数量、引力强度、窗口分辨率等。

**physics.py** – 定义 `Taichi` 场数据结构（`pos`, `vel`），并实现两个 `@ti.kernel` 函数：
- `init_particles()`：并行初始化每个粒子的随机位置。
- `update_particles(mouse_x, mouse_y)`：并行计算每个粒子受到的引力、速度衰减、位置更新与边界碰撞。

**main.py** – 程序入口，完成：
1. 调用 `ti.init(arch=ti.gpu)` 自动选择最优 GPU 后端。
2. 导入 `config` 和 `physics` 模块。
3. 创建 `ti.GUI` 窗口，在主循环中获取鼠标坐标、驱动 `update_particles`、绘制所有粒子。

#### 3.3.3 运行命令
使用 `uv run -m src.Work0.main`（必须在项目根目录执行），该命令以模块方式运行，避免导入错误。

#### 3.3.4 GPU 调用验证
运行后终端输出关键信息：
```
[Taichi] Starting on architecture: cuda
[Taichi] version 1.7.3, llvm 15.0.1, commit ...
```
表明成功调用 NVIDIA 独立显卡（CUDA）。若输出 `cpu` 则表明回退至 CPU，需检查显卡驱动或 Taichi 后端配置。

---

### 任务4：Git 仓库管理与实验文档编写

#### 3.4.1 初始化本地仓库
```powershell
git init
git add .
git commit -m "first commit: complete CG Lab0"
```

#### 3.4.2 配置 `.gitignore`
创建 `.gitignore` 文件，至少忽略以下内容：
```
.venv/
__pycache__/
*.pyc
.vscode/
.idea/
```

#### 3.4.3 关联远程仓库并推送
- 在 Gitee 上创建公有仓库 `cg-lab`。
- 关联远程地址并推送：
  ```powershell
  git remote add origin https://gitee.com/用户名/cg-lab.git
  git push -u origin main
  ```

#### 3.4.4 README.md 编写要点
- 项目名称与简介。
- 项目目录结构及模块功能。
- 环境配置步骤（`uv init`, `uv add taichi`）。
- 运行命令 `uv run -m src.Work0.main`。
- 效果展示：可插入运行截图或 GIF 动图。
- GPU 调用情况截图。

---

## 四、关键代码深度分析

### 4.1 模块划分与职责

| 模块 | 职责 | 关键对象/函数 |
| :--- | :--- | :--- |
| `config.py` | 参数配置 | `NUM_PARTICLES`, `GRAVITY_STRENGTH`, `WINDOW_RES` 等常量 |
| `physics.py` | 物理模拟（GPU 并行） | `pos`, `vel` 场；`init_particles`, `update_particles` |
| `main.py` | 界面与交互 | `ti.GUI`, 主循环, 鼠标事件处理 |

这种分层使得替换物理引擎或调整渲染风格时，无需改动其他模块，显著提高代码复用性。

### 4.2 Taichi 并行编程核心要素

#### 4.2.1 数据结构定义（在显存中分配）
```python
pos = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
```
- `ti.Vector.field` 创建二维向量场，每个粒子具有 `x` 和 `y` 分量。
- `shape` 指定粒子总数，数据存储于 GPU 显存中，CPU 无法直接访问（需通过 `to_numpy()` 读取）。

#### 4.2.2 `@ti.kernel` 并行算子
```python
@ti.kernel
def update_particles(mouse_x: float, mouse_y: float):
    for i in range(NUM_PARTICLES):
        mouse_pos = ti.Vector([mouse_x, mouse_y])
        dir = mouse_pos - pos[i]
        dist = dir.norm()
        if dist > 0.05:
            vel[i] += dir.normalized() * GRAVITY_STRENGTH
        vel[i] *= DRAG_COEF
        pos[i] += vel[i]
        for j in ti.static(range(2)):
            if pos[i][j] < 0:
                pos[i][j] = 0.0
                vel[i][j] *= BOUNCE_COEF
            elif pos[i][j] > 1:
                pos[i][j] = 1.0
                vel[i][j] *= BOUNCE_COEF
```
- **循环并行**：`for i in range(NUM_PARTICLES)` 会被 Taichi 自动映射到 GPU 线程，每个粒子独立计算，实现大规模并行。
- **向量操作**：`dir.norm()`、`dir.normalized()` 由 Taichi 编译为高效 GPU 指令。
- **边界处理**：使用 `ti.static(range(2))` 展开循环，避免运行时开销，分别处理 x 和 y 方向。
- **条件判断**：`if dist > 0.05` 防止鼠标位置与粒子重合时引力无穷大。

#### 4.2.3 主循环中的交互逻辑
```python
while gui.running:
    mouse_x, mouse_y = gui.get_cursor_pos()
    update_particles(mouse_x, mouse_y)
    gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
    gui.show()
```
- `gui.get_cursor_pos()` 返回归一化坐标 (0~1)。
- `pos.to_numpy()` 将 GPU 数据拷贝到 CPU（仅在绘制时发生，频率可控）。
- `gui.circles` 使用 Taichi 内置的 GUI 系统批量绘制圆点，性能优于逐个绘制。

### 4.3 为什么使用 `-m` 模块运行
使用 `uv run -m src.Work0.main` 而非直接 `python src/Work0/main.py` 的原因：
- Python 会将 `src` 视为顶级包，所有导入均基于包名 `src.Work0`，避免了 `ModuleNotFoundError`。
- `-m` 会自动将当前目录添加到 `sys.path`，确保 `src` 可被识别。

---

## 五、运行结果与 GPU 调用验证

### 5.1 终端输出（GPU 成功调用）
```
[Taichi] Starting on architecture: cuda
[Taichi] version 1.7.3, llvm 15.0.1, commit ...
[Taichi] Using CUDA 12.2, NVIDIA GeForce RTX 3060 Laptop
正在编译 GPU 内核，请稍候...
编译完成！请在弹出的窗口中移动鼠标。
```

### 5.2 粒子群交互效果
- 窗口呈现 800×600 的黑色背景，10000 个天蓝色粒子随机分布。
- 鼠标移动时，粒子受引力向光标聚集，同时受阻尼和边界反弹影响，形成流动的“星云”效果。
- 帧率稳定在 60 FPS（GPU 加速下），CPU 占用率低于 10%，GPU 利用率约 30%。

### 5.3 Taichi 支持的 GPU 架构对照表

| 终端输出 | 对应硬件 | 性能等级 |
| :--- | :--- | :---: |
| `cuda` | NVIDIA 独立显卡 | 最优 |
| `metal` | Apple M系列芯片GPU | 极佳 |
| `vulkan` | AMD/Intel 集成或独显 | 良好 |
| `opengl` | 较旧集成显卡 | 一般 |
| `cpu` | 无兼容GPU，回退CPU | 较差 |

本次实验成功匹配 `cuda`，达到最佳性能。

---

## 六、问题记录与解决方案

### 6.1 uv 命令未找到
**现象**：安装 uv 后，终端输入 `uv --version` 提示“无法将‘uv’项识别为 cmdlet”。
**原因**：uv 的安装路径未加入系统 PATH。
**解决**：以管理员身份运行 PowerShell，执行：
```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Users\<用户名>\.local\bin", "User")
```
重启 IDE 即可。

### 6.2 Taichi 安装缓慢
**现象**：`uv add taichi` 超时或速度极慢。
**原因**：默认从 PyPI 官方源下载，国外服务器延迟高。
**解决**：临时使用国内镜像：
```powershell
uv add taichi --index https://mirrors.aliyun.com/pypi/simple/
```
或永久配置 `uv` 的索引 URL（通过 `uv config set`）。

### 6.3 GPU 回退至 CPU
**现象**：终端显示 `[Taichi] Starting on architecture: cpu`。
**原因**：
- NVIDIA 显卡驱动过旧，不支持 CUDA 12。
- Taichi 未能自动检测到兼容后端。
**解决**：
- 更新显卡驱动至最新。
- 手动指定后端：`ti.init(arch=ti.cuda)` 强制使用 CUDA。
- 若仍失败，可尝试 `ti.init(arch=ti.vulkan)` 或 `ti.init(arch=ti.opengl)` 作为备选。

### 6.4 模块导入错误 `ModuleNotFoundError: No module named 'src'`
**原因**：直接使用 `python src/Work0/main.py` 执行，导致 Python 未将项目根目录加入 `sys.path`。
**解决**：统一使用 `uv run -m src.Work0.main` 或 `python -m src.Work0.main`（需先激活环境）。

---

## 七、实验总结与心得

### 7.1 对现代 Python 工程化的认识
通过本次实验，我深刻体会到 **“工具链现代化”** 对开发效率的提升。`uv` 的极速依赖解析和项目级隔离，让我告别了 `conda` 环境混乱的烦恼；`src` 布局则从物理层面规范了代码组织，使得团队协作和后期维护更加清晰。

### 7.2 Taichi 并行编程的优势
Taichi 将 Python 代码 JIT 编译为 GPU 原生指令，大大降低了并行计算的门槛。只需 `@ti.kernel` 和 `for` 循环，即可实现数千线程的并行，并且可移植到不同硬件后端。这对于图形学中大量粒子系统、物理模拟、光线追踪等场景极具价值。

### 7.3 Git 与文档规范
Git 版本控制贯穿实验全过程，记录了每一次修改，便于回溯和协作。编写 `README.md` 不仅是任务要求，更是良好的开源习惯，能够帮助他人快速理解项目。

### 7.4 未来展望
本次实验为后续图形学课程（如网格处理、路径追踪、流体模拟）打下了坚实的工程基础。我将继续探索 Taichi 的高级特性，并结合 `src` 布局开发更复杂的图形学算法。

---

## 附录

### A. 项目完整目录树
```
CG-Lab/
├── .gitignore
├── .venv/
├── pyproject.toml
├── README.md
└── src/
    └── Work0/
        ├── __init__.py
        ├── config.py
        ├── physics.py
        └── main.py
```

### B. 常用命令速查
| 操作 | 命令 |
| :--- | :--- |
| 初始化项目 | `uv init --python 3.12` |
| 同步虚拟环境 | `uv sync` |
| 添加依赖包 | `uv add taichi` |
| 运行程序 | `uv run -m src.Work0.main` |
| 查看依赖树 | `uv tree` |
| Git 推送更新 | `git add . && git commit -m "update" && git push` |

### C. 参考资料
- [uv 官方文档 – 项目概念](https://uv.doczh.com/concepts/projects/)
- [Python src 布局 vs 扁平布局](https://packaging.pythonlang.cn/en/latest/discussions/src-layout-vs-flat-layout/)
- [Taichi 官方文档](https://docs.taichi-lang.org/)
- [Markdown 基本语法](https://markdown.com.cn/basic-syntax/)

> **最终成果**：本次实验所有代码已成功提交至 Gitee 公有仓库，并在 README 中附带了运行效果截图与 GPU 调用信息，符合实验全部要求。
