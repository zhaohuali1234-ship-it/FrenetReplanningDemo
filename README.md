# Frenet 轨迹规划、跟踪与安全停车仿真

这是一个基于 [PythonRobotics 的 Frenet Optimal Trajectory 示例](https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/FrenetOptimalTrajectory) 的学习与仿真扩展。原项目提供 Frenet 候选轨迹规划、样条路径和坐标转换；本项目在其基础上加入车辆模型跟踪、连续重规划、前轮转角及转向速度约束、圆形障碍物检查、规划结果延迟交接与无解时的安全停车。它只用于离散仿真和算法讨论，**不能控制真实车辆**。

## 运行

建议使用 Python 3.13。进入本仓库根目录，安装依赖后运行：

```bash
python -m pip install -r requirements.txt
python run_demo.py --case H01 --delay-ms 300 --expect-complete
```

上述命令在无图形界面模式运行稳定版本，并模拟规划结果固定延迟 300 ms 后交付。程序会报告是否抵达终点、圆形碰撞模型的最小净间距和结束原因。要看动画，去掉 `--expect-complete` 并添加 `--show`：

```bash
python run_demo.py --case H01 --delay-ms 300 --show
```

`--case` 可选 `base`、`H01`、`H05`，分别使用原障碍布局、将全部障碍平移 `(-4, 0) m`、平移 `(-3, 1) m`。`--delay-ms 0` 是无交付延迟演示；300 ms 是固定延迟实验，不代表每轮计算都能在 300 ms 内完成。首轮规划在起步前预装载。

## 目前的结果与限制

稳定版在固定 300 ms 延迟的 H01–H06 六场景回放中，H01、H03、H04 完成，其余三场景安全停车；这表示安全停车与完成通行是两个不同结果。新布局和不同初速下，也存在较长轨迹在延迟后无法通过交接跟踪预演、旧轨迹到期后停车的情况。本项目没有验证实时调度、动态障碍、轮胎动力学或矩形车身；仿真安全检查不构成实车安全保证。

`PathPlanning/FrenetOptimalTrajectory/frenet_tracking_oos_core.py` 是本项目的稳定实验实现。此前试验过的绕障门控和短前视控制器未稳定改善新场景，因此没有并入发布候选。

## 来源与许可

`PathPlanning/FrenetOptimalTrajectory/frenet_optimal_trajectory.py`、`cartesian_frenet_converter.py`、`PathPlanning/QuinticPolynomialsPlanner/quintic_polynomials_planner.py`、`PathPlanning/CubicSpline/cubic_spline_planner.py` 来自 Atsushi Sakai 与 PythonRobotics 贡献者。其原作者信息保留在源码中；仓库根目录的 [LICENSE](LICENSE) 是 PythonRobotics 的 MIT 许可证。其余仿真扩展代码基于该项目开发。请在介绍本项目时注明它是复现与扩展，而非对原始 Frenet 算法的从零原创声明。
