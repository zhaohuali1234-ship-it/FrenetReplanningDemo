"""按交接所需时域生成短候选，并在旧轨迹到期前加密重规划。

放在 frenet_optimal_trajectory.py 同目录运行；不修改原版文件。
本例沿用原版静态点障碍物和圆形碰撞模型。
"""
import sys
import time as clock
from pathlib import Path
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

FRENET_DIR = Path(__file__).resolve().parent
if not (FRENET_DIR / "frenet_optimal_trajectory.py").is_file():
    raise FileNotFoundError(
        "Place frenet_tracking_oos_core.py beside frenet_optimal_trajectory.py"
    )
sys.path.insert(0, str(FRENET_DIR))
import frenet_optimal_trajectory as planner

CONVERTER = planner.CartesianFrenetConverter
CONTROL_DT = 0.02
REPLAN_DT = 0.2
MIN_PLAN_HORIZON = 1.2  # 不排除临近障碍物时唯一可执行的短路径
URGENT_REPLAN_REMAINING_S = 1.2
MIN_PLAN_HORIZON_ACTIVE = MIN_PLAN_HORIZON
HANDOFF_PREVIEW_S = 0.4
HANDOFF_STEER_RESERVE = np.deg2rad(0.5)
TOTAL_TIME = 45.0  # 为低速转向保留观察时间；批量结果同时记录第 30 秒状态
GOAL_POSITION_TOLERANCE = 0.3
GOAL_SPEED_TOLERANCE = 0.05
GOAL_APPROACH_DISTANCE = 30.0  # 提前尝试同时完成横向归位和减速停车
WHEELBASE = 2.5
MAX_STEER = np.deg2rad(35.0)
# 教学用前轮转角变化速度；不是方向盘角速度，也不是实车标定值。
MAX_STEER_RATE = np.deg2rad(30.0)
PLAN_STEER_RATE = np.deg2rad(25.0)  # 留 5°/s 给控制修正
PLAN_SAMPLE_DT = 0.05
MAX_ACCEL = 4.0
PLAN_STEER = np.deg2rad(32.0)  # 给跟踪修正留 3° 余量
PLAN_ACCEL = 3.5  # 给速度跟踪修正留 0.5 m/s² 余量
MAX_LATERAL_ACCEL = 3.0
SAFETY_MARGIN = 0.3
TRACKING_ERROR_LIMIT = 0.1
PLAN_CLEARANCE = SAFETY_MARGIN + TRACKING_ERROR_LIMIT
# 备份策略与 BRAKE 分支一致：立即最大减速，保持当前实际前轮转角。
# 此检查针对本例静态障碍物、无制动延迟的圆形车辆模型。
STOP_GUARD_MARGIN = SAFETY_MARGIN + 0.05
STOP_CHECK_DT = CONTROL_DT  # 对下一次重规划之前的执行前缀逐步检查
PREDICTION_TIMES = (2.0, 3.0, 4.0, 4.8)
END_SPEEDS = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]) / 3.6
RECOVERY_TARGET_SPEED = 2.5  # m/s；局部无解后降低速度目标
RECOVERY_DECEL = 2.0
RECOVERY_MIN_SPEED = 0.35
RECOVERY_MAX_DURATION = 12.0
RECOVERY_EXIT_PROGRESS = 10.0
RECOVERY_TRIGGER_CLEARANCE = 6.0  # 当前备份停车余量收紧时提前减速
RECOVERY_SAMPLING = False
RECOVERY_LATERAL_TARGET = 0.0
SPEED_PROFILE = None
PREVIOUS_GEOMETRY_ENDPOINT = None
USE_SPEED_GUIDANCE = False
INITIAL_PREVIEW_STRESS = 0.0
GUIDANCE_STRESS_THRESHOLD = 1.5
GUIDANCE_PREVIEW_DISTANCE = 20.0
PROFILE_DECEL = 1.5  # 预瞄纵向减速的建议值；后备制动仍使用 MAX_ACCEL
PROFILE_LATERAL_ACCEL = 2.4
PROFILE_RATE_RESERVE = 0.8


class CurveSpeedProfile:
    """参考线上的速度建议；候选与实际车辆仍单独检查物理约束。

    v_lat = sqrt(a_lat/|kappa|)，v_rate = delta_rate/|d(delta)/ds|。
    从前往后查看未来弯道，再向起点回传可减速到该速度的包络。
    初始实际车速不被改写，超出建议值时通过有限加速度逐步减速。
    """
    def __init__(self, csp):
        self.s = np.linspace(0., csp.s[-1]-1e-6, int(np.ceil(csp.s[-1]/.1))+1)
        self.kappa = np.array([csp.calc_curvature(float(v)) for v in self.s])
        rate = np.array([csp.calc_curvature_rate(float(v)) for v in self.s])
        self.steering_gradient = WHEELBASE*rate/(1.+(WHEELBASE*self.kappa)**2)
        lateral = np.sqrt(PROFILE_LATERAL_ACCEL/np.maximum(abs(self.kappa), 1e-8))
        steering_rate = (PROFILE_RATE_RESERVE*PLAN_STEER_RATE
                         /np.maximum(abs(self.steering_gradient), 1e-8))
        self.raw = np.minimum.reduce((np.full_like(self.s, planner.TARGET_SPEED),
                                      lateral, steering_rate))
        self.speed = self.raw.copy()
        for i in range(len(self.s)-2, -1, -1):
            braking_bound = np.sqrt(self.speed[i+1]**2+2*PROFILE_DECEL*(self.s[i+1]-self.s[i]))
            self.speed[i] = min(self.speed[i], braking_bound)

    def at(self, s):
        return np.interp(s, self.s, self.speed)


def profile_cost(path):
    if not USE_SPEED_GUIDANCE or SPEED_PROFILE is None or len(path.s) != len(path.t):
        return 0.0
    excess = np.maximum(np.asarray(path.s_d)-SPEED_PROFILE.at(path.s), 0.)
    return float(8.*np.trapezoid(excess**2, np.asarray(path.t)))


def active_speed_cap(path, age):
    if getattr(path, 'retimed', False):
        return float(np.interp(age, path.t, path.body_speed_cap))
    s = np.interp(age, path.t, path.s)
    longitudinal = max(.2, float(np.interp(age, path.t, path.s_d)))
    actual = float(np.interp(age, path.t, path.v))
    return float(SPEED_PROFILE.at(s)*actual/longitudinal)


def spatial_retimed_sampling(csp, sc, dc, grid_s, grid_xy):
    """先生成空间曲线，再按曲率/转角梯度配置车速；不重置初始车速。

    时间横向多项式在纵向速度趋近零时容易产生很大的空间导数。
    空间曲线保留真实起点的 d、d'、d''，速度配置独立完成。
    """
    counts = {k: 0 for k in ('invalid', 'speed', 'acceleration', 'steering',
        'steering_rate', 'lateral_acceleration', 'clearance', 'stopping', 'ok')}
    reference = (sc[0], *csp.calc_position(sc[0]), csp.calc_yaw(sc[0]),
                 csp.calc_curvature(sc[0]), csp.calc_curvature_rate(sc[0]))
    initial = CONVERTER.frenet_to_cartesian(*reference, sc, dc)
    initial_speed = float(initial[4])
    available = csp.s[-1]-sc[0]-1e-5
    previous_length = (PREVIOUS_GEOMETRY_ENDPOINT[0]-sc[0]
                       if PREVIOUS_GEOMETRY_ENDPOINT is not None else 0.)
    lengths = np.unique(np.minimum([8., 12., 18., 25., 35., previous_length], available))
    targets = np.unique(np.clip(np.r_[np.arange(-5., 3.01, 1.), dc[0],
        RECOVERY_LATERAL_TARGET if RECOVERY_SAMPLING else 0.,
        PREVIOUS_GEOMETRY_ENDPOINT[1] if PREVIOUS_GEOMETRY_ENDPOINT is not None else dc[0]],
        -planner.MAX_ROAD_WIDTH, planner.MAX_ROAD_WIDTH))
    candidates = []
    for length in lengths:
        if length <= 1.0:
            continue
        progress = np.linspace(0., length, int(np.ceil(length/.1))+1)
        s_values = sc[0]+progress
        references = reference_samples(csp, s_values.tolist())
        for target in targets:
            lateral = planner.QuinticPolynomial(*dc, float(target), 0., 0., length)
            d = np.asarray(lateral.calc_point(progress))
            ds = np.asarray(lateral.calc_first_derivative(progress))
            dss = np.asarray(lateral.calc_second_derivative(progress))
            if max(abs(d)) > planner.MAX_ROAD_WIDTH:
                counts['invalid'] += 1
                continue
            if min(1.-references[3]*d) <= .2:
                counts['invalid'] += 1
                continue
            geometric = planner.FrenetPath()
            geometric.d, geometric.d_d, geometric.d_dd = d.tolist(), ds.tolist(), dss.tolist()
            geometric.s_d = [1.]*len(d)
            geometric.s_dd = [0.]*len(d)
            cartesian_from_cached_reference(geometric, references)
            cart = np.column_stack((geometric.x, geometric.y, geometric.yaw,
                                    geometric.c, geometric.v, geometric.a))
            if not np.isfinite(cart).all() or min(cart[:, 4]) <= .1:
                counts['invalid'] += 1
                continue
            x, y, yaw, curvature, metric, _ = cart.T
            deltas = np.arctan(WHEELBASE*curvature)
            if max(abs(deltas)) > PLAN_STEER+1e-6:
                counts['steering'] += 1
                continue
            if trajectory_clearance(x, y) < PLAN_CLEARANCE:
                counts['clearance'] += 1
                continue
            segment = np.hypot(np.diff(x), np.diff(y))
            if min(segment) <= 1e-8:
                counts['invalid'] += 1
                continue
            nominal = min(planner.MAX_SPEED, max(2., planner.TARGET_SPEED))
            travelled = np.r_[0., np.cumsum(segment)]
            # 标称速度是建议值，初速超过建议时允许按有限减速度收敛；
            # 曲率和转向速度的物理上限仍硬性筛选，不裁剪初始实际状态。
            nominal_envelope = np.sqrt(np.maximum(nominal**2,
                initial_speed**2-2*PROFILE_DECEL*travelled))
            raw = np.minimum(nominal_envelope,
                             np.sqrt(PROFILE_LATERAL_ACCEL/np.maximum(abs(curvature), 1e-8)))
            # PLAN_STEER_RATE 已低于实际执行器上限；候选仍逐段复核
            # 同一规划上限，避免在重规划时重复压缩可行速度。
            segment_rate_cap = PLAN_STEER_RATE*segment/np.maximum(abs(np.diff(deltas)), 1e-8)
            raw[:-1] = np.minimum(raw[:-1], segment_rate_cap)
            raw[1:] = np.minimum(raw[1:], segment_rate_cap)
            speed = raw.copy()
            for i in range(len(speed)-2, -1, -1):
                speed[i] = min(speed[i], np.sqrt(speed[i+1]**2+2*PLAN_ACCEL*segment[i]))
            if initial_speed > speed[0]+1e-6:
                counts['speed'] += 1  # 不能把实际初速直接裁剪到包络内
                continue
            speed[0] = initial_speed
            for i in range(len(speed)-1):
                speed[i+1] = min(speed[i+1], np.sqrt(speed[i]**2+2*2.0*segment[i]))
            if min(speed) <= .05:
                counts['invalid'] += 1
                continue
            time = np.r_[0., np.cumsum(2*segment/(speed[:-1]+speed[1:]))]
            if time[-1] < MIN_PLAN_HORIZON_ACTIVE:
                counts['invalid'] += 1
                continue
            a_segments = (speed[1:]**2-speed[:-1]**2)/(2*segment)
            a = np.r_[a_segments, a_segments[-1]]
            path = planner.FrenetPath()
            path.t, path.x, path.y, path.yaw = [list(v) for v in (time, x, y, yaw)]
            path.c, path.v, path.a = [list(v) for v in (curvature, speed, a)]
            path.s, path.d, path.d_d, path.d_dd = [list(v) for v in (s_values, d, ds, dss)]
            path.s_d = list(speed/metric)
            path.s_dd = list(np.gradient(path.s_d, time))
            path.s_ddd = list(np.gradient(path.s_dd, time))
            path.d_ddd = list(np.gradient(np.gradient(np.gradient(d, time), time), time))
            if max(abs(a)) > PLAN_ACCEL+1e-6:
                counts['acceleration'] += 1
                continue
            if path_steering_rate(path) > PLAN_STEER_RATE+1e-6:
                counts['steering_rate'] += 1
                continue
            if max(speed**2*abs(curvature)) > MAX_LATERAL_ACCEL+1e-6:
                counts['lateral_acceleration'] += 1
                continue
            if not candidate_can_stop(path):
                counts['stopping'] += 1
                continue
            path.cf = float(.1*time[-1]+.5*lateral_destination_cost(float(target))
                            +2./length+recovery_risk_cost(path))
            path.model_end_state = False
            path.rollout = False
            path.retimed = True
            path.body_speed_cap = list(raw)
            candidates.append(path)
            counts['ok'] += 1
    # 当前反馈按最近参考点转换；避免选择另一条法线分支上的空间曲线。
    for path in sorted(candidates, key=lambda p: p.cf):
        consistent = True
        indices = sorted(set(range(0, len(path.s), 10)) | {len(path.s)-1})
        for i in indices:
            projection = reference_projection(csp, path.x[i], path.y[i], grid_s, grid_xy)
            if (abs(projection[0]-path.s[i]) > .5
                    or np.cos(path.yaw[i]-projection[3]) <= .2):
                consistent = False
                break
        if consistent:
            return path, counts
        counts['ok'] -= 1
        counts['invalid'] += 1
    return None, counts


def timed_curve_tracking(path, age, state):
    """配置了时间的空间路径：曲率前馈 + 航向/横向误差修正。

    使用当前时刻的规划曲率，避免前视追踪提前改变初始曲率，
    从而破坏下一轮空间曲线的曲率起点。转向执行器仍限幅限速。
    """
    point = np.array([np.interp(age, path.t, path.x), np.interp(age, path.t, path.y)])
    yaw = float(np.interp(age, path.t, np.unwrap(path.yaw)))
    # 执行器在本步从当前转角推进到步末转角；步末曲率作前馈，
    # 与 advance 使用步内中间转角的积分方式一致。
    curvature = float(np.interp(age+CONTROL_DT, path.t, path.c))
    heading_error = (yaw-state[2]+np.pi) % (2*np.pi)-np.pi
    lateral_error = float(np.dot(point-state[:2], [-np.sin(yaw), np.cos(yaw)]))
    requested = float(np.arctan(WHEELBASE*curvature)+heading_error
                      +np.arctan2(lateral_error, max(state[3], .5)))
    return float(np.clip(requested, -MAX_STEER, MAX_STEER)), requested


def speed_aware_planning(csp, sc, dc, grid_s, grid_xy, recovery=False):
    global END_SPEEDS
    if not USE_SPEED_GUIDANCE:
        if recovery:
            return recovery_planning(csp, sc, dc, grid_s, grid_xy)
        return constrained_planning(csp, sc, dc, grid_s, grid_xy)
    saved = END_SPEEDS, planner.TARGET_SPEED
    try:
        limit = float(SPEED_PROFILE.at(sc[0]))
        future_s = np.clip(sc[0]+np.array([2., 3., 4.8])*max(sc[1], .4), 0., csp.s[-1]-1e-6)
        future = SPEED_PROFILE.at(future_s)
        END_SPEEDS = np.unique(np.clip(np.r_[.5*limit, .75*limit, limit, future,
                                            max(.4, sc[1]-.5), max(.4, sc[1]-1.)], .3, planner.MAX_SPEED))
        planner.TARGET_SPEED = limit
        if recovery:
            return recovery_planning(csp, sc, dc, grid_s, grid_xy)
        geometric, counts = spatial_retimed_sampling(csp, sc, dc, grid_s, grid_xy)
        if geometric is not None:
            return geometric, counts
        path, extra = constrained_planning(csp, sc, dc, grid_s, grid_xy)
        return path, {k: counts[k]+extra[k] for k in counts}
    finally:
        END_SPEEDS, planner.TARGET_SPEED = saved


def terminal_offsets(sc, dc):
    offsets = planner.LONGITUDINAL_MOVEMENT_STRATEGY.get_d_arrange(sc[0])
    if RECOVERY_SAMPLING:
        offsets = np.unique(np.clip(np.r_[offsets, dc[0]+np.arange(-1., 1.01, .25),
                                          RECOVERY_LATERAL_TARGET+np.arange(-.5, .51, .25)],
                                    -planner.MAX_ROAD_WIDTH, planner.MAX_ROAD_WIDTH))
    return offsets


def recovery_risk_cost(path):
    """低速恢复优先选择较宽的绕障余量，避免持续贴着最低阈值前进。"""
    if not RECOVERY_SAMPLING:
        return 0.0
    gap = trajectory_clearance(path.x, path.y)
    return 4.0/(max(0.0, gap-SAFETY_MARGIN)+0.5)**2


def lateral_destination_cost(d):
    target = RECOVERY_LATERAL_TARGET if RECOVERY_SAMPLING else 0.0
    return (8.0 if RECOVERY_SAMPLING else 1.0)*(d-target)**2


def choose_recovery_offset(csp, sc, dc):
    """用前方参考线偏移曲线选择绕障方向；它只引导采样，不直接执行。

    所有实际候选仍由车辆模型或多项式生成，并通过完整约束检查。
    """
    depth = min(25.0, csp.s[-1]-sc[0]-1e-5)
    if depth <= 0.1:
        return 0.0
    progress = np.linspace(0., depth, int(np.ceil(depth/.5))+1)
    samples = sc[0]+progress
    reference = np.array([csp.calc_position(float(v)) for v in samples])
    yaws = np.array([csp.calc_yaw(float(v)) for v in samples])
    normal = np.column_stack((-np.sin(yaws), np.cos(yaws)))
    u = np.clip(progress/min(8., depth), 0., 1.)
    blend = 10*u**3-15*u**4+6*u**5
    best, best_score = 0., float('inf')
    for offset in np.arange(-6., 6.01, .5):
        lateral = dc[0]+blend*(offset-dc[0])
        xy = reference + lateral[:, None]*normal
        gap = trajectory_clearance(xy[:, 0], xy[:, 1])
        score = (0.1*(offset-dc[0])**2+0.05*offset**2
                 +4./(max(0., gap-SAFETY_MARGIN)+.5)**2
                 +100.*max(0., PLAN_CLEARANCE-gap)**2)
        if score < best_score:
            best, best_score = float(offset), score
    return best


class ArcLengthReference:
    """保持原参考线几何形状，数值积分弧长后反查原样条参数。

    Frenet 转换公式需要以弧长为参考线参数。原版样条参数使用
    路点之间的累计直线距离，不能将其导数直接视为沿曲线的速度。
    """
    def __init__(self, raw_reference):
        self.raw = raw_reference
        end = np.nextafter(float(raw_reference.s[-1]), 0.0)
        parameter = np.linspace(0.0, end, int(np.ceil(end/0.005)) + 1)
        dx = np.array([self.raw.sx.calc_first_derivative(float(u))
                       for u in parameter])
        dy = np.array([self.raw.sy.calc_first_derivative(float(u))
                       for u in parameter])
        derivative_norm = np.hypot(dx, dy)
        if np.min(derivative_norm) <= 1e-8:
            raise RuntimeError("参考线存在退化点，不能建立弧长映射")
        arc = cumulative_trapezoid(derivative_norm, parameter, initial=0.0)
        self.parameter_from_arc = PchipInterpolator(arc, parameter, extrapolate=False)
        self.s = np.array([0.0, arc[-1]])
        # Read-only spline coefficients for batched reference evaluation.
        sx, sy = self.raw.sx, self.raw.sy
        self.spline_knots = np.asarray(sx.x)
        self.spline_x = tuple(np.asarray(v) for v in (sx.a, sx.b, sx.c, sx.d))
        self.spline_y = tuple(np.asarray(v) for v in (sy.a, sy.b, sy.c, sy.d))

    @lru_cache(maxsize=8192)
    def parameter(self, s):
        if s < 0.0 or s > self.s[-1]:
            return None
        return float(self.parameter_from_arc(s))

    @lru_cache(maxsize=8192)
    def calc_position(self, s):
        parameter = self.parameter(s)
        if parameter is None:
            return None, None
        return self.raw.calc_position(parameter)

    @lru_cache(maxsize=8192)
    def calc_yaw(self, s):
        parameter = self.parameter(s)
        return None if parameter is None else self.raw.calc_yaw(parameter)

    @lru_cache(maxsize=8192)
    def calc_curvature(self, s):
        parameter = self.parameter(s)
        return None if parameter is None else self.raw.calc_curvature(parameter)

    @lru_cache(maxsize=8192)
    def calc_curvature_rate(self, s):
        parameter = self.parameter(s)
        # 原版该方法的分母是 (x'²+y'²)³，已经计算了 dκ/d弧长。
        # 因此直接复用其数值，不再重复除以参数导数的长度。
        return None if parameter is None else self.raw.calc_curvature_rate(parameter)


def trajectory_clearance(px, py):
    """Compute every obstacle-to-segment gap in one array operation.

    The same point/segment geometry and thresholds as polyline_distance are
    retained; only the independent obstacle calculations are batched.
    """
    obstacles = np.asarray(planner.OBSTACLES, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if len(px) == 1:
        distances2 = (obstacles[:, 0]-px[0])**2 + (obstacles[:, 1]-py[0])**2
        return float(np.sqrt(np.min(distances2))-planner.ROBOT_RADIUS)
    starts = np.column_stack((px[:-1], py[:-1]))
    vectors = np.column_stack((np.diff(px), np.diff(py)))
    lengths2 = np.maximum(np.sum(vectors**2, axis=1), 1e-12)
    offsets = obstacles[:, None, :] - starts[None, :, :]
    ratio = np.clip(np.sum(offsets*vectors[None, :, :], axis=2)
                    / lengths2[None, :], 0.0, 1.0)
    residual = offsets-ratio[:, :, None]*vectors[None, :, :]
    distances2 = np.sum(residual**2, axis=2)
    return float(np.sqrt(np.min(distances2))-planner.ROBOT_RADIUS)


def path_steering_rate(path):
    """相邻加密采样点的平均前轮转角变化速度，单位 rad/s。"""
    delta = np.arctan(WHEELBASE*np.asarray(path.c))
    dt = np.diff(np.asarray(path.t))
    if len(delta) < 2 or np.any(dt <= 0.0):
        return float("inf")
    return float(np.max(np.abs(np.diff(delta)/dt)))


def steering_actuator(actual, command, dt):
    """转向执行器：先限幅，再按最大变化速度推进实际前轮转角。"""
    command = float(np.clip(command, -MAX_STEER, MAX_STEER))
    change = float(np.clip(command-actual, -MAX_STEER_RATE*dt,
                           MAX_STEER_RATE*dt))
    return float(np.clip(actual+change, -MAX_STEER, MAX_STEER))


def lateral_without_deepcopy(longitudinal, terminal_d, dc, duration,
                             time_samples=None):
    """构造独立候选，复用只读纵向采样，避免复制整条纵向数组。"""
    path = planner.FrenetPath()
    path.t = longitudinal.t
    path.s = longitudinal.s
    path.s_d = longitudinal.s_d
    path.s_dd = longitudinal.s_dd
    path.s_ddd = longitudinal.s_ddd
    s0_d = longitudinal.s_d[0]
    s0_dd = longitudinal.s_dd[0]
    c_d, c_d_d, c_d_dd = dc
    if time_samples is None:
        lateral = planner.QuinticPolynomial(
            c_d, c_d_d*s0_d, c_d_dd*s0_d**2+c_d_d*s0_dd,
            terminal_d, 0.0, 0.0, duration)
        t = np.asarray(longitudinal.t)
        time_samples = (lateral.calc_point(t),
                        lateral.calc_first_derivative(t),
                        lateral.calc_second_derivative(t),
                        lateral.calc_third_derivative(t))
    d, d_t, d_tt, d_ttt = time_samples
    t = np.asarray(longitudinal.t)
    s_d = np.asarray(longitudinal.s_d)
    s_dd = np.asarray(longitudinal.s_dd)
    inverse = 1.0/(s_d+1e-6)+1e-6
    d_s = d_t*inverse
    path.d = d.tolist()
    path.d_d = d_s.tolist()
    path.d_dd = ((d_tt-d_s*s_dd)*inverse**2).tolist()
    path.d_ddd = d_ttt.tolist()
    return path


def reference_samples(csp, s_values):
    """同一纵向采样的道路数据在不同横向候选之间共用。"""
    arc = np.asarray(s_values, dtype=float)
    if not len(arc):
        return np.empty((5, 0))
    valid = np.isfinite(arc) & (arc >= 0.) & (arc <= csp.s[-1])
    if not np.all(valid):
        arc = arc[:int(np.argmax(~valid))]
    if not len(arc):
        return np.empty((5, 0))
    parameter = np.asarray(csp.parameter_from_arc(arc))
    index = np.searchsorted(csp.spline_knots, parameter, side="right")-1
    offset = parameter-csp.spline_knots[index]

    def evaluate(coefficients):
        a, b, c, d = (v[index] for v in coefficients)
        position = a+b*offset+c*offset**2.0+d*offset**3.0
        first = b+2.0*c*offset+3.0*d*offset**2.0
        second = 2.0*c+6.0*d*offset
        third = 6.0*d
        return position, first, second, third

    x, dx, ddx, dddx = evaluate(csp.spline_x)
    y, dy, ddy, dddy = evaluate(csp.spline_y)
    yaw = np.arctan2(dy, dx)
    square = dx*dx+dy*dy
    curvature = (ddy*dx-ddx*dy)/square**(3/2)
    a = dx*ddy-dy*ddx
    b = dx*dddy-dy*dddx
    c = dx*ddx+dy*ddy
    rate = (b*square-3.0*a*c)/(square*square*square)
    return np.vstack((x, y, yaw, curvature, rate))


def cartesian_from_cached_reference(path, samples):
    """与原版 Frenet 转换等式一致，批量计算候选的全局坐标。"""
    count = samples.shape[1]
    if count == 0:
        return path
    rx, ry, reference_yaw, reference_curvature, curvature_gradient = samples
    d = np.asarray(path.d[:count])
    d_s = np.asarray(path.d_d[:count])
    d_ss = np.asarray(path.d_dd[:count])
    s_d = np.asarray(path.s_d[:count])
    s_dd = np.asarray(path.s_dd[:count])
    cosine = np.cos(reference_yaw)
    sine = np.sin(reference_yaw)
    x = rx - sine*d
    y = ry + cosine*d
    factor = 1.0-reference_curvature*d
    angle = np.arctan2(d_s, factor)
    cosine_angle = np.cos(angle)
    yaw = (angle+reference_yaw+np.pi) % (2*np.pi)-np.pi
    curvature_d = curvature_gradient*d+reference_curvature*d_s
    curvature = (((d_ss+curvature_d*d_s/factor)*cosine_angle**2)/factor
                 +reference_curvature)*cosine_angle/factor
    speed = np.sqrt(factor**2*s_d**2+(d_s*s_d)**2)
    angle_gradient = factor/cosine_angle*curvature-reference_curvature
    acceleration = (s_dd*factor/cosine_angle+s_d**2/cosine_angle*
                    (d_s*angle_gradient-curvature_d))
    path.x, path.y, path.yaw = x.tolist(), y.tolist(), yaw.tolist()
    path.c, path.v, path.a = curvature.tolist(), speed.tolist(), acceleration.tolist()
    return path


def standard_sampling(csp, sc, dc):
    """使用原版多项式与评分，扩大时间/速度采样后检查车辆约束。"""
    counts = {key: 0 for key in (
        "invalid", "speed", "acceleration", "steering",
        "steering_rate", "lateral_acceleration", "clearance", "stopping", "ok",
    )}
    best = None
    offsets = terminal_offsets(sc, dc)
    for duration in PREDICTION_TIMES:
        times = list(np.linspace(
            0.0, duration, int(round(duration/PLAN_SAMPLE_DT)) + 1))
        time_array = np.asarray(times)
        lateral_cache = {}
        for terminal_d in offsets:
            lateral = planner.QuinticPolynomial(
                dc[0], dc[1]*sc[1], dc[2]*sc[1]**2+dc[1]*sc[2],
                terminal_d, 0.0, 0.0, duration)
            lateral_cache[float(terminal_d)] = (
                lateral.calc_point(time_array),
                lateral.calc_first_derivative(time_array),
                lateral.calc_second_derivative(time_array),
                lateral.calc_third_derivative(time_array))
        for terminal_speed in END_SPEEDS:
            longitudinal = planner.FrenetPath()
            polynomial = planner.QuarticPolynomial(
                sc[0], sc[1], sc[2], terminal_speed, 0.0, duration
            )
            # 显式包含终点，避免漏查预测时域最后一段。
            longitudinal.t = times
            longitudinal.s = [polynomial.calc_point(t) for t in longitudinal.t]
            longitudinal.s_d = [polynomial.calc_first_derivative(t)
                                for t in longitudinal.t]
            longitudinal.s_dd = [polynomial.calc_second_derivative(t)
                                 for t in longitudinal.t]
            longitudinal.s_ddd = [polynomial.calc_third_derivative(t)
                                  for t in longitudinal.t]
            samples = reference_samples(csp, longitudinal.s)
            for terminal_d in offsets:
                path = lateral_without_deepcopy(
                    longitudinal, terminal_d, dc, duration,
                    lateral_cache[float(terminal_d)])
                path = cartesian_from_cached_reference(path, samples)
                if len(path.x) != len(path.t) or len(path.x) < 2:
                    counts["invalid"] += 1
                    continue
                arrays = np.asarray([path.v, path.a, path.c])
                if not np.all(np.isfinite(arrays)) or min(path.s_d) < 0.2:
                    counts["invalid"] += 1
                elif max(path.v) > planner.MAX_SPEED:
                    counts["speed"] += 1
                elif max(abs(np.asarray(path.a))) > PLAN_ACCEL + 1e-6:
                    counts["acceleration"] += 1
                elif max(abs(np.arctan(WHEELBASE*np.asarray(path.c)))) > PLAN_STEER + 1e-6:
                    counts["steering"] += 1
                elif path_steering_rate(path) > PLAN_STEER_RATE + 1e-6:
                    counts["steering_rate"] += 1
                elif max(np.asarray(path.v)**2 * abs(np.asarray(path.c))) > MAX_LATERAL_ACCEL:
                    counts["lateral_acceleration"] += 1
                elif trajectory_clearance(path.x, path.y) < PLAN_CLEARANCE:
                    counts["clearance"] += 1
                elif not candidate_can_stop(path):
                    counts["stopping"] += 1
                else:
                    counts["ok"] += 1
                    lateral_cost = (planner.K_J*PLAN_SAMPLE_DT/planner.DT * sum(np.asarray(path.d_ddd)**2)
                                    + planner.K_T*duration + planner.K_D*lateral_destination_cost(path.d[-1]))
                    longitudinal_cost = (
                        planner.K_J*PLAN_SAMPLE_DT/planner.DT*sum(np.asarray(path.s_ddd)**2)
                        + planner.K_T*duration
                        + planner.LONGITUDINAL_MOVEMENT_STRATEGY.calc_destination_cost(path)
                    )
                    path.cf = planner.K_LAT*lateral_cost + planner.K_LON*longitudinal_cost
                    path.cf += recovery_risk_cost(path) + profile_cost(path)
                    if best is None or path.cf <= best.cf:
                        best = path
    return best, counts


def reachable_sampling(csp, sc, dc, grid_s, grid_xy):
    """先用自行车模型生成可达末端状态，再用 Frenet 多项式连接。

    初始与末端 s、d 的位置、一阶、二阶导数均参与连接，避免固定
    末端 s 加速度、横向速度和横向加速度都为零而排除正常运动。
    模型预测通过检查，并不保证连接后的多项式通过检查；须再次验证。
    """
    counts = {key: 0 for key in (
        "invalid", "speed", "acceleration", "steering",
        "steering_rate", "lateral_acceleration", "clearance", "stopping", "ok",
    )}
    reference = (sc[0], *csp.calc_position(sc[0]), csp.calc_yaw(sc[0]),
                 csp.calc_curvature(sc[0]), csp.calc_curvature_rate(sc[0]))
    x, y, yaw, curvature, speed, initial_accel = CONVERTER.frenet_to_cartesian(
        *reference, sc, dc
    )
    start = np.array([x, y, yaw, speed])
    initial_steer = np.arctan(WHEELBASE*curvature)
    steering_targets = np.unique(np.clip(np.r_[
        0.0, initial_steer + np.deg2rad([-15., -7.5, 0., 7.5, 15.]),
        -PLAN_STEER, PLAN_STEER,
    ], -PLAN_STEER, PLAN_STEER))
    accel_targets = np.unique(np.clip(np.r_[
        -3., -1.5, 0., 1.5, 3., initial_accel,
    ], -PLAN_ACCEL, PLAN_ACCEL))
    best = None
    for duration in (0.2, 0.4, 0.6, 1.0, 1.4, 1.8):
        if duration + 1e-9 < MIN_PLAN_HORIZON_ACTIVE:
            continue
        for end_steer in steering_targets:
            if abs(end_steer-initial_steer)/duration > PLAN_STEER_RATE+1e-6:
                counts["steering_rate"] += len(accel_targets)
                continue
            # All acceleration targets share this steering schedule. Propagate
            # them together; every target keeps the original control time grid.
            predicted_states = np.repeat(start[None, :], len(accel_targets), axis=0)
            steps = int(round(duration/CONTROL_DT))
            for index in range(steps):
                ratio = (index+0.5)/steps
                steering = initial_steer + ratio*(end_steer-initial_steer)
                acceleration = initial_accel + ratio*(accel_targets-initial_accel)
                predicted_states = advance_many(
                    predicted_states, acceleration, steering, CONTROL_DT)
            for end_accel, predicted in zip(accel_targets, predicted_states):
                if predicted[3] < 0.3:
                    counts["invalid"] += 1
                    continue
                try:
                    end_sc, end_dc = actual_to_frenet(
                        csp, predicted, end_accel, end_steer, grid_s, grid_xy
                    )
                except RuntimeError:
                    counts["invalid"] += 1
                    continue
                if end_sc[0] <= sc[0] or end_sc[0] >= csp.s[-1]:
                    counts["invalid"] += 1
                    continue
                longitudinal = planner.QuinticPolynomial(*sc, *end_sc, duration)
                lateral = planner.QuinticPolynomial(
                    dc[0], dc[1]*sc[1], dc[2]*sc[1]**2 + dc[1]*sc[2],
                    end_dc[0], end_dc[1]*end_sc[1],
                    end_dc[2]*end_sc[1]**2 + end_dc[1]*end_sc[2], duration,
                )
                path = planner.FrenetPath()
                # 短时域须加密检查，避免只检查两个端点而漏掉中间的超限。
                dt = 0.05
                path.t = list(np.linspace(0., duration, int(round(duration/dt))+1))
                path.s = [longitudinal.calc_point(t) for t in path.t]
                path.s_d = [longitudinal.calc_first_derivative(t) for t in path.t]
                path.s_dd = [longitudinal.calc_second_derivative(t) for t in path.t]
                path.s_ddd = [longitudinal.calc_third_derivative(t) for t in path.t]
                if min(path.s_d) < 0.2:
                    counts["invalid"] += 1
                    continue
                path.d = [lateral.calc_point(t) for t in path.t]
                path.d_d = [lateral.calc_first_derivative(t)/sd
                            for t, sd in zip(path.t, path.s_d)]
                path.d_dd = [(lateral.calc_second_derivative(t)-ds*sdd)/sd**2
                             for t, ds, sd, sdd in
                             zip(path.t, path.d_d, path.s_d, path.s_dd)]
                path.d_ddd = [lateral.calc_third_derivative(t) for t in path.t]
                if np.max(np.abs(path.d)) > planner.MAX_ROAD_WIDTH:
                    counts["invalid"] += 1
                    continue
                samples = reference_samples(csp, path.s)
                path = cartesian_from_cached_reference(path, samples)
                if len(path.x) != len(path.t):
                    counts["invalid"] += 1
                    continue
                arrays = np.asarray([path.v, path.a, path.c])
                if not np.all(np.isfinite(arrays)):
                    counts["invalid"] += 1
                elif max(path.v) > planner.MAX_SPEED:
                    counts["speed"] += 1
                elif np.max(np.abs(path.a)) > PLAN_ACCEL+1e-6:
                    counts["acceleration"] += 1
                elif np.max(np.abs(np.arctan(WHEELBASE*np.asarray(path.c)))) > PLAN_STEER+1e-6:
                    counts["steering"] += 1
                elif path_steering_rate(path) > PLAN_STEER_RATE + 1e-6:
                    counts["steering_rate"] += 1
                elif np.max(np.asarray(path.v)**2*np.abs(path.c)) > MAX_LATERAL_ACCEL:
                    counts["lateral_acceleration"] += 1
                elif trajectory_clearance(path.x, path.y) < PLAN_CLEARANCE:
                    counts["clearance"] += 1
                elif not candidate_can_stop(path):
                    counts["stopping"] += 1
                else:
                    counts["ok"] += 1
                    # 与常规采样使用同一评分；修正加密采样带来的累加密度差异。
                    lateral_cost = (
                        planner.K_J*dt/planner.DT*sum(np.asarray(path.d_ddd)**2)
                        + planner.K_T*duration + planner.K_D*lateral_destination_cost(path.d[-1])
                    )
                    longitudinal_cost = (
                        planner.K_J*dt/planner.DT*sum(np.asarray(path.s_ddd)**2)
                        + planner.K_T*duration
                        + planner.LONGITUDINAL_MOVEMENT_STRATEGY.calc_destination_cost(path)
                    )
                    path.cf = planner.K_LAT*lateral_cost + planner.K_LON*longitudinal_cost
                    path.cf += recovery_risk_cost(path) + profile_cost(path)
                    if best is None or path.cf <= best.cf:
                        best = path
    return best, counts


def rollout_sampling(csp, sc, dc, grid_s, grid_xy):
    """多项式连接无解时，直接保留满足约束的车辆模型预测轨迹。

    不再拟合多项式，避免拟合后的曲率变化速度超过转向机构能力。
    这是局部后备采样，仍要检查整条折线、转向速度及横向加速度。
    """
    counts = {key: 0 for key in (
        "invalid", "speed", "acceleration", "steering", "steering_rate",
        "lateral_acceleration", "clearance", "stopping", "ok",
    )}
    reference = (sc[0], *csp.calc_position(sc[0]), csp.calc_yaw(sc[0]),
                 csp.calc_curvature(sc[0]), csp.calc_curvature_rate(sc[0]))
    x, y, yaw, curvature, speed, initial_accel = CONVERTER.frenet_to_cartesian(
        *reference, sc, dc)
    initial_steer = float(np.arctan(WHEELBASE*curvature))
    targets = np.unique(np.clip(np.r_[
        initial_steer + np.deg2rad([-15., -7.5, 0., 7.5, 15.]),
        0.0, -PLAN_STEER, PLAN_STEER], -PLAN_STEER, PLAN_STEER))
    accel_targets = np.unique(np.clip(np.r_[
        -3., -1.5, 0., 1.5, 3., initial_accel], -PLAN_ACCEL, PLAN_ACCEL))
    best = None
    for duration in (0.4, 0.8, 1.2, 1.8):
        if duration + 1e-9 < MIN_PLAN_HORIZON_ACTIVE:
            continue
        for end_steer in targets:
            if abs(end_steer-initial_steer)/duration > PLAN_STEER_RATE+1e-6:
                counts["steering_rate"] += len(accel_targets)
                continue
            for end_accel in accel_targets:
                count = int(round(duration/CONTROL_DT))
                pt = np.linspace(0.0, duration, count+1)
                deltas = np.linspace(initial_steer, end_steer, count+1)
                pa = np.linspace(initial_accel, end_accel, count+1)
                predicted = np.array([x, y, yaw, speed])
                history = [predicted.copy()]
                for i in range(count):
                    predicted = advance(predicted, (pa[i]+pa[i+1])/2,
                                        (deltas[i]+deltas[i+1])/2, CONTROL_DT)
                    history.append(predicted.copy())
                history = np.asarray(history)
                if np.min(history[:, 3]) < 0.3:
                    counts["invalid"] += 1
                    continue
                path = planner.FrenetPath()
                path.t = list(pt)
                path.x, path.y, path.yaw, path.v = [list(a) for a in history.T]
                path.a = list(pa)
                path.c = list(np.tan(deltas)/WHEELBASE)
                if max(path.v) > planner.MAX_SPEED:
                    counts["speed"] += 1
                    continue
                if np.max(np.abs(pa)) > PLAN_ACCEL+1e-6:
                    counts["acceleration"] += 1
                    continue
                if np.max(np.abs(deltas)) > PLAN_STEER+1e-6:
                    counts["steering"] += 1
                    continue
                if np.max(history[:, 3]**2*np.abs(path.c)) > MAX_LATERAL_ACCEL:
                    counts["lateral_acceleration"] += 1
                    continue
                if trajectory_clearance(path.x, path.y) < PLAN_CLEARANCE:
                    counts["clearance"] += 1
                    continue
                try:
                    end_sc, end_dc = actual_to_frenet(
                        csp, history[-1], end_accel, end_steer, grid_s, grid_xy)
                except RuntimeError:
                    counts["invalid"] += 1
                    continue
                if end_sc[0] <= sc[0] or abs(end_dc[0]) > planner.MAX_ROAD_WIDTH:
                    counts["invalid"] += 1
                    continue
                if not candidate_can_stop(path):
                    counts["stopping"] += 1
                    continue
                counts["ok"] += 1
                jerk = (end_accel-initial_accel)/duration
                path.cf = float(0.2*duration + 0.1*duration*jerk**2
                                + lateral_destination_cost(end_dc[0])
                                + (planner.TARGET_SPEED-history[-1, 3])**2)
                path.model_end_state = False
                path.rollout = True
                path.cf += recovery_risk_cost(path) + profile_cost(path)
                if USE_SPEED_GUIDANCE and SPEED_PROFILE is not None:
                    path.cf += 8.*duration*max(0., end_sc[1]-SPEED_PROFILE.at(end_sc[0]))**2
                if best is None or path.cf < best.cf:
                    best = path
    if best is not None:
        # 仅为最终选中的后备轨迹补齐 Frenet 状态；执行仍使用模型预测位置。
        for i in range(len(best.t)):
            sc_i, dc_i = actual_to_frenet(
                csp, np.array([best.x[i], best.y[i], best.yaw[i], best.v[i]]),
                best.a[i], float(np.arctan(WHEELBASE*best.c[i])), grid_s, grid_xy)
            best.s.append(sc_i[0])
            best.s_d.append(sc_i[1])
            best.s_dd.append(sc_i[2])
            best.d.append(dc_i[0])
            best.d_d.append(dc_i[1])
            best.d_dd.append(dc_i[2])
    return best, counts


def constrained_planning(csp, sc, dc, grid_s, grid_xy):
    best, counts = standard_sampling(csp, sc, dc)
    if best is not None:
        best.model_end_state = False
        return best, counts
    best, extra = reachable_sampling(csp, sc, dc, grid_s, grid_xy)
    counts = {key: counts[key]+extra[key] for key in counts}
    if best is not None:
        best.model_end_state = True
        return best, counts
    best, extra = rollout_sampling(csp, sc, dc, grid_s, grid_xy)
    counts = {key: counts[key]+extra[key] for key in counts}
    return best, counts


def recovery_planning(csp, sc, dc, grid_s, grid_xy):
    """只改变采样和评分目标，仍经过原来的全部约束及安全停车检查。

    恢复时允许低速末端和当前横向偏移附近的细采样；结束后恢复参数。
    """
    global END_SPEEDS, PREDICTION_TIMES, RECOVERY_SAMPLING, RECOVERY_LATERAL_TARGET
    saved = END_SPEEDS, PREDICTION_TIMES, RECOVERY_SAMPLING, planner.TARGET_SPEED, RECOVERY_LATERAL_TARGET
    try:
        END_SPEEDS = np.array([.4, .7, 1., 1.5, 2., 2.5, 3.3])
        PREDICTION_TIMES = (1., 2., 3., 4., 4.8)
        RECOVERY_SAMPLING = True
        RECOVERY_LATERAL_TARGET = choose_recovery_offset(csp, sc, dc)
        if not USE_SPEED_GUIDANCE:
            planner.TARGET_SPEED = RECOVERY_TARGET_SPEED
            return constrained_planning(csp, sc, dc, grid_s, grid_xy)
        planner.TARGET_SPEED = min(RECOVERY_TARGET_SPEED, float(SPEED_PROFILE.at(sc[0])))
        END_SPEEDS = np.unique(np.r_[END_SPEEDS, planner.TARGET_SPEED])
        geometric, counts = spatial_retimed_sampling(csp, sc, dc, grid_s, grid_xy)
        if geometric is not None:
            return geometric, counts
        path, extra = constrained_planning(csp, sc, dc, grid_s, grid_xy)
        return path, {k: counts[k]+extra[k] for k in counts}
    finally:
        END_SPEEDS, PREDICTION_TIMES, RECOVERY_SAMPLING, planner.TARGET_SPEED, RECOVERY_LATERAL_TARGET = saved



def advance(state, acceleration, steering, dt):
    """制动到零时只积分到停车时刻，不允许倒车或继续漂移。"""
    state = state.copy()
    moving_dt = dt
    if acceleration < 0.0 and state[3] + acceleration*dt < 0.0:
        moving_dt = state[3] / -acceleration
    next_speed = max(0.0, state[3] + acceleration*moving_dt)
    average_speed = 0.5*(state[3] + next_speed)
    yaw_change = average_speed/WHEELBASE*np.tan(steering)*moving_dt
    state[0] += average_speed*np.cos(state[2] + yaw_change/2)*moving_dt
    state[1] += average_speed*np.sin(state[2] + yaw_change/2)*moving_dt
    state[2] = (state[2] + yaw_change + np.pi) % (2*np.pi) - np.pi
    state[3] = next_speed
    return state


def advance_many(states, accelerations, steering, dt):
    """Propagate independent candidates with the same bicycle-model step."""
    speeds = states[:, 3]
    stop_dt = np.divide(speeds, -accelerations,
                        out=np.full_like(speeds, dt), where=accelerations < 0.0)
    moving_dt = np.where((accelerations < 0.0)
                         & (speeds+accelerations*dt < 0.0), stop_dt, dt)
    next_speeds = np.maximum(0.0, speeds+accelerations*moving_dt)
    average = .5*(speeds+next_speeds)
    yaw_change = average/WHEELBASE*np.tan(steering)*moving_dt
    result = states.copy()
    result[:, 0] += average*np.cos(states[:, 2]+yaw_change/2)*moving_dt
    result[:, 1] += average*np.sin(states[:, 2]+yaw_change/2)*moving_dt
    result[:, 2] = (states[:, 2]+yaw_change+np.pi) % (2*np.pi)-np.pi
    result[:, 3] = next_speeds
    return result


def braking_positions(state, steering):
    """向量化重现 advance 的恒转角最大制动，每个点间隔不超过 CONTROL_DT。

    使用每段行驶距离和中点航向，与执行器冻结转角后的制动积分相同。
    最后一个不足 0.02 秒的区间只积分到速度为零。
    """
    stop_time = max(0.0, float(state[3]))/MAX_ACCEL
    if stop_time <= 1e-12:
        return np.asarray(state[:2], dtype=float)[None, :]
    count = int(np.ceil(stop_time/CONTROL_DT))
    t = np.minimum(np.arange(count+1)*CONTROL_DT, stop_time)
    distance = state[3]*t - 0.5*MAX_ACCEL*t*t
    increments = np.diff(distance)
    curvature = np.tan(steering)/WHEELBASE
    headings = state[2] + curvature*(distance[:-1]+distance[1:])/2
    dx = np.r_[0.0, np.cumsum(increments*np.cos(headings))]
    dy = np.r_[0.0, np.cumsum(increments*np.sin(headings))]
    return np.column_stack((state[0]+dx, state[1]+dy))


def braking_preview(state, steering):
    positions = braking_positions(state, steering)
    return trajectory_clearance(positions[:, 0], positions[:, 1])


def batched_braking_clearance(states, steerings):
    """Evaluate every scheduled backup-stop check with shared array work.

    Each row retains the same 0.02 s samples and the same constant-steering,
    maximum-deceleration backup policy as braking_positions. Samples after a
    row has stopped duplicate its endpoint and cannot reduce the minimum gap.
    """
    states = np.asarray(states, dtype=float)
    steerings = np.asarray(steerings, dtype=float)
    stop_times = np.maximum(states[:, 3], 0.0)/MAX_ACCEL
    count = int(np.max(np.ceil(stop_times/CONTROL_DT)))
    obstacles = np.asarray(planner.OBSTACLES, dtype=float)
    if count == 0:
        dx = obstacles[:, None, 0]-states[None, :, 0]
        dy = obstacles[:, None, 1]-states[None, :, 1]
        return float(np.sqrt(np.min(dx*dx+dy*dy))-planner.ROBOT_RADIUS)
    t = np.minimum(np.arange(count+1)[None, :]*CONTROL_DT,
                   stop_times[:, None])
    distance = states[:, 3, None]*t-0.5*MAX_ACCEL*t*t
    increments = np.diff(distance, axis=1)
    curvature = np.tan(steerings)/WHEELBASE
    headings = (states[:, 2, None]
                +curvature[:, None]*(distance[:, :-1]+distance[:, 1:])/2)
    x = states[:, 0, None]+np.concatenate((
        np.zeros((len(states), 1)),
        np.cumsum(increments*np.cos(headings), axis=1)), axis=1)
    y = states[:, 1, None]+np.concatenate((
        np.zeros((len(states), 1)),
        np.cumsum(increments*np.sin(headings), axis=1)), axis=1)
    starts = np.stack((x[:, :-1], y[:, :-1]), axis=2).reshape(-1, 2)
    vectors = np.stack((np.diff(x, axis=1), np.diff(y, axis=1)), axis=2).reshape(-1, 2)
    lengths2 = np.maximum(np.sum(vectors*vectors, axis=1), 1e-12)
    offsets = obstacles[:, None, :]-starts[None, :, :]
    ratio = np.clip(np.sum(offsets*vectors[None, :, :], axis=2)
                    /lengths2[None, :], 0.0, 1.0)
    residual = offsets-ratio[:, :, None]*vectors[None, :, :]
    return float(np.sqrt(np.min(np.sum(residual*residual, axis=2)))
                 -planner.ROBOT_RADIUS)


def candidate_can_stop(path):
    """检查下一次重规划前的执行前缀，包含其终点的备份停车余量。

    后面的长预测轨迹不会一次全部执行，因此不因远期恒转角制动失败
    排除当前可安全执行的前缀。整个候选路径仍检查碰撞和车辆约束。
    PARK 期间每个实际控制步也要通过保护器，失败便切换到已验证的制动。
    这是离散模型检查，不是连续时间可达集证明。
    """
    times = np.asarray(path.t)
    prefix_end = min(REPLAN_DT, times[-1])
    check_times = np.unique(np.r_[0.0, np.arange(STOP_CHECK_DT, prefix_end,
                                                STOP_CHECK_DT), prefix_end])
    yaws = np.unwrap(np.asarray(path.yaw))
    deltas = np.arctan(WHEELBASE*np.asarray(path.c))
    states = np.column_stack((np.interp(check_times, times, path.x),
                              np.interp(check_times, times, path.y),
                              np.interp(check_times, times, yaws),
                              np.interp(check_times, times, path.v)))
    steering = np.interp(check_times, times, deltas)
    minimum = batched_braking_clearance(states, steering)
    # 停车分支和实际保护使用同一阈值。整条行驶轨迹的碰撞余量
    # 仍为 PLAN_CLEARANCE，不能把该余量误用成当前状态的停车阈值。
    if minimum < STOP_GUARD_MARGIN:
        return False
    path.minimum_stop_clearance = minimum
    return True


def pursuit(x, y, yaw, px, py, speed):
    """连续折线与前视圆求交，目标点不局限于离散采样点。"""
    point = np.array([x, y])
    starts = np.column_stack((px[:-1], py[:-1]))
    vectors = np.column_stack((np.diff(px), np.diff(py)))
    lengths2 = np.sum(vectors**2, axis=1)
    ratios = np.clip(
        np.sum((point - starts) * vectors, axis=1)
        / np.maximum(lengths2, 1e-12), 0.0, 1.0,
    )
    projections = starts + ratios[:, None] * vectors
    index = int(np.argmin(np.sum((projections - point)**2, axis=1)))
    lookahead = 1.5 + 0.2 * speed
    target = np.array([px[-1], py[-1]])
    for i in range(index, len(vectors)):
        if lengths2[i] <= 1e-12:
            continue
        relative = starts[i] - point
        b = 2.0 * np.dot(relative, vectors[i])
        c = np.dot(relative, relative) - lookahead**2
        discriminant = b*b - 4.0*lengths2[i]*c
        if discriminant < 0.0:
            continue
        root = (-b + np.sqrt(discriminant)) / (2.0*lengths2[i])
        lower = ratios[index] if i == index else 0.0
        if lower - 1e-10 <= root <= 1.0 + 1e-10:
            target = starts[i] + np.clip(root, 0.0, 1.0) * vectors[i]
            break
    dx, dy = target - point
    alpha = np.arctan2(dy, dx) - yaw
    requested = np.arctan2(
        2.0 * WHEELBASE * np.sin(alpha), max(np.hypot(dx, dy), 1e-6)
    )
    return float(np.clip(requested, -MAX_STEER, MAX_STEER)), requested


def polyline_distance(point, px, py):
    if len(px) == 1:
        return float(np.linalg.norm(np.array([px[0], py[0]])-point))
    starts = np.column_stack((px[:-1], py[:-1]))
    vectors = np.column_stack((np.diff(px), np.diff(py)))
    lengths2 = np.maximum(np.sum(vectors**2, axis=1), 1e-12)
    ratio = np.clip(np.sum((point - starts)*vectors, axis=1)/lengths2, 0, 1)
    closest = starts + ratio[:, None] * vectors
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


def reference_projection(csp, x, y, grid_s, grid_xy):
    """先找粗略最近点，再连续优化；不能直接拿粗采样点作转换基准。"""
    point = np.array([x, y])
    index = int(np.argmin(np.sum((grid_xy - point)**2, axis=1)))
    left = grid_s[max(0, index - 2)]
    right = grid_s[min(len(grid_s) - 1, index + 2)]

    def objective(s):
        rx, ry = csp.calc_position(float(s))
        return (x - rx)**2 + (y - ry)**2

    result = minimize_scalar(
        objective, bounds=(left, right), method="bounded",
        options={"xatol": 1e-9},
    )
    # bounded 搜索不保证取到端点，显式比较端点。
    s = min((left, float(result.x), right), key=objective)
    rx, ry = csp.calc_position(s)
    return (s, rx, ry, csp.calc_yaw(s), csp.calc_curvature(s),
            csp.calc_curvature_rate(s))


def actual_to_frenet(csp, state, acceleration, steering, grid_s, grid_xy):
    x, y, yaw, speed = state
    reference = reference_projection(csp, x, y, grid_s, grid_xy)
    _, rx, ry, reference_yaw, reference_curvature, _ = reference
    lateral = -(x - rx)*np.sin(reference_yaw) + (y - ry)*np.cos(reference_yaw)
    if (np.cos(yaw - reference_yaw) <= 0.1
            or 1.0 - reference_curvature*lateral <= 0.1):
        raise RuntimeError("车辆姿态超出当前 Frenet 转换的适用范围")
    # 自行车模型的实际曲率来自已执行转角，而非规划轨迹。
    curvature = np.tan(steering) / WHEELBASE
    sc, dc = CONVERTER.cartesian_to_frenet(
        *reference, x, y, speed, acceleration, yaw, curvature,
    )
    if not np.all(np.isfinite(sc + dc)):
        raise RuntimeError("Frenet 反馈状态出现非有限值")
    return sc, dc


def unpack_path(path):
    count = min(len(path.t), len(path.x), len(path.y),
                len(path.yaw), len(path.v), len(path.a))
    if count < 2:
        raise RuntimeError("规划轨迹的有效点数不足")
    return tuple(np.asarray(getattr(path, key)[:count])
                 for key in ("t", "x", "y", "v", "a"))


def goal_planning(csp, sc, dc, goal_s):
    """纵向按时间规划到零速，横向按弧长连接到参考线终点。

    横向使用空间导数，避免停车时除以 s_dot=0。规划端点为后轴中心。
    每条候选均检查整段采样轨迹的车辆约束及障碍物折线净间距。
    """
    counts = {key: 0 for key in (
        "invalid", "speed", "acceleration", "steering",
        "steering_rate", "lateral_acceleration", "clearance", "stopping", "ok",
    )}
    remaining = goal_s - sc[0]
    if remaining <= 0.05 or sc[1] <= 0.05:
        counts["invalid"] = 1
        return None, counts
    lateral = planner.QuinticPolynomial(*dc, 0.0, 0.0, 0.0, remaining)
    best = None
    for duration in np.arange(2.0, 10.01, 0.25):
        longitudinal = planner.QuinticPolynomial(*sc, goal_s, 0.0, 0.0, duration)
        path = planner.FrenetPath()
        path.t = list(np.linspace(0.0, duration, int(round(duration/0.02)) + 1))
        path.s = [longitudinal.calc_point(t) for t in path.t]
        path.s_d = [longitudinal.calc_first_derivative(t) for t in path.t]
        path.s_dd = [longitudinal.calc_second_derivative(t) for t in path.t]
        path.s_ddd = [longitudinal.calc_third_derivative(t) for t in path.t]
        if (min(path.s_d) < -1e-7 or min(path.s) < sc[0]-1e-7
                or max(path.s) > goal_s+1e-7):
            counts["invalid"] += 1
            continue
        path.s = list(np.clip(path.s, sc[0], goal_s))
        path.s_d = list(np.maximum(path.s_d, 0.0))
        offsets = np.asarray(path.s) - sc[0]
        path.d = [lateral.calc_point(u) for u in offsets]
        path.d_d = [lateral.calc_first_derivative(u) for u in offsets]
        path.d_dd = [lateral.calc_second_derivative(u) for u in offsets]
        # jerk 在时间坐标中的链式转换，供评分使用。
        path.d_ddd = [lateral.calc_third_derivative(u)*v**3
                      + 3*dd*v*a + d*jerk
                      for u, v, a, d, dd, jerk in zip(
                          offsets, path.s_d, path.s_dd, path.d_d,
                          path.d_dd, path.s_ddd)]
        path = planner.LATERAL_MOVEMENT_STRATEGY.calc_cartesian_parameters(path, csp)
        if (len(path.x) != len(path.t)
                or not np.all(np.isfinite([path.v, path.a, path.c]))):
            counts["invalid"] += 1
        elif max(path.v) > planner.MAX_SPEED:
            counts["speed"] += 1
        elif max(abs(np.asarray(path.a))) > PLAN_ACCEL + 1e-6:
            counts["acceleration"] += 1
        elif max(abs(np.arctan(WHEELBASE*np.asarray(path.c)))) > PLAN_STEER + 1e-6:
            counts["steering"] += 1
        elif path_steering_rate(path) > PLAN_STEER_RATE + 1e-6:
            counts["steering_rate"] += 1
        elif max(np.asarray(path.v)**2*abs(np.asarray(path.c))) > MAX_LATERAL_ACCEL:
            counts["lateral_acceleration"] += 1
        elif trajectory_clearance(path.x, path.y) < PLAN_CLEARANCE:
            counts["clearance"] += 1
        elif not candidate_can_stop(path):
            counts["stopping"] += 1
        else:
            counts["ok"] += 1
            path.cf = float(0.2*duration + 0.02*np.sum(
                np.asarray(path.s_ddd)**2 + np.asarray(path.d_ddd)**2))
            path.model_end_state = False
            path.goal_path = True
            if best is None or path.cf < best.cf:
                best = path
    return best, counts


def handoff_preview(path, start_age, state, steering):
    """从交接时实际状态预演短时控制；安全检查只针对本例的离散模型。"""
    pt, px, py, pv, pa = unpack_path(path)
    current = np.asarray(state, dtype=float).copy()
    delta = float(steering)
    steps = min(int(round(HANDOFF_PREVIEW_S/CONTROL_DT)),
                int(np.floor((pt[-1]-start_age)/CONTROL_DT)))
    if steps < int(round(REPLAN_DT/CONTROL_DT)):
        return False, "horizon"
    for i in range(steps):
        age = start_age+i*CONTROL_DT
        target = float(np.interp(age, pt, pv))
        accel = float(np.clip(np.interp(age, pt, pa)
                              +1.5*(target-current[3]), -MAX_ACCEL, MAX_ACCEL))
        if getattr(path, "retimed", False):
            _, requested = timed_curve_tracking(path, age, current)
        else:
            _, requested = pursuit(*current[:3], px, py, current[3])
        if abs(requested) > MAX_STEER-HANDOFF_STEER_RESERVE:
            return False, "steering"
        next_delta = steering_actuator(delta, requested, CONTROL_DT)
        following = advance(current, accel, 0.5*(delta+next_delta), CONTROL_DT)
        lateral = max(current[3], following[3])**2*abs(
            np.tan(max(abs(delta), abs(next_delta))))/WHEELBASE
        if (polyline_distance(following[:2], px, py) > TRACKING_ERROR_LIMIT
                or trajectory_clearance([current[0], following[0]],
                                        [current[1], following[1]]) < STOP_GUARD_MARGIN
                or braking_preview(following, next_delta) < STOP_GUARD_MARGIN
                or lateral > MAX_LATERAL_ACCEL+1e-6):
            return False, "tracking_or_safety"
        current, delta = following, next_delta
    return True, "ok"


def main(show_animation=True, fixed_delay_ms=None,
         delay_schedule_ms=None, short_horizon_enabled=True):
    global SPEED_PROFILE, PREVIOUS_GEOMETRY_ENDPOINT
    global USE_SPEED_GUIDANCE, INITIAL_PREVIEW_STRESS
    global MIN_PLAN_HORIZON_ACTIVE
    if fixed_delay_ms is not None and fixed_delay_ms < 0:
        raise ValueError("固定规划延迟不能为负")
    # 常态保留较长预测；接近障碍物时才允许覆盖交接所需时域的短候选。
    MIN_PLAN_HORIZON_ACTIVE = (0.0 if fixed_delay_ms == 0 else
        max(MIN_PLAN_HORIZON, (fixed_delay_ms or 0)/1000.0 + HANDOFF_PREVIEW_S))
    PREVIOUS_GEOMETRY_ENDPOINT = None
    if planner.LATERAL_MOVEMENT != planner.LateralMovement.HIGH_SPEED:
        raise RuntimeError("本练习使用原版 HIGH_SPEED 场景，请恢复该配置")
    if planner.LONGITUDINAL_MOVEMENT != planner.LongitudinalMovement.VELOCITY_KEEPING:
        raise RuntimeError("本练习使用原版 VELOCITY_KEEPING 场景，请恢复该配置")
    _, _, _, _, raw_reference = planner.generate_target_course(planner.WX, planner.WY)
    csp = ArcLengthReference(raw_reference)
    SPEED_PROFILE = CurveSpeedProfile(csp)
    draw_s = np.linspace(0.0, csp.s[-1] - 1e-6, 800)
    reference_xy = np.asarray([csp.calc_position(float(s)) for s in draw_s])
    rx, ry = reference_xy.T
    print(f"原样条参数范围：{raw_reference.s[-1]:.3f}；数值积分曲线弧长：{csp.s[-1]:.3f} m")
    end_s = float(csp.s[-1]) - 1e-6
    goal_s = end_s
    goal_xy = np.asarray(csp.calc_position(goal_s))
    goal_reached = False
    goal_start = None
    grid_s = np.linspace(0.0, end_s, int(np.ceil(end_s / 0.2)) + 1)
    grid_xy = np.asarray([csp.calc_position(float(s)) for s in grid_s])
    s0 = planner.INITIAL_COURSE_POSITION
    preview_mask = ((SPEED_PROFILE.s >= s0)
                    & (SPEED_PROFILE.s <= min(csp.s[-1], s0+GUIDANCE_PREVIEW_DISTANCE)))
    maximum_gradient = float(np.max(np.abs(
        SPEED_PROFILE.steering_gradient[preview_mask]))) if np.any(preview_mask) else 0.0
    INITIAL_PREVIEW_STRESS = (planner.INITIAL_SPEED*maximum_gradient
                              / max(PLAN_STEER_RATE, 1e-9))
    USE_SPEED_GUIDANCE = INITIAL_PREVIEW_STRESS > GUIDANCE_STRESS_THRESHOLD
    print(f"前方 {GUIDANCE_PREVIEW_DISTANCE:.0f} m 转向速度压力系数："
          f"{INITIAL_PREVIEW_STRESS:.3f}；"
          f"{'启用曲率速度预瞄' if USE_SPEED_GUIDANCE else '沿用常规重规划'}。")
    ref_x, ref_y = csp.calc_position(s0)
    x, y, yaw, curvature, speed, acceleration = CONVERTER.frenet_to_cartesian(
        s0, ref_x, ref_y, csp.calc_yaw(s0), csp.calc_curvature(s0),
        csp.calc_curvature_rate(s0),
        [s0, planner.INITIAL_SPEED, planner.INITIAL_ACCEL],
        [planner.INITIAL_LAT_POSITION, planner.INITIAL_LAT_SPEED,
         planner.INITIAL_LAT_ACCELERATION],
    )
    steering = float(np.arctan(WHEELBASE * curvature))
    if abs(steering) > MAX_STEER:
        raise RuntimeError("初始曲率需要的转角超出车辆限制")
    initial_steering = steering
    steering_commands, rate_limited = [], []
    state = np.array([x, y, yaw, speed], dtype=float)
    times, states = [0.0], [state.copy()]
    speed_caps = [float(SPEED_PROFILE.at(s0)*speed/max(planner.INITIAL_SPEED, .2))
                  if USE_SPEED_GUIDANCE else speed]
    print(f"初始实际车速 {speed*3.6:.2f} km/h；预瞄建议 {speed_caps[0]*3.6:.2f} km/h。")
    errors, timed_errors, target_speeds = [np.nan], [np.nan], [speed]
    steerings, accelerations, saturation = [], [], []
    plans, attempts = [], []
    path = None
    recovery_active = False
    recovery_start = None
    recovery_start_s = None
    recovery_events = []
    mode = "TRACK"
    modes = [0]
    brake_start = None
    predicted_brake_clearance = None
    status = "达到设定仿真时长"
    initial_stop_clearance = braking_preview(state, steering)
    stop_clearances = [initial_stop_clearance]
    shield_interventions = 0
    initial_backup_feasible = initial_stop_clearance >= STOP_GUARD_MARGIN
    print(f"初始备份制动净间距：{initial_stop_clearance:.3f} m；"
          f"控制保护阈值：{STOP_GUARD_MARGIN:.3f} m")
    if not initial_backup_feasible:
        # 不能宣称所有转向策略均无解；只说明本例备份制动从初始状态已不安全。
        mode = "BRAKE"
        modes[0] = 2
        status = "初始状态不满足备份安全停车条件，立即最大制动（未完成通行）"
        print(status)
    replan_steps = int(round(REPLAN_DT / CONTROL_DT))
    if not np.isclose(replan_steps * CONTROL_DT, REPLAN_DT):
        raise ValueError("重规划周期应为控制周期的整数倍")
    pending = None
    latency_events = []
    old_plan_control_steps = 0
    rejected_late_plans = 0
    delayed_activations = 0
    # 初始规划视为车辆开始运动前已装载；此项假设单独记录，不把启动耗时藏进实时保证。
    startup_plan_preloaded = True

    if show_animation:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rx, ry, "k:", label="Road reference")
        ax.plot(*goal_xy, "g*", markersize=12, label="Goal (rear axle)")
        for i, obstacle in enumerate(planner.OBSTACLES):
            ax.add_patch(plt.Circle(obstacle, planner.ROBOT_RADIUS,
                                   fill=False, color="gray", linestyle=":"))
            ax.plot(*obstacle, "kx", label="Obstacles" if i == 0 else None)
        planned_line, = ax.plot([], [], "r--", label="Current plan")
        actual_line, = ax.plot([], [], "b-", label="Actual trajectory")
        vehicle, = ax.plot([x], [y], "bo")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)
        ax.legend()

    for step in range(int(round(TOTAL_TIME / CONTROL_DT))):
        simulation_time = step * CONTROL_DT
        if pending is not None and step >= pending["ready_step"]:
            candidate = pending["path"]
            candidate_t, candidate_x, candidate_y, candidate_v, candidate_a = unpack_path(candidate)
            candidate_age = simulation_time - pending["started_t"]
            expected = np.array([np.interp(candidate_age, candidate_t, candidate_x),
                                 np.interp(candidate_age, candidate_t, candidate_y)])
            mismatch = float(np.linalg.norm(state[:2] - expected))
            speed_mismatch = abs(float(state[3]) - float(np.interp(candidate_age, candidate_t, candidate_v)))
            prelim = (candidate_age+HANDOFF_PREVIEW_S <= candidate_t[-1]+1e-9
                     and mismatch <= TRACKING_ERROR_LIMIT
                     and speed_mismatch <= 0.5
                     and polyline_distance(state[:2], candidate_x, candidate_y)
                     <= TRACKING_ERROR_LIMIT
                     and braking_preview(state, steering) >= STOP_GUARD_MARGIN)
            if prelim:
                valid, handoff_reason = handoff_preview(candidate, candidate_age, state, steering)
            else:
                valid, handoff_reason = False, "state_or_horizon"
            latency_events.append({"start_t": pending["started_t"],
                                   "ready_t": simulation_time,
                                   "measured_ms": pending["ms"],
                                   "effective_ms": pending["effective_ms"],
                                   "age_s": candidate_age,
                                   "position_mismatch_m": mismatch,
                                   "speed_mismatch_m_s": speed_mismatch,
                                   "handoff_reason": handoff_reason,
                                   "accepted": bool(valid)})
            if valid and mode in ("TRACK", "SLOW"):
                path = candidate
                pt, px, py, pv, pa = candidate_t, candidate_x, candidate_y, candidate_v, candidate_a
                plan_start = pending["started_t"]
                mode = "PARK" if pending["goal"] else "TRACK"
                if mode == "PARK":
                    goal_start = simulation_time
                delayed_activations += 1
                plans.append({"t": simulation_time, "s": pending["s"],
                              "d": pending["d"], "counts": pending["counts"],
                              "cost": path.cf, "ms": pending["ms"],
                              "recovery": pending["recovery"],
                              "retimed": getattr(path, "retimed", False),
                              "start_mismatch": mismatch,
                              "steering_rate": path_steering_rate(path),
                              "required_steer": float(np.max(np.abs(np.arctan(
                                  WHEELBASE * np.asarray(path.c))))),
                              "max_accel": float(np.max(np.abs(path.a))),
                              "max_lateral_accel": float(np.max(
                                  np.asarray(path.v)**2*np.abs(path.c))),
                              "clearance": trajectory_clearance(px, py),
                              "minimum_stop_clearance": path.minimum_stop_clearance,
                              "model_end_state": path.model_end_state,
                              "goal_path": path.goal_path,
                              "rollout": getattr(path, "rollout", False)})
                if show_animation:
                    planned_line.set_data(px, py)
            else:
                rejected_late_plans += 1
                if mode == "SLOW" and path is not None:
                    mode = "TRACK"
            pending = None
        old_remaining = (float(pt[-1]-(simulation_time-plan_start))
                         if path is not None else float("inf"))
        normal_replan = step % replan_steps == 0
        urgent_replan = (fixed_delay_ms != 0 and path is not None
                         and old_remaining <= URGENT_REPLAN_REMAINING_S
                         and step % max(1, replan_steps//2) == 0)
        if (mode in ("TRACK", "SLOW") and pending is None
                and (normal_replan or urgent_replan)):
            try:
                sampled_delay_ms = (float(delay_schedule_ms(len(attempts)))
                                    if delay_schedule_ms is not None
                                    else fixed_delay_ms)
                if (sampled_delay_ms is not None
                        and (not np.isfinite(sampled_delay_ms)
                             or sampled_delay_ms < 0)):
                    raise ValueError("规划结果延迟必须为非负有限值")
                started = clock.perf_counter()
                sc, dc = actual_to_frenet(
                    csp, state, acceleration, steering, grid_s, grid_xy
                )
                # 保留上一条空间曲线的末端作为候选，再从真实状态重建。
                # 避免每轮都强迫末端前移而破坏尚可执行的曲线。
                if path is not None and getattr(path, 'retimed', False):
                    PREVIOUS_GEOMETRY_ENDPOINT = (path.s[-1], path.d[-1])
                elif (PREVIOUS_GEOMETRY_ENDPOINT is not None
                      and PREVIOUS_GEOMETRY_ENDPOINT[0] <= sc[0]+1.0):
                    PREVIOUS_GEOMETRY_ENDPOINT = None
                current_backup_gap = braking_preview(state, steering)
                if (short_horizon_enabled and fixed_delay_ms != 0
                        and current_backup_gap < 14.0):
                    MIN_PLAN_HORIZON_ACTIVE = ((sampled_delay_ms or 0)/1000.0
                                               + HANDOFF_PREVIEW_S + 0.1)
                else:
                    MIN_PLAN_HORIZON_ACTIVE = (0.0 if fixed_delay_ms == 0 else
                        max(MIN_PLAN_HORIZON,
                            (sampled_delay_ms or 0)/1000.0 + HANDOFF_PREVIEW_S))
                nearest_gap = float(np.min(np.linalg.norm(
                    np.asarray(planner.OBSTACLES)-state[:2], axis=1))-planner.ROBOT_RADIUS)
                response_distance = ((2.0+PLAN_STEER/max(PLAN_STEER_RATE, 1e-6))
                                     * state[3]+SAFETY_MARGIN)
                recovery_gap_limit = (max(RECOVERY_TRIGGER_CLEARANCE, response_distance)
                                      if USE_SPEED_GUIDANCE else
                                      max(RECOVERY_TRIGGER_CLEARANCE,
                                          2.0*state[3]+SAFETY_MARGIN))
                obstacle_gap_limit = max(12.0, response_distance) if USE_SPEED_GUIDANCE else 12.0
                if (not recovery_active and sc[0] > 10.0
                        and current_backup_gap < recovery_gap_limit
                        and nearest_gap < obstacle_gap_limit):
                    recovery_active = True
                    recovery_start = simulation_time
                    recovery_start_s = float(sc[0])
                    recovery_events.append({"t": simulation_time, "event": "enter", "s": float(sc[0]),
                                            "reason": "parking_margin_tightening"})
                    print(f"t={simulation_time:.2f} s 停车余量收紧至 {current_backup_gap:.2f} m，提前低速规划。")
                if (recovery_active and sc[0]-recovery_start_s >= RECOVERY_EXIT_PROGRESS
                        and current_backup_gap > RECOVERY_TRIGGER_CLEARANCE and nearest_gap > 4.0):
                    recovery_active = False
                    recovery_events.append({"t": simulation_time, "event": "exit", "s": float(sc[0])})
                    print(f"t={simulation_time:.2f} s 通过局部困难路段，恢复常规速度目标。")
                new_path = None
                approach_distance = max(GOAL_APPROACH_DISTANCE,
                                        state[3]**2/(2*2.0) + 5.0)
                if goal_s - sc[0] <= approach_distance:
                    new_path, goal_counts = goal_planning(csp, sc, dc, goal_s)
                if new_path is not None:
                    counts = goal_counts
                    if step == 0 or fixed_delay_ms == 0:
                        mode = "PARK"
                        goal_start = simulation_time
                        print(f"t={simulation_time:.2f} s 进入终点停车；"
                              f"剩余弧长 {goal_s-sc[0]:.3f} m；"
                              f"停车轨迹时长 {new_path.t[-1]:.2f} s。")
                else:
                    if recovery_active:
                        new_path, counts = speed_aware_planning(csp, sc, dc, grid_s, grid_xy, recovery=True)
                    else:
                        new_path, counts = speed_aware_planning(csp, sc, dc, grid_s, grid_xy)
                    if new_path is None and not recovery_active:
                        recovery_active = True
                        recovery_start = simulation_time
                        recovery_start_s = float(sc[0])
                        recovery_events.append({"t": simulation_time, "event": "enter", "s": float(sc[0])})
                        print(f"t={simulation_time:.2f} s 常规规划无解，降低目标速度并重新采样。")
                        new_path, recovery_counts = speed_aware_planning(csp, sc, dc, grid_s, grid_xy, recovery=True)
                        counts = {key: counts[key]+recovery_counts[key] for key in counts}
                    if new_path is not None and (step == 0 or fixed_delay_ms == 0):
                        mode = "TRACK"
                        status = "达到设定仿真时长"
                    if new_path is not None:
                        new_path.goal_path = False
                elapsed_ms = 1000.0 * (clock.perf_counter() - started)
                effective_ms = (elapsed_ms if sampled_delay_ms is None
                                else float(sampled_delay_ms))
                attempts.append({"t": simulation_time, "counts": counts,
                                 "ms": elapsed_ms, "success": new_path is not None,
                                 "applied_delay_ms": effective_ms,
                                 "recovery": recovery_active,
                                 "old_remaining_s": (float(pt[-1]-(simulation_time-plan_start))
                                                     if path is not None else None),
                                 "backup_clearance_m": current_backup_gap,
                                 "speed_m_s": float(state[3]),
                                 "new_horizon_s": (float(new_path.t[-1])
                                                   if new_path is not None else None)})
                if new_path is None:
                    if path is not None and mode == "TRACK" and fixed_delay_ms != 0:
                        # 旧轨迹仍有足够时间则继续跟踪；接近末端才提前减速。
                        remaining = attempts[-1]["old_remaining_s"]
                        slow_threshold = max(.6, effective_ms/1000.0 + HANDOFF_PREVIEW_S)
                        if remaining <= slow_threshold:
                            mode = "SLOW"
                            status = "新规划无解，旧轨迹即将到期，提前减速重试"
                            print(f"t={simulation_time:.2f} s 无新轨迹；旧轨迹剩余 "
                                  f"{remaining:.2f} s，提前减速重试。")
                        else:
                            print(f"t={simulation_time:.2f} s 无新轨迹；旧轨迹剩余 "
                                  f"{remaining:.2f} s，继续逐步安全检查并重试。")
                    elif (state[3] > RECOVERY_MIN_SPEED
                            and simulation_time-recovery_start < RECOVERY_MAX_DURATION):
                        mode = "SLOW"
                        status = "低速重采样仍无解，保持实际转角减速并继续重规划"
                        print(f"t={simulation_time:.2f} s 暂无可执行轨迹，减速后再次规划。")
                    else:
                        mode = "BRAKE"
                        status = "减速恢复仍无解，安全制动停车（未完成通行）"
                        print(f"t={simulation_time:.2f} s 恢复失败，进入后备制动。")
                    print("失败时分类：", counts)
                elif step > 0 and effective_ms > 0:
                    # 规划在快照上同步计算，但结果仅在测得的计算时间之后交给控制器。
                    # 仿真在等待期按 20 ms 步长继续跟踪旧轨迹。
                    ready_step = step + max(1, int(np.ceil(effective_ms / (1000*CONTROL_DT))))
                    pending = {"path": new_path, "started_t": simulation_time,
                               "ready_step": ready_step, "ms": elapsed_ms,
                               "effective_ms": effective_ms,
                               "counts": counts, "recovery": recovery_active,
                               "s": sc[0], "d": dc[0],
                               "goal": bool(new_path.goal_path)}
                    if path is None:
                        mode = "SLOW"
                else:
                    path = new_path
                    pt, px, py, pv, pa = unpack_path(path)
                    if pt[-1] < REPLAN_DT:
                        raise RuntimeError("剩余轨迹过短")
                    plan_start = simulation_time
                    start_mismatch = float(np.hypot(px[0] - state[0], py[0] - state[1]))
                    plans.append({
                        "t": simulation_time, "s": sc[0], "d": dc[0],
                        "counts": counts, "cost": path.cf, "ms": elapsed_ms,
                        "recovery": recovery_active,
                        "retimed": getattr(path, 'retimed', False),
                        "start_mismatch": start_mismatch,
                        "steering_rate": path_steering_rate(path),
                        "required_steer": float(np.max(np.abs(np.arctan(
                            WHEELBASE * np.asarray(path.c)
                        )))),
                        "max_accel": float(np.max(np.abs(path.a))),
                        "max_lateral_accel": float(np.max(
                            np.asarray(path.v)**2 * np.abs(path.c)
                        )),
                        "clearance": trajectory_clearance(px, py),
                        "minimum_stop_clearance": path.minimum_stop_clearance,
                        "model_end_state": path.model_end_state,
                        "goal_path": path.goal_path,
                        "rollout": getattr(path, "rollout", False),
                    })
                    if len(plans) == 1:
                        print("第一轮候选分类：", counts)
                    if path.model_end_state and sum(p["model_end_state"] for p in plans) == 1:
                        print(f"t={simulation_time:.2f} s 使用车辆模型生成末端状态，找到 {counts['ok']} 条可执行轨迹。")
                    if getattr(path, "rollout", False) and sum(p["rollout"] for p in plans) == 1:
                        print(f"t={simulation_time:.2f} s 多项式连接无解，使用转向速度受限的车辆预测轨迹。")
                    if show_animation:
                        planned_line.set_data(px, py)
            except RuntimeError as exc:
                mode = "BRAKE"
                status = str(exc) + "，制动停车（未完成通行）"

        if mode == "TRACK":
            age = simulation_time - plan_start
            if fixed_delay_ms != 0 and age >= pt[-1] - 1e-9:
                mode = "BRAKE"
                status = "旧轨迹到达时间末端且新规划尚未可用，后备制动（未完成通行）"
        if mode in ("TRACK", "PARK"):
            age = simulation_time - plan_start
            if pending is not None:
                old_plan_control_steps += 1
            target_speed = np.interp(age, pt, pv)
            acceleration = float(np.clip(
                np.interp(age, pt, pa) + 1.5*(target_speed - state[3]),
                -MAX_ACCEL, MAX_ACCEL,
            ))
            if (USE_SPEED_GUIDANCE and mode == "TRACK"
                    and not getattr(path, 'retimed', False)):
                guide = active_speed_cap(path, age)
                guide_accel = float(np.clip(guide-state[3], -PROFILE_DECEL, MAX_ACCEL))
                acceleration = min(acceleration, guide_accel)
                target_speed = min(target_speed, guide)
            if mode == "PARK" and age >= pt[-1]:
                # 清除离散积分留下的微小残余速度，不修改车辆位置。
                acceleration = -min(MAX_ACCEL, state[3]/CONTROL_DT)
            goal_distance = float(np.linalg.norm(state[:2] - goal_xy))
            if (mode == "PARK" and goal_distance <= GOAL_POSITION_TOLERANCE
                    and state[3] < 0.3):
                # 近零速时目标点距离过小，保持末端曲率避免追点角度发散。
                proposed_steering = float(np.arctan(WHEELBASE*path.c[-1]))
                requested = proposed_steering
            elif getattr(path, 'retimed', False):
                proposed_steering, requested = timed_curve_tracking(path, age, state)
            else:
                proposed_steering, requested = pursuit(*state[:3], px, py, state[3])
            if abs(requested) > MAX_STEER + 1e-6:
                mode = "BRAKE"
                status = "控制器需要超限转角，制动停车（未完成通行）"
            elif polyline_distance(state[:2], px, py) > TRACKING_ERROR_LIMIT:
                mode = "BRAKE"
                status = "跟踪偏差超过预留间距，制动停车（未完成通行）"
            else:
                requested = proposed_steering
        if mode == "SLOW":
            acceleration = -RECOVERY_DECEL
            requested = steering
            target_speed = 0.0
            if state[3] <= RECOVERY_MIN_SPEED:
                mode = "BRAKE"
                status = "恢复速度过低仍无可执行轨迹，安全停车（未完成通行）"
        if mode in ("TRACK", "PARK", "SLOW"):
            # 先模拟将要执行的限速转角与加速度，再检查从步末实际状态刹停。
            next_delta = steering_actuator(steering, requested, CONTROL_DT)
            preview_state = advance(state, acceleration,
                                    0.5*(steering+next_delta), CONTROL_DT)
            backup_gap = braking_preview(preview_state, next_delta)
            step_gap = trajectory_clearance(
                [state[0], preview_state[0]], [state[1], preview_state[1]])
            lateral_bound = max(state[3], preview_state[3])**2*abs(
                np.tan(max(abs(steering), abs(next_delta))))/WHEELBASE
            if (backup_gap < STOP_GUARD_MARGIN or step_gap < STOP_GUARD_MARGIN
                    or lateral_bound > MAX_LATERAL_ACCEL+1e-6):
                mode = "BRAKE"
                shield_interventions += 1
                status = "下一控制步会失去安全停车余量或超限，提前制动（未完成通行）"
                print(f"t={simulation_time:.2f} s 控制保护触发；"
                      f"拟执行步末备份净间距 {backup_gap:.3f} m。")
        if mode == "BRAKE":
            if brake_start is None:
                brake_start = simulation_time
                predicted_brake_clearance = braking_preview(state, steering)
                print(f"制动起始时刻：{brake_start:.2f} s；预测制动净间距：{predicted_brake_clearance:.3f} m")
                print("制动保持进入制动前的实际转角，不再执行旧规划轨迹。")
            acceleration = -MAX_ACCEL
            requested = steering
            target_speed = 0.0
        previous_position = state[:2].copy()
        previous_steering = steering
        steering = steering_actuator(previous_steering, requested, CONTROL_DT)
        steering_commands.append(requested)
        rate_limited.append(abs(steering-requested) > 1e-8)
        # 实际转角在本步内按线性斜坡变化；用中间转角积分运动。
        # 下一轮 Frenet 反馈使用步末实际转角，不能使用尚未达到的命令角。
        applied_steering = 0.5*(previous_steering+steering)
        state = advance(state, acceleration, applied_steering, CONTROL_DT)
        stop_clearances.append(braking_preview(state, steering))
        if mode in ("TRACK", "PARK") and USE_SPEED_GUIDANCE:
            speed_caps.append(active_speed_cap(path, age+CONTROL_DT))
        elif mode in ("TRACK", "PARK"):
            speed_caps.append(float(np.interp(age+CONTROL_DT, pt, pv)))
        else:
            speed_caps.append(np.nan)
        if mode in ("TRACK", "PARK"):
            next_age = age + CONTROL_DT
            errors.append(polyline_distance(state[:2], px, py))
            desired_position = np.array([
                np.interp(next_age, pt, px), np.interp(next_age, pt, py)
            ])
            timed_errors.append(float(np.linalg.norm(state[:2] - desired_position)))
            desired_speed = np.interp(next_age, pt, pv)
            target_speeds.append(min(desired_speed, speed_caps[-1])
                                 if USE_SPEED_GUIDANCE and mode == "TRACK"
                                 else desired_speed)
        else:
            # 制动阶段没有活动跟踪任务，不能将其记成零跟踪误差。
            errors.append(np.nan)
            timed_errors.append(np.nan)
            target_speeds.append(0.0)
        if mode == "BRAKE" and state[3] <= 1e-8:
            mode = "STOP"
        if (mode == "PARK" and next_age >= pt[-1]
                and state[3] <= 1e-8):
            goal_reached = (np.linalg.norm(state[:2]-goal_xy)
                            <= GOAL_POSITION_TOLERANCE
                            and state[3] <= GOAL_SPEED_TOLERANCE)
            mode = "STOP"
            status = ("到达终点并完成停车" if goal_reached
                      else "终点轨迹执行完毕，但停车位置超出容差")
        modes.append({"TRACK": 0, "PARK": 1, "BRAKE": 2, "STOP": 3, "SLOW": 4}[mode])
        times.append((step + 1) * CONTROL_DT)
        states.append(state.copy())
        steerings.append(steering)
        accelerations.append(acceleration)
        saturation.append(abs(requested) > MAX_STEER + 1e-10)

        step_positions = np.array([previous_position, state[:2]])
        if trajectory_clearance(step_positions[:, 0], step_positions[:, 1]) <= 0.0:
            status = "实际车辆触及圆形碰撞边界，终止仿真"
            break
        if show_animation and step % 5 == 0:
            if not plt.fignum_exists(fig.number):
                status = "用户关闭动画窗口"
                break
            history = np.asarray(states)
            actual_line.set_data(history[:, 0], history[:, 1])
            vehicle.set_data([state[0]], [state[1]])
            ax.set_title(f"Steering-rate constrained | {mode} | t={times[-1]:.2f} s")
            plt.pause(0.001)
        if mode == "STOP":
            break

    times, states = np.asarray(times), np.asarray(states)
    errors, timed_errors = np.asarray(errors), np.asarray(timed_errors)
    target_speeds = np.asarray(target_speeds)
    # 检查实际位置之间的折线，而非只检查离散位置。
    clearance = min(
        polyline_distance(obstacle, states[:, 0], states[:, 1])
        for obstacle in planner.OBSTACLES
    ) - planner.ROBOT_RADIUS if len(states) > 1 else float("nan")
    plan_ms = np.array([p["ms"] for p in attempts])
    mismatch = max((p["start_mismatch"] for p in plans), default=float("nan"))
    final_s = reference_projection(
        csp, states[-1, 0], states[-1, 1], grid_s, grid_xy
    )[0]
    print("\n========== 转向速度限制、完整路线与终点停车 ==========")
    print("结束原因：", status)
    print(f"实际模拟时长：{times[-1]:.2f} s；成功规划次数：{len(plans)}")
    print(f"重规划尝试次数：{len(attempts)}；无解次数：{sum(not p['success'] for p in attempts)}")
    print(f"使用模型预测末端状态的成功规划次数：{sum(p['model_end_state'] for p in plans)}")
    print(f"直接使用车辆模型预测轨迹的次数：{sum(p['rollout'] for p in plans)}")
    print(f"最大规划起点位置偏差：{mismatch:.6f} m")
    tracking = np.isfinite(errors)
    if np.any(tracking):
        print(f"跟踪阶段最大路径距离误差：{errors[tracking].max():.3f} m")
        print(f"跟踪阶段路径距离误差 RMSE：{np.sqrt(np.mean(errors[tracking]**2)):.3f} m")
        print(f"跟踪阶段时间对应位置误差 RMSE：{np.sqrt(np.mean(timed_errors[tracking]**2)):.3f} m")
        print(f"跟踪阶段速度 RMSE：{np.sqrt(np.mean((states[tracking, 3]-target_speeds[tracking])**2)):.3f} m/s")
    print(f"实际轨迹折线的圆形碰撞模型净间距：{clearance:.3f} m")
    print("碰撞模型检查：", "通过" if clearance > 0 else "未通过")
    print(f"最终速度：{states[-1, 3]:.3f} m/s；最终状态：{mode}")
    print(f"最终位置：x={states[-1, 0]:.3f} m，y={states[-1, 1]:.3f} m")
    print(f"沿参考线进度：{final_s:.3f}/{csp.s[-1]:.3f} m")
    goal_error = float(np.linalg.norm(states[-1, :2]-goal_xy))
    print(f"目标后轴位置：x={goal_xy[0]:.3f} m，y={goal_xy[1]:.3f} m")
    print(f"终点位置误差：{goal_error:.4f} m；允许 {GOAL_POSITION_TOLERANCE:.2f} m")
    print("终点停车检查：", "通过" if goal_reached else "未通过")
    print(f"实际轨迹是否保留 {SAFETY_MARGIN:.1f} m 间距：", "是" if clearance >= SAFETY_MARGIN else "否")
    if len(plan_ms):
        print(f"规划耗时均值/最大：{plan_ms.mean():.1f}/{plan_ms.max():.1f} ms")
    if len(steerings):
        print(f"最大绝对转角：{np.rad2deg(np.max(np.abs(steerings))):.2f}°")
        print(f"转角限幅占比：{100*np.mean(saturation):.1f}%")
        actual_rates = np.diff(np.r_[initial_steering, steerings])/CONTROL_DT
        print(f"最大实际转角变化速度：{np.rad2deg(np.max(np.abs(actual_rates))):.2f}°/s；"
              f"上限 {np.rad2deg(MAX_STEER_RATE):.1f}°/s")
        print(f"转向速度限幅占比：{100*np.mean(rate_limited):.1f}%")
        print("实际转向速度检查：", "通过" if np.max(np.abs(actual_rates))
              <= MAX_STEER_RATE+1e-6 else "未通过")
        peak_speeds = np.maximum(states[:-1, 3], states[1:, 3])
        actual_lateral_accel = peak_speeds**2*np.abs(np.tan(steerings))/WHEELBASE
        print(f"车辆模型最大横向加速度：{actual_lateral_accel.max():.3f} m/s²")
    if plans:
        print(f"选中轨迹最大采样转向速度："
              f"{np.rad2deg(max(p['steering_rate'] for p in plans)):.2f}°/s；"
              f"规划上限 {np.rad2deg(PLAN_STEER_RATE):.1f}°/s")
        print(f"选中规划轨迹的最大需求转角：{np.rad2deg(max(p['required_steer'] for p in plans)):.2f}°")
        print(f"选中规划轨迹的最大纵向/横向加速度：{max(p['max_accel'] for p in plans):.3f}/{max(p['max_lateral_accel'] for p in plans):.3f} m/s²")
        print(f"选中规划轨迹的最小净间距：{min(p['clearance'] for p in plans):.3f} m")
    print(f"候选轨迹因无法安全刹停被拒绝：{sum(a['counts'].get('stopping', 0) for a in attempts)} 条次")
    print(f"实际控制保护触发次数：{shield_interventions}")
    print(f"空间路径配置速度的成功规划次数：{sum(p['retimed'] for p in plans)}")
    print(f"进入减速恢复次数：{sum(e['event']=='enter' for e in recovery_events)}；"
          f"恢复采样成功次数：{sum(p['recovery'] for p in plans)}")
    print(f"实际状态最小备份制动净间距：{min(stop_clearances):.3f} m")
    print(f"规划结果延迟启用 {delayed_activations} 次；迟到后拒绝 {rejected_late_plans} 次；"
          f"等待期间执行旧轨迹 {old_plan_control_steps} 个控制步。")
    print("初始规划视为起步前已装载；计算耗时按控制周期向上取整。"
          "这是测得耗时的确定性回放，不是并行线程或实时系统保证。")
    print("停车避免碰撞与完成通行分别评价；本次加入了转角速度受限的执行器；未加入轮胎动力学或矩形车身。")
    if status == "达到设定仿真时长":
        print("本次运行到设定时长，车辆仍可能在运动，不代表已到达终点或停车。")

    _, axes = plt.subplots(6, 1, figsize=(10, 14))
    axes[0].plot(times, errors, label="Distance to active plan")
    axes[0].plot(times, timed_errors, "--", label="Time-matched position error")
    axes[0].set_ylabel("Error [m]")
    axes[0].legend()
    axes[1].plot(times, states[:, 3]*3.6, label="Actual")
    axes[1].plot(times, target_speeds*3.6, "r--", label="Target")
    axes[1].plot(times, np.asarray(speed_caps)*3.6, "g:", label="Preview speed guide")
    axes[1].set_ylabel("Speed [km/h]")
    axes[1].legend()
    axes[2].plot(times[1:], np.rad2deg(steering_commands), "--", label="Command")
    axes[2].plot(times[1:], np.rad2deg(steerings), label="Actual")
    axes[2].legend()
    axes[2].set_ylabel("Steering [deg]")
    axes[4].plot([p["t"] for p in attempts], [p["counts"]["ok"] for p in attempts], ".-")
    axes[4].set_ylabel("Feasible plans")
    if steerings:
        axes[3].plot(times[1:], np.rad2deg(actual_rates), label="Actual steering rate")
    axes[3].axhline(np.rad2deg(MAX_STEER_RATE), color="red", linestyle="--")
    axes[3].axhline(-np.rad2deg(MAX_STEER_RATE), color="red", linestyle="--")
    axes[3].set_ylabel("Steering rate [deg/s]")
    axes[3].legend()
    state_axis = axes[4].twinx()
    state_axis.step(times, modes, where="post", color="orange", linestyle="--")
    state_axis.set_yticks([0, 1, 2, 3, 4], ["TRACK", "PARK", "BRAKE", "STOP", "SLOW"])
    state_axis.set_ylim(-0.2, 4.2)
    axes[5].plot(times, stop_clearances, label="Backup stopping clearance")
    axes[5].axhline(STOP_GUARD_MARGIN, color="orange", linestyle="--",
                    label="Control guard margin")
    axes[5].axhline(SAFETY_MARGIN, color="red", linestyle=":",
                    label="Required body clearance")
    axes[5].set_ylabel("Stop clearance [m]")
    axes[5].set_xlabel("Time [s]")
    axes[5].legend()
    for summary_axis in axes:
        summary_axis.grid(True)
    plt.tight_layout()
    if show_animation:
        if plt.fignum_exists(fig.number):
            actual_line.set_data(states[:, 0], states[:, 1])
            vehicle.set_data([state[0]], [state[1]])
            ax.set_title(f"Curve-speed preview | {mode} | t={times[-1]:.2f} s")
        plt.show()
    else:
        plt.close("all")
    return {"status": status, "times": times, "states": states,
            "errors": errors, "plans": plans, "clearance": clearance,
            "acceleration": acceleration, "steering": steering, "attempts": attempts,
            "mode": mode, "brake_start": brake_start,
            "steerings": np.asarray(steerings), "modes": np.asarray(modes),
            "accelerations": np.asarray(accelerations), "final_s": final_s,
            "goal_reached": goal_reached, "goal_error": goal_error,
            "goal_xy": goal_xy, "goal_start": goal_start,
            "target_speeds": target_speeds, "timed_errors": timed_errors,
            "speed_caps": np.asarray(speed_caps),
            "profile_s": SPEED_PROFILE.s, "profile_speed": SPEED_PROFILE.speed,
            "profile_raw": SPEED_PROFILE.raw, "profile_kappa": SPEED_PROFILE.kappa,
            "profile_steering_gradient": SPEED_PROFILE.steering_gradient,
            "speed_guidance_enabled": USE_SPEED_GUIDANCE,
            "initial_preview_stress": INITIAL_PREVIEW_STRESS,
            "guidance_stress_threshold": GUIDANCE_STRESS_THRESHOLD,
            "recovery_events": recovery_events,
            "initial_stop_clearance": initial_stop_clearance,
            "initial_backup_feasible": initial_backup_feasible,
            "latency_events": latency_events,
            "old_plan_control_steps": old_plan_control_steps,
            "rejected_late_plans": rejected_late_plans,
            "delayed_activations": delayed_activations,
            "startup_plan_preloaded": startup_plan_preloaded,
            "fixed_delay_ms": fixed_delay_ms,
            "stop_clearances": np.asarray(stop_clearances),
            "shield_interventions": shield_interventions,
            "predicted_brake_clearance": predicted_brake_clearance,
            "initial_steering": initial_steering,
            "steering_commands": np.asarray(steering_commands),
            "steering_rates": np.diff(np.r_[initial_steering, steerings])/CONTROL_DT,
            "rate_limited": np.asarray(rate_limited)}


if __name__ == "__main__":
    main()
