"""
Модуль 6. Монте-Карло баллистика неуправляемого снаряда (6-DoF, твёрдое тело).

Физическая модель:
  - 13 уравнений состояния:
      [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
  - Интегратор: классический RK4, фиксированный шаг dt.
  - Ориентация: кватернионы (нет проблемы шарнирного замка / gimbal lock).
  - Аэродинамика: Cd и Cl из CFD-таблиц, интерполяция по углу атаки.
    Для неуправляемого тела управляющий трим pitch_trim = 0.
  - Моменты: статический момент тангажа Cm_alpha, демпфирование по q/p/r,
    плечо аэродинамической силы (смещение CP от CG).
  - Атмосфера: экспоненциальная модель rho(y) = rho0 * exp(-y / H0).
  - Стохастика: случайный изотропный ветер у земли (равномерное распределение
    скорости и направления) — одно значение на весь полёт каждого снаряда,
    что соответствует модели «постоянный средний ветер с неопределённостью».

Ссылки:
  - Mitchell Stolk, «6-DoF-Lite Guided Projectile», MIT License, Dec 2025
    (адаптировано: убрано пропорциональное наведение, добавлен пуск с y=0)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Интерфейс модели (обязательные функции для загрузчика main.py)
# ---------------------------------------------------------------------------

def get_name() -> str:
    """Возвращает название модели для отображения в списке."""
    return "6. Неуправляемый снаряд (6-DoF, Монте-Карло)"


def get_info() -> dict:
    """Метаданные модели для интерфейса."""
    return {
        "description": (
            "Физический симулятор баллистического снаряда (6 степеней свободы). "
            "Тело — вращающееся твёрдое тело с тензором инерции I. "
            "Пуск с земли (y = 0) под углом к горизонту. "
            "Cd и Cl интерполируются из CFD-таблиц по углу атаки. "
            "Наведение отключено — исследуется естественная баллистика. "
            "Метод Монте-Карло: N пусков с разными случайными ветрами. "
            "На графике: облако траекторий, среднее, эллипсы 1σ / 2σ / 3σ."
        ),
        "formula": (
            r"$\mathbf{s} = [x,y,z,v_x,v_y,v_z,q_w,q_x,q_y,q_z,p,q,r]^T$"
            "\n\n"
            r"$\mathbf{M} = M_\alpha + M_q + (\mathbf{r}_{cp}\times\mathbf{F}_{aero})$"
        ),
        "parameters_info": {
            "N": "Количество пусков Монте-Карло (рек. 40–100)",
            "v0": "Начальная скорость снаряда (м/с)",
            "angle": "Угол возвышения орудия (градусы)",
            "azimuth": "Азимут стрельбы (градусы, 0 = ось X)",
            "Wind_Max": "Макс. скорость случайного ветра у земли (м/с)",
            "m": "Масса снаряда (кг)",
        },
    }


def get_params() -> dict:
    """Параметры по умолчанию."""
    return {
        "N": "60",
        "v0": "250.0",
        "angle": "45.0",
        "azimuth": "0.0",
        "Wind_Max": "6.0",
        "m": "1.5",
    }


# ---------------------------------------------------------------------------
# Константы геометрии снаряда (задаются один раз при загрузке модуля)
# ---------------------------------------------------------------------------

_D_REF: float = 0.05
_S_REF: float = 0.25 * np.pi * _D_REF ** 2   # Площадь миделя, м²
_C_REF: float = 0.50                           # Продольный референс, м
_B_REF: float = 0.15                           # Поперечный референс, м

_I_BODY: np.ndarray = np.diag([8.5e-4, 0.03, 0.03])  # Тензор инерции, кг·м²
_I_INV: np.ndarray  = np.linalg.inv(_I_BODY)          # Предвычисленный обратный тензор

_R_CP_CG: np.ndarray = np.array([0.03, 0.0, 0.0])     # Плечо CP–CG, м

# Коэффициенты аэродинамического демпфирования угловых скоростей
_CM_Q: float = -20.0   # Демпфирование тангажа (pitch)
_CL_P: float = -10.0   # Демпфирование крена   (roll)
_CN_R: float = -20.0   # Демпфирование рысканья (yaw)

# Атмосфера: стандартная экспоненциальная модель
_RHO0: float  = 1.225    # Плотность у земли, кг/м³
_H_ATM: float = 8430.0   # Масштабная высота, м


# ---------------------------------------------------------------------------
# CFD-таблицы аэродинамических коэффициентов
# ---------------------------------------------------------------------------

_AOA_GRID: np.ndarray = np.array([0.0, 4.0, 6.0, 8.0, 12.0, 16.0])
_CAN_GRID: np.ndarray = np.array([0.0, 4.0, 6.0, 8.0, 12.0, 16.0])

_CD_TABLE: np.ndarray = np.array([
    [0.35, 0.40, 0.50, 0.65, 1.10, 1.80],
    [0.40, 0.50, 0.60, 0.80, 1.30, 2.00],
    [0.45, 0.55, 0.65, 0.85, 1.40, 2.10],
    [0.50, 0.65, 0.80, 1.00, 1.60, 2.30],
    [0.60, 0.75, 0.90, 1.10, 1.70, 2.35],
    [0.70, 0.85, 1.00, 1.20, 1.80, 2.45],
])

_CL_TABLE: np.ndarray = np.array([
    [0.00, 1.10, 1.75, 2.50, 3.90, 5.50],
    [0.05, 1.20, 1.90, 2.60, 4.00, 5.40],
    [0.20, 1.25, 1.95, 2.70, 4.10, 5.30],
    [0.40, 1.35, 2.00, 2.80, 4.20, 5.20],
    [0.50, 1.40, 2.05, 2.70, 4.10, 5.10],
    [0.60, 1.45, 2.10, 2.60, 4.00, 5.00],
])

_CM_ALPHA_AOA: np.ndarray = np.array(
    [0, 4, 6, 8, 12, 16, 25, 45, 60, 90, 180], dtype=float
)
_CM_ALPHA_VAL: np.ndarray = np.array(
    [0, -0.10, -0.15, -0.30, -1, -2, -4, -8, -10, -8, -6], dtype=float
)

# Билинейные интерполяторы — вычисляются один раз при импорте модуля
_CL_INTERP: RegularGridInterpolator = RegularGridInterpolator(
    (_AOA_GRID, _CAN_GRID), _CL_TABLE,
    method='linear', bounds_error=False, fill_value=None,
)
_CD_INTERP: RegularGridInterpolator = RegularGridInterpolator(
    (_AOA_GRID, _CAN_GRID), _CD_TABLE,
    method='linear', bounds_error=False, fill_value=None,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции (приватные — без экспорта)
# ---------------------------------------------------------------------------

def _lookup_cl_cd(alpha_deg: float, trim_deg: float) -> tuple:
    """
    Возвращает (Cl, Cd) из CFD-таблиц для заданных угла атаки и трима.

    Args:
        alpha_deg: Угол атаки, °.
        trim_deg:  Угол управляющего трима, ° (0 для неуправляемого снаряда).

    Returns:
        Кортеж (Cl, Cd) — безразмерные коэффициенты подъёмной силы и сопротивления.
    """
    a = float(np.clip(abs(alpha_deg), 0.0, 16.0))
    t = float(np.clip(abs(trim_deg), 0.0, 16.0))
    return _CL_INTERP([[a, t]]).item(), _CD_INTERP([[a, t]]).item()


def _cm_alpha(alpha_deg: float) -> float:
    """Линейная интерполяция статического момента тангажа Cm(alpha)."""
    return float(np.interp(abs(alpha_deg), _CM_ALPHA_AOA, _CM_ALPHA_VAL))


def _quat_to_rotm(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Матрица поворота R_bw из кватерниона (тело → инерциальная СК).

    Стандартная формула Гамильтона. R_wb = R_bw.T

    Returns:
        ndarray формы (3, 3).
    """
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)  ],
        [2*(qx*qy + qz*qw),      1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)  ],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),      1 - 2*(qx*qx + qy*qy)],
    ], dtype=float)


def _initial_quaternion(alpha_rad: float, beta_rad: float) -> np.ndarray:
    """
    Начальный кватернион ориентации: поворот по yaw (beta) затем pitch (alpha).

    Args:
        alpha_rad: Угол возвышения, рад.
        beta_rad:  Азимут, рад.

    Returns:
        ndarray (4,) — (qw, qx, qy, qz).
    """
    cy, sy = np.cos(beta_rad / 2.0),  np.sin(beta_rad / 2.0)
    cp, sp = np.cos(alpha_rad / 2.0), np.sin(alpha_rad / 2.0)
    # Перемножаем кватернионы: q_yaw * q_pitch
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


# ---------------------------------------------------------------------------
# Правая часть ОДУ (d/dt вектора состояния)
# ---------------------------------------------------------------------------

def _derivatives(state: np.ndarray, p: dict) -> np.ndarray:
    """
    Правая часть системы 13 ОДУ первого порядка для 6-DoF снаряда.

    Вектор состояния state[]:
        0..2  — положение (x, y, z), м
        3..5  — скорость (vx, vy, vz), м/с
        6..9  — кватернион (qw, qx, qy, qz)
        10..12 — угловые скорости тела (p, q, r), рад/с

    Args:
        state: Вектор состояния (13,).
        p:     Параметры среды (rho, m, g, wx, wy, wz).

    Returns:
        ds/dt, ndarray (13,).
    """
    vx, vy, vz       = state[3], state[4], state[5]
    qw, qx, qy, qz   = state[6], state[7], state[8], state[9]
    pr, qr, rr        = state[10], state[11], state[12]

    m   = p['m']
    g   = p['g']
    wx  = p['wx']
    wy  = p['wy']
    wz  = p['wz']

    # Плотность воздуха на текущей высоте (экспоненциальная атмосфера)
    y_pos = float(state[1])
    rho = _RHO0 * np.exp(-max(y_pos, 0.0) / _H_ATM)

    # --- Нормализация кватерниона ---
    q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if q_norm < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        inv_qn = 1.0 / q_norm
        qw *= inv_qn;  qx *= inv_qn;  qy *= inv_qn;  qz *= inv_qn

    R_bw = _quat_to_rotm(qw, qx, qy, qz)
    R_wb = R_bw.T   # обратный поворот: инерциальная → тело

    # --- Относительная скорость снаряда относительно воздуха ---
    v_rel = np.array([vx - wx, vy - wy, vz - wz])
    V = float(np.sqrt(v_rel @ v_rel)) + 1e-12   # скалярная, всегда > 0

    # Угол атаки в связанной СК
    v_body = R_wb @ v_rel
    alpha_deg = float(np.degrees(np.arctan2(v_body[2], v_body[0])))

    # Для неуправляемого снаряда трим = 0
    Cl, Cd = _lookup_cl_cd(alpha_deg, 0.0)

    # Сила лобового сопротивления (направлена против v_rel)
    F_drag = -0.5 * rho * _S_REF * Cd * V * v_rel

    # Подъёмная сила: перп. v_rel в плоскости (v_rel, e_y)
    e_y = np.array([0.0, 1.0, 0.0])
    lift_dir = np.cross(np.cross(v_rel, e_y), v_rel)
    lift_norm = float(np.sqrt(lift_dir @ lift_dir))
    if lift_norm > 1e-12:
        lift_dir /= lift_norm
    else:
        lift_dir = np.zeros(3)
    F_lift = 0.5 * rho * _S_REF * Cl * V * V * lift_dir

    # Суммарная сила → линейные ускорения
    F_total = F_drag + F_lift + np.array([0.0, -m * g, 0.0])
    a_lin = F_total / m

    # --- Аэродинамические моменты ---
    q_dyn = 0.5 * rho * V * V   # динамическое давление, Па

    # Статический момент тангажа + демпфирование по pitch-rate
    q_hat = (qr * _C_REF) / (2.0 * V)
    My = q_dyn * _S_REF * _C_REF * (_cm_alpha(alpha_deg) + _CM_Q * q_hat)

    # Демпфирующие моменты крена и рысканья
    p_hat = (pr * _B_REF) / (2.0 * V)
    r_hat = (rr * _B_REF) / (2.0 * V)
    Mx = q_dyn * _S_REF * _B_REF * (_CL_P * p_hat)
    Mz = q_dyn * _S_REF * _B_REF * (_CN_R * r_hat)

    # Момент плеча CP–CG (сила аэро в связанной СК)
    F_aero_body = R_wb @ (F_drag + F_lift)
    M_cp = np.cross(_R_CP_CG, F_aero_body)

    M_body = np.array([Mx, My, Mz]) + M_cp

    # Уравнения Эйлера: I·dω/dt = M - ω × (I·ω)
    omega = np.array([pr, qr, rr])
    d_omega = _I_INV @ (M_body - np.cross(omega, _I_BODY @ omega))

    # Кинематика кватерниона: dq/dt = ½·Ω(ω)·q
    dqw = 0.5 * (-pr*qx - qr*qy - rr*qz)
    dqx = 0.5 * ( pr*qw + rr*qy - qr*qz)
    dqy = 0.5 * ( qr*qw - rr*qx + pr*qz)
    dqz = 0.5 * ( rr*qw + qr*qx - pr*qy)

    return np.array([
        vx, vy, vz,
        a_lin[0], a_lin[1], a_lin[2],
        dqw, dqx, dqy, dqz,
        d_omega[0], d_omega[1], d_omega[2],
    ])


# ---------------------------------------------------------------------------
# Интегрирование одного полёта методом RK4
# ---------------------------------------------------------------------------

def _simulate_one(
    state0: np.ndarray,
    sim_params: dict,
    dt: float,
    max_steps: int,
) -> np.ndarray:
    """
    Интегрирует один полёт снаряда до касания земли (y ≤ 0).

    Условие останова: высота state[1] ≤ 0 И уже прошло > 2 шагов
    (исключает мгновенный выход на старте).
    Финальная точка — линейная интерполяция на плоскость y = 0.

    Args:
        state0:    Начальный вектор состояния (13,).
        sim_params: Параметры среды.
        dt:        Шаг RK4, с.
        max_steps: Лимит итераций.

    Returns:
        ndarray (K, 3) — координаты (x, y, z) по шагам полёта.
    """
    state = state0.copy()
    traj = [state[:3].copy()]   # начальная точка

    for _ in range(max_steps):
        k1 = _derivatives(state,                   sim_params)
        k2 = _derivatives(state + 0.5 * dt * k1,  sim_params)
        k3 = _derivatives(state + 0.5 * dt * k2,  sim_params)
        k4 = _derivatives(state +        dt * k3,  sim_params)
        state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # state[1] — высота (скаляр), сравнение со скаляром безопасно
        y_new = float(state[1])

        if y_new <= 0.0 and len(traj) > 2:
            # Линейная интерполяция: находим точку пересечения y = 0
            y_prev = float(traj[-1][1])
            dy = y_prev - y_new
            frac = y_prev / dy if abs(dy) > 1e-12 else 0.0
            x_land = float(traj[-1][0]) + frac * (float(state[0]) - float(traj[-1][0]))
            z_land = float(traj[-1][2]) + frac * (float(state[2]) - float(traj[-1][2]))
            traj.append(np.array([x_land, 0.0, z_land]))
            break

        traj.append(state[:3].copy())

    return np.array(traj)   # форма (K, 3)


# ---------------------------------------------------------------------------
# Публичная точка входа, вызывается из main.py
# ---------------------------------------------------------------------------

def calculate(params: dict) -> tuple:
    """
    Ансамблевый расчёт методом Монте-Карло: N пусков 6-DoF снаряда.

    Каждый пуск — независимый случайный вектор ветра (равномерное
    распределение скорости на [0, Wind_Max] и направления на [0, 2π]).

    Args:
        params: Строковый словарь параметров из UI.

    Returns:
        (trajectories, is_3d, label):
          trajectories — list[ndarray(K_i, 3)] — список траекторий;
          is_3d        — True (3D-визуализация в main.py);
          label        — строка для легенды.
    """
    n_runs    = int(params.get('N', 60))
    v0        = float(params.get('v0', 250.0))
    angle_deg = float(params.get('angle', 45.0))
    azim_deg  = float(params.get('azimuth', 0.0))
    wind_max  = float(params.get('Wind_Max', 6.0))
    mass      = float(params.get('m', 1.5))

    alpha = np.radians(angle_deg)
    beta  = np.radians(azim_deg)

    # Начальная скорость в инерциальной СК
    vx0 = v0 * np.cos(alpha) * np.cos(beta)
    vy0 = v0 * np.sin(alpha)
    vz0 = v0 * np.cos(alpha) * np.sin(beta)

    # Начальная ориентация (кватернион по углам пушки)
    q0 = _initial_quaternion(alpha, beta)

    base_params = {
        'm':  mass,
        'g':  9.81,
        'wx': 0.0,
        'wy': 0.0,
        'wz': 0.0,
    }

    dt        = 0.01    # с: шаг интегрирования (хорошо для v ~ 250 м/с)
    max_steps = 20000   # лимит: ~200 с полёта — заведомо больше любой реальной дальности

    rng = np.random.default_rng()
    all_trajectories = []

    for _ in range(n_runs):
        # Случайный ветер: равномерное по скорости и направлению
        w_speed = float(rng.uniform(0.0, wind_max))
        w_dir   = float(rng.uniform(0.0, 2.0 * np.pi))

        sim_p = dict(base_params)
        sim_p['wx'] = w_speed * np.cos(w_dir)
        sim_p['wz'] = w_speed * np.sin(w_dir)

        # Вектор состояния на старте
        s0 = np.zeros(13)
        s0[0:3]  = [0.0, 0.0, 0.0]
        s0[3:6]  = [vx0, vy0, vz0]
        s0[6:10] = q0
        # s0[10:13] = 0 — нет начального вращения

        traj = _simulate_one(s0, sim_p, dt, max_steps)
        all_trajectories.append(traj)

    label = f"6-DoF Монте-Карло (N={n_runs}, Wind ≤ {wind_max} м/с)"
    return all_trajectories, True, label