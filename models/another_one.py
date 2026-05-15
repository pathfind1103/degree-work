"""
Модуль 7. Монте-Карло баллистика неуправляемого снаряда (6-DoF, твёрдое тело).

Физическая модель:
  - 13 уравнений состояния:
      [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
  - Интегратор: RK4 с фиксированным шагом dt.
  - Ориентация: кватернионы (нет gimbal lock).
  - Тело снаряда: цилиндр длиной L с конической головной частью.
    Тензор инерции вычисляется аналитически из геометрии и массы.
  - Аэродинамика: Cd и Cl из CFD-таблиц (билинейная интерполяция по углу
    атаки). Cm_alpha — стабилизирующий момент тангажа. Демпфирование p/q/r.
  - Атмосфера: стандартная экспоненциальная ρ(y) = ρ₀·exp(-y/H₀).
  - Стохастика: случайный изотропный ветер (равномерные скорость и
    направление) — одно значение на весь полёт каждого снаряда.

Снаряд моделируется как ТЕЛО, а не точка:
  - Геометрия: калибр D, длина L, масса m → аналитический тензор инерции.
  - При отрисовке в main.py передаётся список траекторий центра масс (как
    обычно), но тензор I влияет на угловую динамику (вращение тела в полёте).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Интерфейс модели
# ---------------------------------------------------------------------------

def get_name() -> str:
    """Возвращает название модели для отображения в списке."""
    return "7. Неуправляемый снаряд (6-DoF, твёрдое тело, МК)"


def get_info() -> dict:
    """Метаданные модели для интерфейса."""
    return {
        "description": (
            "6-DoF симулятор баллистического снаряда как ТВЁРДОГО ТЕЛА. "
            "Геометрия: цилиндр диаметром D, длиной L с конической головной "
            "частью. Тензор инерции вычисляется аналитически. "
            "Ориентация — кватернионы. Аэродинамика — CFD-таблицы Cd/Cl. "
            "Монте-Карло: N пусков с независимыми случайными ветрами. "
            "Результат: облако траекторий + средняя + эллипсы 1σ/2σ/3σ."
        ),
        "formula": (
            r"$I_{xx} = \frac{m D^2}{8},\quad "
            r"I_{yy} = m\!\left(\frac{D^2}{16}+\frac{L^2}{12}\right)$"
            "\n\n"
            r"$\dot{\omega} = I^{-1}(M - \omega \times (I \cdot \omega))$"
        ),
        "parameters_info": {
            "N": "Количество пусков Монте-Карло (рек. 40–100)",
            "v0": "Начальная скорость снаряда, м/с",
            "angle": "Угол возвышения орудия, градусы",
            "azimuth": "Азимут стрельбы, градусы (0 = ось X)",
            "Wind_Max": "Макс. случайный ветер у земли, м/с",
            "m": "Масса снаряда, кг",
            "D": "Калибр (диаметр) снаряда, м",
            "L": "Длина снаряда, м",
        },
    }


def get_params() -> dict:
    """Параметры по умолчанию: типичный 40-мм снаряд."""
    return {
        "N": "60",
        "v0": "250.0",
        "angle": "45.0",
        "azimuth": "0.0",
        "Wind_Max": "6.0",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
    }


# ---------------------------------------------------------------------------
# Аналитический тензор инерции снаряда-цилиндра
# ---------------------------------------------------------------------------

def _inertia_tensor(m: float, D: float, L: float) -> tuple:
    """
    Вычисляет тензор инерции снаряда, аппроксимированного однородным цилиндром.

    Для однородного цилиндра с массой m, диаметром D, длиной L:
      I_xx (осевой, вокруг оси симметрии) = m * R² / 2
      I_yy = I_zz (поперечные)            = m * (R²/4 + L²/12)

    где R = D / 2 — радиус.

    Args:
        m: Масса, кг.
        D: Диаметр (калибр), м.
        L: Длина, м.

    Returns:
        (I_mat, I_inv) — тензор инерции (3,3) и его обратная матрица.
    """
    R = D / 2.0
    I_xx = 0.5 * m * R * R                          # осевой момент инерции
    I_yy = m * (R * R / 4.0 + L * L / 12.0)         # поперечный
    I_mat = np.diag([I_xx, I_yy, I_yy])
    I_inv = np.linalg.inv(I_mat)
    return I_mat, I_inv


# ---------------------------------------------------------------------------
# CFD-таблицы аэродинамических коэффициентов
# ---------------------------------------------------------------------------

_AOA_GRID = np.array([0.0, 4.0, 6.0, 8.0, 12.0, 16.0])
_CAN_GRID = np.array([0.0, 4.0, 6.0, 8.0, 12.0, 16.0])

_CD_TABLE = np.array([
    [0.35, 0.40, 0.50, 0.65, 1.10, 1.80],
    [0.40, 0.50, 0.60, 0.80, 1.30, 2.00],
    [0.45, 0.55, 0.65, 0.85, 1.40, 2.10],
    [0.50, 0.65, 0.80, 1.00, 1.60, 2.30],
    [0.60, 0.75, 0.90, 1.10, 1.70, 2.35],
    [0.70, 0.85, 1.00, 1.20, 1.80, 2.45],
])

_CL_TABLE = np.array([
    [0.00, 1.10, 1.75, 2.50, 3.90, 5.50],
    [0.05, 1.20, 1.90, 2.60, 4.00, 5.40],
    [0.20, 1.25, 1.95, 2.70, 4.10, 5.30],
    [0.40, 1.35, 2.00, 2.80, 4.20, 5.20],
    [0.50, 1.40, 2.05, 2.70, 4.10, 5.10],
    [0.60, 1.45, 2.10, 2.60, 4.00, 5.00],
])

_CM_AOA_PTS = np.array([0, 4, 6, 8, 12, 16, 25, 45, 60, 90, 180], dtype=float)
_CM_VAL_PTS = np.array([0, -0.10, -0.15, -0.30, -1, -2, -4, -8, -10, -8, -6],
                        dtype=float)

_CL_INTERP = RegularGridInterpolator(
    (_AOA_GRID, _CAN_GRID), _CL_TABLE,
    method='linear', bounds_error=False, fill_value=None,
)
_CD_INTERP = RegularGridInterpolator(
    (_AOA_GRID, _CAN_GRID), _CD_TABLE,
    method='linear', bounds_error=False, fill_value=None,
)

# Геометрические константы аэродинамики (одинаковы для всех пусков,
# пересчитываются внутри calculate() под конкретный калибр D)
_CM_Q  = -20.0   # демпфирование тангажа
_CL_P  = -10.0   # демпфирование крена
_CN_R  = -20.0   # демпфирование рысканья
_H_ATM = 8430.0  # масштабная высота атмосферы, м
_RHO0  = 1.225   # плотность у земли, кг/м³


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _lookup_cl_cd(alpha_deg: float) -> tuple:
    """CFD-таблица: (Cl, Cd) по углу атаки (трим=0 для неуправляемого)."""
    a = float(np.clip(abs(alpha_deg), 0.0, 16.0))
    return _CL_INTERP([[a, 0.0]]).item(), _CD_INTERP([[a, 0.0]]).item()


def _cm_alpha(alpha_deg: float) -> float:
    """Статический момент тангажа Cm(alpha) — линейная интерполяция."""
    return float(np.interp(abs(alpha_deg), _CM_AOA_PTS, _CM_VAL_PTS))


def _quat_to_rotm(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Матрица поворота R_bw из кватерниона (тело → инерциальная СК)."""
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)  ],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)  ],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=float)


def _initial_quaternion(alpha_rad: float, beta_rad: float) -> np.ndarray:
    """
    Начальный кватернион: поворот по yaw=beta, затем pitch=alpha.

    Returns:
        ndarray (4,) — (qw, qx, qy, qz), нормирован.
    """
    cy, sy = np.cos(beta_rad / 2.0),  np.sin(beta_rad / 2.0)
    cp, sp = np.cos(alpha_rad / 2.0), np.sin(alpha_rad / 2.0)
    return np.array([cy*cp, -sy*sp, cy*sp, sy*cp])


# ---------------------------------------------------------------------------
# Правая часть ОДУ (13 уравнений состояния)
# ---------------------------------------------------------------------------

def _derivatives(
    state: np.ndarray,
    p: dict,
    S_ref: float,
    C_ref: float,
    B_ref: float,
    R_cp: np.ndarray,
    I_mat: np.ndarray,
    I_inv: np.ndarray,
) -> np.ndarray:
    """
    Производные вектора состояния 6-DoF твёрдого тела.

    state[0:3]  — положение (x, y, z), м
    state[3:6]  — скорость (vx, vy, vz), м/с
    state[6:10] — кватернион (qw, qx, qy, qz)
    state[10:13]— угловые скорости тела (p, q, r), рад/с

    Args:
        state:  Вектор состояния (13,).
        p:      Параметры среды (m, g, wx, wy, wz).
        S_ref:  Площадь миделя, м².
        C_ref:  Продольный аэродинамический референс, м.
        B_ref:  Поперечный аэродинамический референс, м.
        R_cp:   Вектор плеча CP–CG в связанной СК, м.
        I_mat:  Тензор инерции (3,3), кг·м².
        I_inv:  Обратный тензор инерции (3,3).

    Returns:
        ds/dt, ndarray (13,).
    """
    vx, vy, vz     = state[3], state[4], state[5]
    qw, qx, qy, qz = state[6], state[7], state[8], state[9]
    pr, qr, rr     = state[10], state[11], state[12]

    m  = p['m']
    g  = p['g']
    wx = p['wx']
    wy = p['wy']
    wz = p['wz']

    # Плотность атмосферы на текущей высоте
    y_pos = max(float(state[1]), 0.0)
    rho = _RHO0 * np.exp(-y_pos / _H_ATM)

    # Нормализация кватерниона (устраняет накопление погрешности RK4)
    q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if q_norm < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        inv = 1.0 / q_norm
        qw *= inv; qx *= inv; qy *= inv; qz *= inv

    R_bw = _quat_to_rotm(qw, qx, qy, qz)
    R_wb = R_bw.T   # инерциальная → тело

    # Скорость снаряда относительно воздушной массы
    v_rel = np.array([vx - wx, vy - wy, vz - wz])
    V = float(np.sqrt(v_rel @ v_rel)) + 1e-12

    # Угол атаки в связанной СК
    v_body = R_wb @ v_rel
    alpha_deg = float(np.degrees(np.arctan2(v_body[2], v_body[0])))

    Cl, Cd = _lookup_cl_cd(alpha_deg)

    # Сила сопротивления (против v_rel)
    F_drag = -0.5 * rho * S_ref * Cd * V * v_rel

    # Подъёмная сила (перп. v_rel в плоскости v_rel–e_y)
    e_y = np.array([0.0, 1.0, 0.0])
    lift_dir = np.cross(np.cross(v_rel, e_y), v_rel)
    ln = float(np.sqrt(lift_dir @ lift_dir))
    lift_dir = lift_dir / ln if ln > 1e-12 else np.zeros(3)
    F_lift = 0.5 * rho * S_ref * Cl * V * V * lift_dir

    # Суммарные линейные ускорения
    a_lin = (F_drag + F_lift + np.array([0.0, -m * g, 0.0])) / m

    # --- Моменты твёрдого тела ---
    q_dyn = 0.5 * rho * V * V

    # Момент тангажа: статика + демпфирование по q
    q_hat = (qr * C_ref) / (2.0 * V)
    My = q_dyn * S_ref * C_ref * (_cm_alpha(alpha_deg) + _CM_Q * q_hat)

    # Демпфирование крена и рысканья
    p_hat = (pr * B_ref) / (2.0 * V)
    r_hat = (rr * B_ref) / (2.0 * V)
    Mx = q_dyn * S_ref * B_ref * (_CL_P * p_hat)
    Mz = q_dyn * S_ref * B_ref * (_CN_R * r_hat)

    # Момент плеча CP–CG (аэро-сила в связанной СК)
    F_aero_body = R_wb @ (F_drag + F_lift)
    M_body = np.array([Mx, My, Mz]) + np.cross(R_cp, F_aero_body)

    # Уравнения Эйлера: I·dω/dt = M - ω × (I·ω)
    omega = np.array([pr, qr, rr])
    d_omega = I_inv @ (M_body - np.cross(omega, I_mat @ omega))

    # Кинематика кватерниона: dq/dt = ½ Ω(ω) q
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
# Один полёт: RK4 до касания земли
# ---------------------------------------------------------------------------

def _simulate_one(
    state0: np.ndarray,
    sim_params: dict,
    aero: tuple,
    dt: float,
    max_steps: int,
) -> np.ndarray:
    """
    Интегрирует один полёт снаряда (RK4) до y ≤ 0.

    Args:
        state0:     Начальный вектор состояния (13,).
        sim_params: Параметры среды.
        aero:       (S_ref, C_ref, B_ref, R_cp, I_mat, I_inv).
        dt:         Шаг RK4, с.
        max_steps:  Лимит итераций.

    Returns:
        ndarray (K, 3) — (x, y, z) по шагам полёта.
    """
    S_ref, C_ref, B_ref, R_cp, I_mat, I_inv = aero
    state = state0.copy()
    traj = [state[:3].copy()]

    def f(s):
        return _derivatives(s, sim_params, S_ref, C_ref, B_ref, R_cp, I_mat, I_inv)

    for _ in range(max_steps):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state +       dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        y_new = float(state[1])   # высота — всегда скаляр float

        if y_new <= 0.0 and len(traj) > 2:
            # Линейная интерполяция точки касания y = 0
            y_prev = float(traj[-1][1])
            dy = y_prev - y_new
            frac = (y_prev / dy) if abs(dy) > 1e-12 else 0.0
            frac = float(np.clip(frac, 0.0, 1.0))
            x_land = float(traj[-1][0]) + frac * (float(state[0]) - float(traj[-1][0]))
            z_land = float(traj[-1][2]) + frac * (float(state[2]) - float(traj[-1][2]))
            traj.append(np.array([x_land, 0.0, z_land]))
            break

        traj.append(state[:3].copy())

    return np.array(traj)   # (K, 3)


# ---------------------------------------------------------------------------
# Публичный интерфейс
# ---------------------------------------------------------------------------

def calculate(params: dict) -> tuple:
    """
    Ансамблевый расчёт 6-DoF твёрдотельного снаряда методом Монте-Карло.

    Тензор инерции вычисляется аналитически из геометрии снаряда (D, L, m).
    Каждый пуск — независимый случайный вектор ветра.

    Args:
        params: Строковый словарь из UI.

    Returns:
        (trajectories, is_3d, label):
          trajectories — list[ndarray(K_i, 3)];
          is_3d        — True;
          label        — строка для легенды.
    """
    n_runs    = int(params.get('N', 60))
    v0        = float(params.get('v0', 250.0))
    angle_deg = float(params.get('angle', 45.0))
    azim_deg  = float(params.get('azimuth', 0.0))
    wind_max  = float(params.get('Wind_Max', 6.0))
    mass      = float(params.get('m', 0.85))
    D         = float(params.get('D', 0.04))
    L         = float(params.get('L', 0.18))

    # Геометрия снаряда
    R     = D / 2.0
    S_ref = np.pi * R * R            # площадь миделя
    C_ref = L                        # продольный референс = длина снаряда
    B_ref = D                        # поперечный референс = калибр
    R_cp  = np.array([0.25 * L, 0.0, 0.0])   # CP смещён на 1/4 L от носа

    # Аналитический тензор инерции (цилиндр)
    I_mat, I_inv = _inertia_tensor(mass, D, L)
    aero = (S_ref, C_ref, B_ref, R_cp, I_mat, I_inv)

    alpha = np.radians(angle_deg)
    beta  = np.radians(azim_deg)

    vx0 = v0 * np.cos(alpha) * np.cos(beta)
    vy0 = v0 * np.sin(alpha)
    vz0 = v0 * np.cos(alpha) * np.sin(beta)
    q0  = _initial_quaternion(alpha, beta)

    base_params = {'m': mass, 'g': 9.81, 'wx': 0.0, 'wy': 0.0, 'wz': 0.0}

    dt        = 0.01     # шаг RK4, с
    max_steps = 20000    # лимит: ~200 с полёта

    rng = np.random.default_rng()
    all_trajectories = []

    for _ in range(n_runs):
        w_speed = float(rng.uniform(0.0, wind_max))
        w_dir   = float(rng.uniform(0.0, 2.0 * np.pi))

        sim_p = dict(base_params)
        sim_p['wx'] = w_speed * np.cos(w_dir)
        sim_p['wz'] = w_speed * np.sin(w_dir)

        s0 = np.zeros(13)
        s0[0:3]   = [0.0, 0.0, 0.0]
        s0[3:6]   = [vx0, vy0, vz0]
        s0[6:10]  = q0
        s0[10:13] = [0.0, 0.0, 0.0]   # без начального вращения

        traj = _simulate_one(s0, sim_p, aero, dt, max_steps)
        all_trajectories.append(traj)

    label = (
        f"6-DoF МК: N={n_runs}, D={D*100:.0f}мм, "
        f"L={L*100:.0f}см, m={mass}кг, Wind≤{wind_max}м/с"
    )
    return all_trajectories, True, label