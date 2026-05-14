"""
Модуль пространственного баллистического моделирования снаряда как твердого тела (6-DoF).

Точная физика Mitchell Stolk, адаптированная под неуправляемый баллистический полет.
Учитывает:
1. 13 уравнений состояния (координаты, компоненты скорости, кватернионы ориентации, угловые скорости).
2. Динамику вращения снаряда и аэродинамические моменты (демпфирование, опрокидывание).
3. Интерполяцию таблиц Cd и Cl из CFD по углам атаки.
4. Случайное распределение ветра у земли по методу Монте-Карло.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_name():
    """Возвращает название модели для отображения в списке."""
    return "6. Неуправляемый снаряд Mitchell Stolk (6-DoF Твердое тело)"


def get_info():
    """Метаданные модели для интерфейса."""
    return {
        "description": (
            "Физический симулятор баллистического снаряда (6 степеней свободы). "
            "Тело моделируется как вращающееся твердое тело с собственной матрицей инерции. "
            "Пуск производится с земли (y=0) под углом к горизонту. "
            "Силы и моменты (лобовое сопротивление, подъемная сила, момент тангажа) "
            "динамически интерполируются по таблицам CFD в зависимости от угла атаки. "
            "Наведение отключено — исследуется естественная баллистическая траектория."
        ),
        "formula": (
            r"$\mathbf{u} = [x, y, z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, p, q, r]^T$"
            "\n\n"
            r"$M_{body} = M_{static} + M_{damping} + (\mathbf{r}_{cp} \times \mathbf{F}_{aero})$"
        ),
        "parameters_info": {
            "N": "Количество запусков Монте-Карло (50-100)",
            "v0": "Начальная скорость вылета снаряда (м/с)",
            "angle": "Угол наклона орудия к горизонту (градусы)",
            "azimuth": "Азимут стрельбы (градусы, 0 - вдоль оси X)",
            "Wind_Max": "Максимальная скорость случайного ветра у земли (м/с)",
            "m": "Масса снаряда (кг)"
        }
    }


def get_params():
    """Параметры по умолчанию для пуска с земли."""
    return {
        "N": "60",
        "v0": "250.0",
        "angle": "45.0",
        "azimuth": "0.0",
        "Wind_Max": "6.0",
        "m": "1.5"
    }


# === АЭРОДИНАМИЧЕСКИЕ ТАБЛИЦЫ И ИНТЕРПОЛЯТОРЫ ===
AoA = np.array([0, 4, 6, 8, 12, 16], dtype=float)
canP = np.array([0, 4, 6, 8, 12, 16], dtype=float)

CdP = np.array([
    [0.35, 0.40, 0.50, 0.65, 1.10, 1.80],
    [0.40, 0.50, 0.60, 0.80, 1.30, 2.00],
    [0.45, 0.55, 0.65, 0.85, 1.40, 2.10],
    [0.50, 0.65, 0.80, 1.00, 1.60, 2.30],
    [0.60, 0.75, 0.90, 1.10, 1.70, 2.35],
    [0.70, 0.85, 1.00, 1.20, 1.80, 2.45]
], dtype=float)

ClP = np.array([
    [0.00, 1.10, 1.75, 2.50, 3.90, 5.50],
    [0.05, 1.20, 1.90, 2.60, 4.00, 5.40],
    [0.20, 1.25, 1.95, 2.70, 4.10, 5.30],
    [0.40, 1.35, 2.00, 2.80, 4.20, 5.20],
    [0.50, 1.40, 2.05, 2.70, 4.10, 5.10],
    [0.60, 1.45, 2.10, 2.60, 4.00, 5.00]
], dtype=float)

_cl_interp = RegularGridInterpolator((AoA, canP), ClP, bounds_error=False, fill_value=None)
_cd_interp = RegularGridInterpolator((AoA, canP), CdP, bounds_error=False, fill_value=None)

alpha_table_full = np.array([0, 4, 6, 8, 12, 16, 25, 45, 60, 90, 180], dtype=float)
Cm_alpha_values = np.array([0, -0.10, -0.15, -0.30, -1, -2, -4, -8, -10, -8, -6], dtype=float)

# Геометрия снаряда
D_ref = 0.05
S_ref = 0.25 * np.pi * (D_ref**2)
c_ref = 0.50
b_ref = 0.15
I_body = (8.5e-4, 0.03, 0.03)
r_cp_cg = (0.03, 0.0, 0.0)

I_mat = np.diag(I_body)
I_inv = np.linalg.inv(I_mat)


def lookup_cl_cd(alpha_deg, pitch_deg):
    a = np.clip(abs(alpha_deg), 0.0, 16.0)
    p = np.clip(abs(pitch_deg), 0.0, 16.0)
    return _cl_interp(np.array([[a, p]])).item(), _cd_interp(np.array([[a, p]])).item()


def cm_alpha_interp(alpha_deg):
    return float(np.interp(abs(alpha_deg), alpha_table_full, Cm_alpha_values))


def quat_to_rotm(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx**2+qy**2)]
    ], dtype=float)


def get_derivatives_projectile(y, p):
    """Вычисляет 13 производных вектора состояния неуправляемого баллистического снаряда."""
    x, y_pos, z = y[0], y[1], y[2]
    vx, vy, vz = y[3], y[4], y[5]
    qw, qx, qy, qz = y[6], y[7], y[8], y[9]
    pr, qr, rr = y[10], y[11], y[12]

    rho, S, m, g = p['rho'], p['S'], p['m'], p['g']
    wx, wy, wz = p['wx'], p['wy'], p['wz']

    # Нормализация кватерниона для исключения накопления вычислительной погрешности
    qnorm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if qnorm < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw, qx, qy, qz = qw/qnorm, qx/qnorm, qy/qnorm, qz/qnorm

    Rbw = quat_to_rotm(qw, qx, qy, qz)
    Rwb = Rbw.T

    # Относительная скорость снаряда в воздушной среде
    vrel = np.array([vx - wx, vy - wy, vz - wz])
    V = np.sqrt(vrel[0]**2 + vrel[1]**2 + vrel[2]**2) + 1e-12
    v_hat = vrel / V

    # Угол атаки снаряда в связанной системе координат
    Vb = Rwb @ vrel
    alpha_deg = np.degrees(np.arctan2(Vb[2], Vb[0]))

    # Никакихcmd наведения нет. Снаряд имеет постоянный балансировочный трим (угол скоса)
    pitch_trim = 0.0

    # Аэродинамические коэффициенты по таблицам CFD
    Cl_basic, Cd_val = lookup_cl_cd(alpha_deg, pitch_trim)
    F_drag = -0.5 * rho * S * Cd_val * V * vrel

    # Направление подъемной силы, возникающей при перекосе снаряда на углах атаки
    e_y = np.array([0.0, 1.0, 0.0])
    cross1 = np.cross(vrel, e_y)
    lift_basic_dir = np.cross(cross1, vrel)
    lb_norm = np.sqrt(lift_basic_dir[0]**2 + lift_basic_dir[1]**2 + lift_basic_dir[2]**2)
    if lb_norm > 1e-12:
        lift_basic_dir /= lb_norm
    else:
        lift_basic_dir = np.zeros(3)

    F_lift = 0.5 * rho * S * Cl_basic * V**2 * lift_basic_dir

    # Полная сила: Аэродинамика + Гравитация Земли
    F_total = F_drag + F_lift + np.array([0.0, -m*g, 0.0])
    ax, ay, az = F_total / m

    # --- СИЛОВЫЕ МОМЕНТЫ ТВЕРДОГО ТЕЛА ---
    q_dyn = 0.5 * rho * V**2
    Cm_static = cm_alpha_interp(alpha_deg)
    q_hat = (qr * c_ref) / (2.0*V + 1e-12)
    Cm_total = Cm_static + (-20.0) * q_hat  # Включая демпфирование тангажа
    My = q_dyn * S * c_ref * Cm_total

    # Демпфирующие моменты крена и рысканья (вращение снаряда затухает из-за вязкости)
    Mx = q_dyn * S * b_ref * ((-10.0) * (pr * b_ref) / (2.0*V + 1e-12))
    Mz = q_dyn * S * b_ref * ((-20.0) * (rr * b_ref) / (2.0*V + 1e-12))

    M_body = np.array([Mx, My, Mz])

    # Момент от несовпадения центра давления (CP) и центра масс (CG) — плечо силы
    F_aero_B = Rwb @ (F_drag + F_lift)
    M_arm = np.cross(np.array(r_cp_cg), F_aero_B)
    M_body = M_body + M_arm

    # Динамические уравнения Эйлера для угловых скоростей
    omega = np.array([pr, qr, rr])
    Iomega = I_mat @ omega
    domega = I_inv @ (M_body - np.cross(omega, Iomega))

    # Кинематические уравнения для кватернионов
    qdot = np.array([
        0.5 * (-pr*qx - qr*qy - rr*qz),
        0.5 * (pr*qw + rr*qy - qr*qz),
        0.5 * (qr*qw - rr*qx + pr*qz),
        0.5 * (rr*qw + qr*qx - pr*qy)
    ])

    return np.array([vx, vy, vz, ax, ay, az, qdot[0], qdot[1], qdot[2], qdot[3], domega[0], domega[1], domega[2]])


def calculate(params):
    """
    Выполняет пакетный расчет 6-DoF траекторий методом Монте-Карло.
    Возвращает список массивов траекторий.
    """
    num_simulations = int(params.get('N', 60))
    v0_val = float(params.get('v0', 250.0))
    alpha = np.radians(float(params.get('angle', 45.0)))
    beta = np.radians(float(params.get('azimuth', 0.0)))
    wind_max = float(params.get('Wind_Max', 6.0))
    m_val = float(params.get('m', 1.5))

    sim_params = {
        'rho': 1.225, 'S': S_ref, 'm': m_val, 'g': 9.81, 'wy': 0.0
    }

    dt = 0.01
    all_trajectories = []
    rng = np.random.default_rng()

    vx0 = v0_val * np.cos(alpha) * np.cos(beta)
    vy0 = v0_val * np.sin(alpha)
    vz0 = v0_val * np.cos(alpha) * np.sin(beta)

    qw0 = np.cos(alpha / 2) * np.cos(beta / 2)
    qx0 = np.sin(alpha / 2) * np.sin(beta / 2)
    qy0 = np.cos(alpha / 2) * np.sin(beta / 2)
    qz0 = np.sin(alpha / 2) * np.cos(beta / 2)

    # Запускаем N случайных симуляций в стохастическом поле
    for i in range(num_simulations):
        w_spd = rng.uniform(0.0, wind_max)
        w_dir = rng.uniform(0.0, 2 * np.pi)
        sim_params['wx'] = w_spd * np.cos(w_dir)
        sim_params['wz'] = w_spd * np.sin(w_dir)

        state = np.zeros(13)
        state[0:6] = [0.0, 0.0, 0.0, vx0, vy0, vz0]
        state[6:10] = [qw0, qx0, qy0, qz0]

        traj_single = [state[:3].copy()]

        for _ in range(15000):
            k1 = get_derivatives_projectile(state, sim_params)
            k2 = get_derivatives_projectile(state + 0.5 * dt * k1, sim_params)
            k3 = get_derivatives_projectile(state + 0.5 * dt * k2, sim_params)
            k4 = get_derivatives_projectile(state + dt * k3, sim_params)
            state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            if state <= 0.0 and len(traj_single) > 5:
                prev = traj_single[-1]
                curr_y = state
                frac = prev / (prev - curr_y + 1e-12)

                x_end = prev + frac * (state - prev)
                z_end = prev + frac * (state - prev)
                traj_single.append(np.array([x_end, 0.0, z_end]))
                break
            traj_single.append(state[:3].copy())

        all_trajectories.append(np.array(traj_single))

    # Возвращаем только список массивов случайных пусков
    return all_trajectories, True, "6-DoF Статистический ансамбль снарядов"

