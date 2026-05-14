"""
Модуль трехмерного статистического моделирования методом Монте-Карло (3-DoF).

Описывает ансамбль пространственных траекторий (X, Y, Z) материальной точки.
Учитывает случайный трехмерный вектор ветра (попутный/встречный и боковой снос)
и случайное рассеивание начальной скорости вылета тела по закону Гаусса.
 Интегрирование системы нелинейных уравнений движения выполняется методом RK4.
"""

import numpy as np


def get_name():
    """Возвращает название модели для отображения в списке."""
    return "4. Пространственное моделирование Монте-Карло (Честное 3D)"


def get_info():
    """Метаданные модели для интерфейса."""
    return {
        "description": (
            "Полноценная трехмерная стохастическая модель баллистического полета. "
            "Учитывает случайную погрешность начальной скорости вылета, "
            "а также случайную силу и направление ветра в 3D-пространстве. "
            "Ось X — дальность полета, Y — высота, Z — боковое отклонение (снос). "
            "Результаты выводятся в виде интерактивного облака траекторий (пучка линий), "
            "позволяющего оценить эллипс рассеивания снаряда на плоскости падения."
        ),
        "formula": (
            r"$\mathbf{r} = (x, y, z), \quad \mathbf{v}_{wind} = (w_x, 0, w_z)$"
            "\n\n"
            r"$\dot{v}_x = -k \cdot v_{rel} \cdot (v_x - w_x)$"
            "\n\n"
            r"$\dot{v}_z = -k \cdot v_{rel} \cdot (v_z - w_z)$"
        ),
        "parameters_info": {
            "N": "Количество запусков (рекомендовано 50-200)",
            "v0_mean": "Средняя начальная скорость (м/с)",
            "v0_std": "Разброс скорости (м/с)",
            "angle": "Угол броска к горизонту (градусы)",
            "azimuth": "Азимут стрельбы (градусы, 0 - строго по оси X)",
            "wx_mean": "Продольный ветер (м/с, попутный > 0)",
            "wz_mean": "Боковой ветер (м/с, снос вправо > 0)",
            "wind_std": "Флуктуации ветра (сигма, м/с)",
            "Cx": "Коэффициент лобового сопротивления"
        }
    }


def get_params():
    """Параметры по умолчанию."""
    return {
        "N": "80",
        "v0_mean": "150.0",
        "v0_std": "4.0",
        "angle": "45.0",
        "azimuth": "0.0",
        "wx_mean": "-3.0",
        "wz_mean": "5.0",
        "wind_std": "2.0",
        "Cx": "0.25",
        "m": "1.0",
        "S": "0.01"
    }


def calculate(params):
    """
    Расчет ансамбля трехмерных траекторий.
    Возвращает список массивов формы (Steps, 3), где каждый столбец — это X, Y, Z.
    """
    num_simulations = int(params.get('N', 80))
    v0_mean = float(params.get('v0_mean', 150.0))
    v0_std = float(params.get('v0_std', 4.0))
    alpha = np.radians(float(params.get('angle', 45.0)))
    beta = np.radians(float(params.get('azimuth', 0.0)))

    wx_mean = float(params.get('wx_mean', -3.0))
    wz_mean = float(params.get('wz_mean', 5.0))
    wind_std = float(params.get('wind_std', 2.0))

    cx = float(params.get('Cx', 0.25))
    m = float(params.get('m', 1.0))
    s_area = float(params.get('S', 0.01))

    g = 9.81
    dt = 0.05

    all_trajectories = []

    # Генерация случайных распределений по Гауссу
    v0_samples = np.random.normal(v0_mean, v0_std, num_simulations)
    wx_samples = np.random.normal(wx_mean, wind_std, num_simulations)
    wz_samples = np.random.normal(wz_mean, wind_std, num_simulations)

    for i in range(num_simulations):
        v0_rand = v0_samples[i]
        w_x = wx_samples[i]
        w_z = wz_samples[i]

        # Проекции начальной скорости на 3 оси координат
        vx = v0_rand * np.cos(alpha) * np.cos(beta)
        vy = v0_rand * np.sin(alpha)
        vz = v0_rand * np.cos(alpha) * np.sin(beta)

        # Вектор состояния: [x, y, z, vx, vy, vz]
        state = np.array([0.0, 0.0, 0.0, vx, vy, vz])
        traj_single = [state[:3].copy()]  # Сохраняем X, Y, Z

        for _ in range(4000):
            x, y, z, vx_c, vy_c, vz_c = state

            # Расчет относительной скорости снаряда с учетом ветра
            vx_rel = vx_c - w_x
            vz_rel = vz_c - w_z
            v_rel = np.sqrt(vx_rel ** 2 + vy_c ** 2 + vz_rel ** 2)

            if v_rel < 1e-6:
                ax, ay, az = 0.0, -g, 0.0
            else:
                rho = 1.225 * np.exp(-y / 8430.0)  # Экспоненциальная атмосфера
                coeff = (rho * s_area * v_rel) / (2 * m)

                # Сила сопротивления направлена строго против вектора относительной скорости
                ax = -coeff * cx * vx_rel
                ay = -g - coeff * cx * vy_c
                az = -coeff * cx * vz_rel

            # Шаг Рунге-Кутты 4-го порядка
            def rk_derivatives(st):
                return np.array([vx_c, vy_c, vz_c, ax, ay, az])

            k1 = rk_derivatives(state)
            k2 = rk_derivatives(state + k1 * dt / 2)
            k3 = rk_derivatives(state + k2 * dt / 2)
            k4 = rk_derivatives(state + k3 * dt)

            state += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Условие встречи с поверхностью земли (y < 0)
            if state[1] < 0:
                # Интерполяция конечной точки по осям X и Z
                fraction = y / (y - state[1])
                x_end = x + fraction * (state[0] - x)
                z_end = z + fraction * (state[2] - z)
                traj_single.append(np.array([x_end, 0.0, z_end]))
                break

            traj_single.append(state[:3].copy())

        all_trajectories.append(np.array(traj_single))

    # Возвращаем флаг True во втором аргументе, сообщая main.py, что этот результат — трехмерный
    return all_trajectories, True, f"3D Монте-Карло ({num_simulations} траекторий)"
