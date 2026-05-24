"""
Модуль высокоточного баллистического моделирования в стохастическом ветровом поле.

Реализует физику движения тела с учетом:
1. Динамического профиля ветра: скорость растет с высотой (сдвиг ветра).
2. Стохастических порывов ветра (турбулентность, генерируемая на каждом шаге интеграции).
3. Волнового кризиса: зависимость коэффициента Cx от числа Маха (скорости звука).
4. Экспоненциальной атмосферы и аэродинамической подъемной силы.

Интегрирование системы ОДУ 1-го порядка выполняется методом RK4.
"""

import numpy as np


def get_name():
    """Возвращает название модели для отображения в списке."""
    return "5. Динамическое ветровое поле и Мах-зависимость (Advanced 3D)"


def get_info():
    """Метаданные модели для интерфейса."""
    return {
        "description": (
            "Максимально приближенная к реальным симуляторам модель. "
            "Реализует высотный сдвиг ветра W(y) = W₀·(y/y₀)^0.2 и накладывает "
            "случайные турбулентные пульсации на каждом шаге расчета. "
            "Учитывает волновой кризис: при приближении скорости тела к числу Маха (M=1) "
            "коэффициент лобового сопротивления Cx автоматически возрастает в 2.5 раза. "
            "Расчет ансамбля траекторий выполнен методом RK4."
        ),
        "formula": (
            r"$W_x(y) = W_{x0} \cdot \left(\frac{y}{10}\right)^{0.15} + \xi(t), \quad M = \frac{v}{v_{sound}}$"
            "\n\n"
            r"$C_x(M) = C_{x0} \cdot \left(1 + \frac{1.5}{1 + e^{-20(M-1)}}\right)$"
        ),
        "parameters_info": {
            "N": "Количество запусков (10-100)",
            "v0": "Начальная скорость вылета (м/с)",
            "angle": "Угол броска к горизонту (градусы)",
            "w0_x": "Базовый ветер у земли по X (м/с)",
            "w0_z": "Базовый боковой ветер у земли по Z (м/с)",
            "turb_std": "Интенсивность турбулентности (сигма порывов, м/с)",
            "Cx0": "Базовый дозвуковой коэф. сопротивления",
            "Cy": "Коэффициент подъемной силы",
            "m": "Масса тела (кг)"
        }
    }


def get_params():
    """Параметры по умолчанию."""
    return {
        "N": "50",
        "v0": "350.0",  # Выше скорости звука (~340 м/с) для демонстрации Мах-эффекта
        "angle": "45.0",
        "w0_x": "-4.0",
        "w0_z": "3.0",
        "turb_std": "2.5",
        "Cx0": "0.2",
        "Cy": "0.05",
        "m": "1.0",
        "S": "0.01"
    }


def calculate(params):
    """Расчет пучка 3D-траекторий в динамической стохастической среде."""
    num_simulations = int(params.get('N', 50))
    v0 = float(params.get('v0', 350.0))
    alpha = np.radians(float(params.get('angle', 45.0)))

    w0_x = float(params.get('w0_x', -4.0))
    w0_z = float(params.get('w0_z', 3.0))
    turb_std = float(params.get('turb_std', 2.5))

    cx0 = float(params.get('Cx0', 0.2))
    cy = float(params.get('Cy', 0.05))
    m = float(params.get('m', 1.0))
    s_area = float(params.get('S', 0.01))

    g = 9.81
    dt = 0.04
    v_sound = 340.0  # Скорость звука у земли (м/с)

    all_trajectories = []

    for i in range(num_simulations):
        # Вектор состояния: [x, y, z, vx, vy, vz]
        state = np.array([0.0, 0.0, 0.0, v0 * np.cos(alpha), v0 * np.sin(alpha), 0.0])
        traj_single = [state[:3].copy()]

        for _ in range(5000):
            # Один случайный порыв на шаг интегрирования; высотная часть ветра
            # пересчитывается внутри RK4 для каждого промежуточного состояния.
            gust_x = np.random.normal(0, turb_std)
            gust_z = np.random.normal(0, turb_std)

            # Алгоритм численного интегрирования RK4
            def get_derivs(st):
                """Правая часть ОДУ для текущего состояния RK4."""
                _, y, _, vx_c, vy_c, vz_c = st

                # 1. Генерация ветрового поля: высотный сдвиг + порыв.
                height_factor = (max(y, 0.0) / 10.0) ** 0.15
                w_x = w0_x * height_factor + gust_x
                w_z = w0_z * height_factor + gust_z

                # Относительная скорость снаряда в ветровом поле
                vx_rel = vx_c - w_x
                vz_rel = vz_c - w_z
                v_rel = np.sqrt(vx_rel ** 2 + vy_c ** 2 + vz_rel ** 2)

                if v_rel < 1e-6:
                    ax, ay, az = 0.0, -g, 0.0
                else:
                    # 2. Учет волнового кризиса (число Маха)
                    mach = v_rel / v_sound
                    cx_dynamic = cx0 * (
                        1.0 + 1.5 / (1.0 + np.exp(-20.0 * (mach - 1.0)))
                    )

                    # Атмосфера
                    rho = 1.225 * np.exp(-max(y, 0.0) / 8430.0)
                    base_coeff = (rho * s_area) / (2 * m)

                    # Силы: лобовое сопротивление (Cx) и подъемная сила (Cy)
                    ax = -base_coeff * v_rel * (cx_dynamic * vx_rel + cy * vy_c)
                    ay = -g - base_coeff * v_rel * (cx_dynamic * vy_c - cy * vx_rel)
                    az = -base_coeff * v_rel * (cx_dynamic * vz_rel)

                return np.array([vx_c, vy_c, vz_c, ax, ay, az])

            prev_state = state.copy()
            k1 = get_derivs(state)
            k2 = get_derivs(state + k1 * dt / 2)
            k3 = get_derivs(state + k2 * dt / 2)
            k4 = get_derivs(state + k3 * dt)

            state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            if state[1] < 0:
                fraction = prev_state[1] / (prev_state[1] - state[1])
                x_end = prev_state[0] + fraction * (state[0] - prev_state[0])
                z_end = prev_state[2] + fraction * (state[2] - prev_state[2])
                traj_single.append(np.array([x_end, 0.0, z_end]))
                break

            traj_single.append(state[:3].copy())

        all_trajectories.append(np.array(traj_single))

    return all_trajectories, True, f"Динамическая турбулентность ({num_simulations} линий)"
