"""
Модуль 8. Итоговая 6-DoF модель с заранее известным динамическим 3D-полем ветра.

Идея модели:
  - Снаряд считается как твердое тело с 13 переменными состояния:
    [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r].
  - Ветер задан не одной случайной скоростью, а 4D-метеокартами:
    W = W(t, x, y, z). Для каждого пуска поле заранее строится на
    регулярной сетке и далее интерполируется в каждой точке траектории.
  - Так как поле известно на будущих слоях времени, RK4 берет ветер на
    промежуточных подшагах t + dt/2 и t + dt. Это имитирует расчет "на шаг
    вперед" по известной метеокарте.
  - Стохастика Монте-Карло сохранена через разброс начальных условий
    (скорость, угол, азимут) и через ансамбль близких, но разных
    реализаций заранее известных полей ветра.
  - Для GUI возвращается не только пучок траекторий, но и кадры ветрового
    поля для воспроизведения записи расчета.
"""

import numpy as np


WIND_LAYERS = [
    {
        "name": "Пограничный слой: мелкая турбулентность",
        "bottom": 0.0,
        "top": 180.0,
        "color": (0.36, 0.74, 0.32, 0.10),
    },
    {
        "name": "Средний слой: сдвиг и вихревые ячейки",
        "bottom": 180.0,
        "top": 850.0,
        "color": (0.95, 0.70, 0.24, 0.08),
    },
    {
        "name": "Верхний слой: организованный перенос",
        "bottom": 850.0,
        "top": 2600.0,
        "color": (0.38, 0.58, 0.95, 0.06),
    },
]


# ---------------------------------------------------------------------------
# Интерфейс модели
# ---------------------------------------------------------------------------

def get_name() -> str:
    """Возвращает название модели для отображения в списке."""
    return "8. Итоговая модель (6-DoF + динамическое поле ветра)"


def get_info() -> dict:
    """Метаданные модели для интерфейса."""
    return {
        "description": (
            "Финальная модель объединяет 6-DoF динамику твердого тела, "
            "Монте-Карло разброс начальных условий и ансамбль заранее "
            "известных трехмерных метеокарт ветра W(t,x,y,z). Для каждого "
            "пуска создается собственная близкая реализация поля из одного "
            "погодного сценария. Ветер задан на сетке и интерполируется в "
            "каждой точке расчета. Для RK4 используются значения поля на "
            "текущем и будущих подшагах времени, поэтому расчет учитывает "
            "динамику потока вперед по времени. "
            "Результат содержит траектории, точки падения и кадры ветрового "
            "поля для воспроизведения процесса."
        ),
        "formula": (
            r"$\mathbf{W}=\mathbf{W}(t,x,y,z),\quad "
            r"\mathbf{v}_{rel}=\mathbf{v}-\mathbf{W}$"
            "\n\n"
            r"$\dot{\omega}=I^{-1}\left(\mathbf{M}-"
            r"\omega\times(I\omega)\right)$"
            "\n\n"
            r"$\mathbf{s}_{n+1}=\mathbf{s}_n+\frac{\Delta t}{6}"
            r"(k_1+2k_2+2k_3+k_4)$"
        ),
        "parameters_info": {
            "N": "Количество пусков Монте-Карло",
            "v0": "Средняя начальная скорость снаряда, м/с",
            "v0_std": "СКО начальной скорости, м/с",
            "angle": "Средний угол возвышения, градусы",
            "angle_std": "СКО угла возвышения, градусы",
            "azimuth": "Средний азимут стрельбы, градусы",
            "azimuth_std": "СКО азимута, градусы",
            "scenario": "Сценарий ветра: calm, shear, storm, vortex, turbulent",
            "Wind_Strength": "Характерная скорость ветра в поле, м/с",
            "Turbulence": "Интенсивность вихревых неоднородностей, м/с",
            "Vortex_X_Shift": "Сдвиг центра вихревой зоны по дальности X, м",
            "Vortex_Z_Shift": "Сдвиг центра вихревой зоны по боковому сносу Z, м",
            "Vortex_Strength_Factor": "Множитель силы основного вихря",
            "m": "Масса снаряда, кг",
            "D": "Калибр/диаметр снаряда, м",
            "L": "Длина снаряда, м",
            "seed": "Зерно генератора случайных чисел",
        },
    }


def get_params() -> dict:
    """Параметры по умолчанию: умеренно тяжелый расчет для GUI."""
    return {
        "N": "24",
        "v0": "250.0",
        "v0_std": "2.0",
        "angle": "45.0",
        "angle_std": "0.25",
        "azimuth": "0.0",
        "azimuth_std": "0.20",
        "scenario": "turbulent",
        "Wind_Strength": "10.0",
        "Turbulence": "3.0",
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    }


# ---------------------------------------------------------------------------
# Динамическое поле ветра
# ---------------------------------------------------------------------------

class DynamicWindField:
    """
    Регулярная 4D-сетка ветра W(t, x, y, z) с быстрой линейной интерполяцией.

    data.shape = (nt, nx, ny, nz, 3), где последние компоненты — wx, wy, wz.
    """

    def __init__(
        self,
        scenario: str,
        strength: float,
        turbulence: float,
        seed: int,
        x_max: float,
        y_max: float,
        z_span: float,
        t_max: float,
        shape: tuple = (18, 13, 7, 11),
        ensemble_index: int = 0,
        ensemble_spread: float = 0.0,
        vortex_x_shift: float = 0.0,
        vortex_z_shift: float = 0.0,
        vortex_strength_factor: float = 1.0,
    ) -> None:
        self.scenario = _normalize_scenario(scenario)
        self.strength = float(strength)
        self.turbulence = float(turbulence)
        self.seed = int(seed)
        self.ensemble_index = int(ensemble_index)
        self.ensemble_spread = max(float(ensemble_spread), 0.0)
        self.vortex_x_shift = float(vortex_x_shift)
        self.vortex_z_shift = float(vortex_z_shift)
        self.vortex_strength_factor = max(float(vortex_strength_factor), 0.0)

        nt, nx, ny, nz = shape
        self.t_axis = np.linspace(0.0, t_max, nt)
        self.x_axis = np.linspace(0.0, x_max, nx)
        self.y_axis = np.linspace(0.0, y_max, ny)
        self.z_axis = np.linspace(-z_span, z_span, nz)
        self.layers = WIND_LAYERS
        self._configure_ensemble_variation(x_max, z_span)

        self.data = self._build_field()

    def _configure_ensemble_variation(self, x_max: float, z_span: float) -> None:
        """Задает индивидуальные параметры метеокарты для одного пуска."""
        rng = np.random.default_rng(self.seed + 7919 * self.ensemble_index)
        spread = self.ensemble_spread

        self.phase_offset = float(rng.uniform(0.0, 2.0 * np.pi) * spread)
        self.front_phase_shift = float(rng.uniform(-0.55, 0.55) * spread)
        self.front_x_shift = float(rng.uniform(-0.14, 0.14) * x_max * spread)
        self.front_z_shift = float(rng.uniform(-0.22, 0.22) * z_span * spread)
        self.front_speed_factor = float(np.clip(1.0 + rng.normal(0.0, 0.12) * spread, 0.65, 1.45))
        self.strength_factor = float(np.clip(1.0 + rng.normal(0.0, 0.13) * spread, 0.62, 1.55))
        self.crosswind_bias = float(rng.normal(0.0, 0.16) * self.strength * spread)
        self.headwind_bias = float(rng.normal(0.0, 0.09) * self.strength * spread)
        self.updraft_bias = float(rng.normal(0.0, 0.035) * self.strength * spread)
        self.vortex_spin = float(rng.choice([-1.0, 1.0])) if spread > 0.0 else 1.0
        self.vortex_radius_factor = float(np.clip(1.0 + rng.normal(0.0, 0.22) * spread, 0.55, 1.65))
        self.vortex_speed_factor = float(np.clip(1.0 + rng.normal(0.0, 0.16) * spread, 0.65, 1.55))

    def _build_field(self) -> np.ndarray:
        """Строит всю 4D-метеокарту до начала расчета траекторий."""
        rng = np.random.default_rng(self.seed)
        xg, yg, zg = np.meshgrid(
            self.x_axis, self.y_axis, self.z_axis, indexing="ij"
        )
        field = np.zeros(
            (len(self.t_axis), len(self.x_axis), len(self.y_axis), len(self.z_axis), 3),
            dtype=float,
        )

        eddies = _make_eddies(
            rng,
            self.turbulence,
            self.strength * self.strength_factor,
            self.x_axis[-1],
            self.z_axis[-1],
            count=_eddy_count_for_scenario(self.scenario),
        )
        for it, t in enumerate(self.t_axis):
            field[it] = self._base_wind(xg, yg, zg, t)
            field[it] += self._eddy_wind(xg, yg, zg, t, eddies)

        return field

    def _base_wind(
        self,
        xg: np.ndarray,
        yg: np.ndarray,
        zg: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Детерминированный крупномасштабный сценарий ветра."""
        wind = np.zeros(xg.shape + (3,), dtype=float)
        h = (np.maximum(yg, 0.0) / 10.0) ** 0.15
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        t_norm = t / max(self.t_axis[-1], 1.0)
        phase = 2.0 * np.pi * (t_norm * self.front_speed_factor) + self.phase_offset
        x_shifted = xg + self.front_x_shift
        z_shifted = zg + self.front_z_shift
        x_norm = x_shifted / max(self.x_axis[-1], 1.0)
        z_norm = z_shifted / max(abs(self.z_axis[-1]), 1.0)
        strength = self.strength * self.strength_factor
        boundary_w, middle_w, upper_w = _layer_weights(yg)

        if self.scenario == "calm":
            wind[..., 0] = 0.08 * strength * h
            wind[..., 2] = 0.10 * strength * np.sin(phase + 2.0 * z_norm)

        elif self.scenario == "shear":
            wind[..., 0] = -0.32 * strength * h
            wind[..., 2] = 0.48 * strength * h * np.sin(phase + np.pi * x_norm)
            wind[..., 1] = 0.04 * strength * np.sin(2.0 * phase + 2.5 * z_norm)

        elif self.scenario == "vortex":
            wind[..., 0] = -0.10 * strength * h + 0.45 * self.headwind_bias * middle_w
            wind[..., 2] = 0.06 * strength * h + 0.75 * self.crosswind_bias * middle_w
            cx = (
                0.30 * self.x_axis[-1]
                + 0.16 * self.x_axis[-1] * np.sin(phase)
                + self.front_x_shift
                + self.vortex_x_shift
            )
            cz = (
                0.34 * self.z_axis[-1] * np.cos(0.8 * phase)
                + self.front_z_shift
                + self.vortex_z_shift
            )
            dx = xg - cx
            dz = zg - cz
            r = np.sqrt(dx * dx + dz * dz) + 1e-9
            radius = (0.15 * self.x_axis[-1]) * self.vortex_radius_factor
            core = np.exp(-(dx * dx + dz * dz) / (radius ** 2))
            swirl = (
                1.55
                * strength
                * self.vortex_speed_factor
                * self.vortex_strength_factor
                * core
            )
            wind[..., 0] += self.vortex_spin * (-dz / r) * swirl
            wind[..., 2] += self.vortex_spin * (dx / r) * swirl
            wind[..., 1] += 0.16 * strength * core * np.sin(phase + self.front_phase_shift)

        elif self.scenario == "turbulent":
            cell_a = np.sin(4.0 * np.pi * z_norm + 1.7 * phase)
            cell_b = np.cos(3.4 * np.pi * x_norm - 1.3 * phase)
            cell_c = np.sin(2.2 * np.pi * (x_norm + z_norm) + 0.9 * phase)
            cell_d = np.cos(2.6 * np.pi * (x_norm - z_norm) - 1.1 * phase)

            boundary_noise_x = np.sin(7.0 * np.pi * x_norm + 5.0 * np.pi * z_norm + 2.4 * phase)
            boundary_noise_z = np.cos(5.5 * np.pi * x_norm - 6.0 * np.pi * z_norm - 1.9 * phase)
            middle_x = 0.48 * cell_a + 0.30 * cell_c
            middle_z = 0.48 * cell_b - 0.32 * cell_d
            upper_x = -0.34 + 0.16 * np.sin(phase + 1.3 * z_norm)
            upper_z = 0.18 * np.sin(0.7 * phase + 1.8 * x_norm)

            wind[..., 0] = strength * h * (
                boundary_w * 0.50 * boundary_noise_x
                + middle_w * middle_x
                + upper_w * upper_x
            )
            wind[..., 2] = strength * h * (
                boundary_w * 0.55 * boundary_noise_z
                + middle_w * middle_z
                + upper_w * upper_z
            )
            wind[..., 1] = strength * (
                boundary_w * 0.10 * np.sin(2.2 * phase + 6.0 * z_norm)
                + middle_w * 0.12 * np.sin(
                phase + 3.0 * x_norm - 2.0 * z_norm
                )
                + upper_w * 0.04 * np.sin(0.6 * phase + 1.2 * x_norm)
            )

        else:  # storm
            front = np.sin(
                2.0 * np.pi * (1.85 * x_norm - self.front_speed_factor * t_norm)
                + self.front_phase_shift
            )
            band = np.cos(
                2.0 * np.pi * (1.15 * z_norm + 0.42 * self.front_speed_factor * t_norm)
                - self.front_phase_shift
            )
            gust = np.sin(2.8 * np.pi * (x_norm + 0.45 * z_norm) + 1.4 * phase)
            squall = np.clip(0.55 + 0.45 * front, 0.0, 1.0)
            wind[..., 0] = (
                -strength * h * (0.30 + 0.32 * front)
                + self.headwind_bias * (0.45 * middle_w + 0.75 * upper_w)
            )
            wind[..., 2] = (
                strength * h * (0.36 * band + 0.24 * gust * squall)
                + self.crosswind_bias * (0.35 * boundary_w + 0.80 * middle_w + 0.55 * upper_w)
            )
            wind[..., 1] = (
                0.11 * strength * np.sin(phase + np.pi * x_norm) * squall
                + self.updraft_bias * (0.50 * boundary_w + 0.80 * middle_w)
            )

        return wind

    def _eddy_wind(
        self,
        xg: np.ndarray,
        yg: np.ndarray,
        zg: np.ndarray,
        t: float,
        eddies: list,
    ) -> np.ndarray:
        """Мелкомасштабные вихревые неоднородности, известные заранее."""
        wind = np.zeros(xg.shape + (3,), dtype=float)
        if self.turbulence <= 0.0:
            return wind

        for eddy in eddies:
            cx = eddy["x0"] + eddy["vx"] * t
            cz = eddy["z0"] + eddy["vz"] * t
            cy = eddy["y0"]
            dx = xg - cx
            dy = yg - cy
            dz = zg - cz
            r2 = dx * dx + 0.35 * dy * dy + dz * dz
            core = np.exp(-r2 / (eddy["radius"] ** 2))
            amp = eddy["amp"] * core
            r = np.sqrt(dx * dx + dz * dz) + 1e-9
            pulse = 0.72 + 0.28 * np.sin(eddy["omega"] * t + eddy["phase"])
            layer_gain = _eddy_layer_gain(yg, eddy["layer"])
            wind[..., 0] += -dz / r * amp * pulse * layer_gain
            wind[..., 2] += dx / r * amp * pulse * layer_gain
            wind[..., 1] += 0.22 * amp * np.sin(0.7 * t + eddy["phase"]) * layer_gain

        return wind

    def sample(self, t: float, position: np.ndarray) -> np.ndarray:
        """Возвращает вектор ветра в произвольной точке через 4D-интерполяцию."""
        t0, ft = _axis_index(self.t_axis, t)
        ix, fx = _axis_index(self.x_axis, position[0])
        iy, fy = _axis_index(self.y_axis, position[1])
        iz, fz = _axis_index(self.z_axis, position[2])

        c0 = _trilinear(self.data[t0], ix, iy, iz, fx, fy, fz)
        c1 = _trilinear(self.data[t0 + 1], ix, iy, iz, fx, fy, fz)
        return c0 * (1.0 - ft) + c1 * ft

    def sample_points(self, t: float, points: np.ndarray) -> np.ndarray:
        """Интерполирует поле для набора точек визуализации."""
        out = np.zeros_like(points, dtype=float)
        for i, point in enumerate(points):
            out[i] = self.sample(t, point)
        return out


def _normalize_scenario(value: str) -> str:
    text = str(value).strip().lower()
    aliases = {
        "штиль": "calm",
        "calm": "calm",
        "сдвиг": "shear",
        "shear": "shear",
        "шторм": "storm",
        "storm": "storm",
        "вихрь": "vortex",
        "vortex": "vortex",
        "турбулентность": "turbulent",
        "турбулентный": "turbulent",
        "turbulent": "turbulent",
    }
    return aliases.get(text, "storm")


def _eddy_count_for_scenario(scenario: str) -> int:
    """Возвращает число вихревых неоднородностей для выбранного режима."""
    if scenario == "storm":
        return 24
    if scenario == "vortex":
        return 20
    if scenario == "turbulent":
        return 26
    if scenario == "shear":
        return 12
    return 8


def _make_eddies(
    rng: np.random.Generator,
    turbulence: float,
    strength: float,
    x_max: float,
    z_max: float,
    count: int = 16,
) -> list:
    """Создает параметры вихрей для заранее известного поля."""
    eddies = []
    for _ in range(count):
        layer = str(rng.choice(["boundary", "middle"], p=[0.45, 0.55]))
        if layer == "boundary":
            y0 = float(rng.uniform(35.0, 220.0))
            radius = float(rng.uniform(90.0, 260.0))
            amp_mult = float(rng.uniform(0.55, 1.20))
        else:
            y0 = float(rng.uniform(220.0, 1050.0))
            radius = float(rng.uniform(220.0, 620.0))
            amp_mult = float(rng.uniform(0.45, 1.05))

        eddies.append({
            "x0": float(rng.uniform(0.02 * x_max, 0.55 * x_max)),
            "y0": y0,
            "z0": float(rng.uniform(-0.35 * z_max, 0.35 * z_max)),
            "vx": float(rng.uniform(-10.0, 10.0)),
            "vz": float(rng.uniform(-6.0, 6.0)),
            "radius": radius,
            "amp": float(rng.choice([-1.0, 1.0]) * amp_mult
                         * max(turbulence, 0.25 * strength)),
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            "omega": float(rng.uniform(0.18, 0.55)),
            "layer": layer,
        })
    return eddies


def _smooth_band(y: np.ndarray, low: float, high: float, softness: float) -> np.ndarray:
    """Плавная маска высотного слоя."""
    enter = 1.0 / (1.0 + np.exp(-(y - low) / softness))
    leave = 1.0 / (1.0 + np.exp((y - high) / softness))
    return enter * leave


def _layer_weights(y: np.ndarray) -> tuple:
    """Весовые функции: у земли, средний слой, верхний перенос."""
    boundary = _smooth_band(y, -50.0, 230.0, 45.0)
    middle = _smooth_band(y, 150.0, 950.0, 120.0)
    upper = 1.0 / (1.0 + np.exp(-(y - 760.0) / 150.0))
    norm = np.maximum(boundary + middle + upper, 1e-9)
    return boundary / norm, middle / norm, upper / norm


def _eddy_layer_gain(y: np.ndarray, layer: str) -> np.ndarray:
    """Ограничивает вихри их высотной зоной."""
    if layer == "boundary":
        return _smooth_band(y, -30.0, 260.0, 55.0)
    return _smooth_band(y, 130.0, 1150.0, 150.0)


def _axis_index(axis: np.ndarray, value: float) -> tuple:
    """Индекс левой ячейки и доля внутри регулярной оси."""
    value = float(np.clip(value, axis[0], axis[-1]))
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    raw = (value - axis[0]) / step
    idx = int(np.floor(raw))
    idx = int(np.clip(idx, 0, len(axis) - 2))
    frac = float(raw - idx)
    return idx, frac


def _trilinear(
    grid_t: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    fx: float,
    fy: float,
    fz: float,
) -> np.ndarray:
    """Трилинейная интерполяция одного временного слоя."""
    c000 = grid_t[ix,     iy,     iz    ]
    c100 = grid_t[ix + 1, iy,     iz    ]
    c010 = grid_t[ix,     iy + 1, iz    ]
    c110 = grid_t[ix + 1, iy + 1, iz    ]
    c001 = grid_t[ix,     iy,     iz + 1]
    c101 = grid_t[ix + 1, iy,     iz + 1]
    c011 = grid_t[ix,     iy + 1, iz + 1]
    c111 = grid_t[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


# ---------------------------------------------------------------------------
# 6-DoF аэробаллистика
# ---------------------------------------------------------------------------

_AOA_GRID = np.array([0.0, 4.0, 6.0, 8.0, 12.0, 16.0])
_CD_TABLE = np.array([0.30, 0.32, 0.35, 0.40, 0.52, 0.70])
_CL_TABLE = np.array([0.00, 0.035, 0.055, 0.075, 0.115, 0.160])
_CM_AOA_PTS = np.array([0, 4, 6, 8, 12, 16, 25, 45, 60, 90, 180], dtype=float)
_CM_VAL_PTS = np.array([0, -0.02, -0.03, -0.05, -0.10, -0.18, -0.28, -0.45, -0.55, -0.45, -0.35], dtype=float)

_CM_Q = -20.0
_CL_P = -10.0
_CN_R = -20.0
_H_ATM = 8430.0
_RHO0 = 1.225


def _inertia_tensor(m: float, d: float, length: float) -> tuple:
    r = d / 2.0
    i_xx = 0.5 * m * r * r
    i_yy = m * (r * r / 4.0 + length * length / 12.0)
    i_mat = np.diag([i_xx, i_yy, i_yy])
    return i_mat, np.linalg.inv(i_mat)


def _lookup_cl_cd(alpha_deg: float) -> tuple:
    a = float(np.clip(abs(alpha_deg), _AOA_GRID[0], _AOA_GRID[-1]))
    return float(np.interp(a, _AOA_GRID, _CL_TABLE)), float(np.interp(a, _AOA_GRID, _CD_TABLE))


def _cm_alpha(alpha_deg: float) -> float:
    return float(np.interp(abs(alpha_deg), _CM_AOA_PTS, _CM_VAL_PTS))


def _quat_to_rotm(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)  ],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)  ],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=float)


def _initial_quaternion(alpha_rad: float, beta_rad: float) -> np.ndarray:
    """
    Начальная ориентация, при которой продольная ось тела совпадает
    с вектором начальной скорости.

    В мировой СК: X — дальность, Y — высота, Z — боковой снос.
    """
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    cb, sb = np.cos(beta_rad), np.sin(beta_rad)

    # R = R_y(-azimuth) * R_z(elevation)
    rotm = np.array([
        [cb * ca, -cb * sa, -sb],
        [sa,       ca,       0.0],
        [sb * ca, -sb * sa,  cb],
    ], dtype=float)
    return _rotm_to_quat(rotm)


def _rotm_to_quat(rotm: np.ndarray) -> np.ndarray:
    """Преобразует матрицу поворота тело→мир в кватернион (qw, qx, qy, qz)."""
    trace = float(np.trace(rotm))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotm[2, 1] - rotm[1, 2]) / s
        qy = (rotm[0, 2] - rotm[2, 0]) / s
        qz = (rotm[1, 0] - rotm[0, 1]) / s
    elif rotm[0, 0] > rotm[1, 1] and rotm[0, 0] > rotm[2, 2]:
        s = np.sqrt(1.0 + rotm[0, 0] - rotm[1, 1] - rotm[2, 2]) * 2.0
        qw = (rotm[2, 1] - rotm[1, 2]) / s
        qx = 0.25 * s
        qy = (rotm[0, 1] + rotm[1, 0]) / s
        qz = (rotm[0, 2] + rotm[2, 0]) / s
    elif rotm[1, 1] > rotm[2, 2]:
        s = np.sqrt(1.0 + rotm[1, 1] - rotm[0, 0] - rotm[2, 2]) * 2.0
        qw = (rotm[0, 2] - rotm[2, 0]) / s
        qx = (rotm[0, 1] + rotm[1, 0]) / s
        qy = 0.25 * s
        qz = (rotm[1, 2] + rotm[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotm[2, 2] - rotm[0, 0] - rotm[1, 1]) * 2.0
        qw = (rotm[1, 0] - rotm[0, 1]) / s
        qx = (rotm[0, 2] + rotm[2, 0]) / s
        qy = (rotm[1, 2] + rotm[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=float)
    return quat / np.linalg.norm(quat)


def _derivatives(
    state: np.ndarray,
    time_s: float,
    params: dict,
    wind_field: DynamicWindField,
    aero: tuple,
) -> np.ndarray:
    """Правая часть 13 уравнений движения с динамическим полем ветра."""
    s_ref, c_ref, b_ref, r_cp, i_mat, i_inv = aero
    vx, vy, vz = state[3], state[4], state[5]
    qw, qx, qy, qz = state[6], state[7], state[8], state[9]
    pr, qr, rr = state[10], state[11], state[12]

    mass = params["m"]
    gravity = params["g"]

    y_pos = max(float(state[1]), 0.0)
    rho = _RHO0 * np.exp(-y_pos / _H_ATM)

    q_norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if q_norm < 1e-12:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    else:
        qw, qx, qy, qz = qw / q_norm, qx / q_norm, qy / q_norm, qz / q_norm

    r_bw = _quat_to_rotm(qw, qx, qy, qz)
    r_wb = r_bw.T

    wind = wind_field.sample(time_s, state[:3])
    v_rel = np.array([vx, vy, vz]) - wind
    v_abs = float(np.sqrt(v_rel @ v_rel)) + 1e-12

    v_body = r_wb @ v_rel
    normal_speed = float(np.sqrt(v_body[1] * v_body[1] + v_body[2] * v_body[2]))
    alpha_deg = float(np.degrees(np.arctan2(normal_speed, max(v_body[0], 1e-9))))
    cl, cd = _lookup_cl_cd(alpha_deg)

    f_drag = -0.5 * rho * s_ref * cd * v_abs * v_rel
    e_y = np.array([0.0, 1.0, 0.0])
    lift_dir = np.cross(np.cross(v_rel, e_y), v_rel)
    lift_norm = float(np.sqrt(lift_dir @ lift_dir))
    lift_dir = lift_dir / lift_norm if lift_norm > 1e-12 else np.zeros(3)
    f_lift = 0.5 * rho * s_ref * cl * v_abs * v_abs * lift_dir

    a_lin = (f_drag + f_lift + np.array([0.0, -mass * gravity, 0.0])) / mass

    q_dyn = 0.5 * rho * v_abs * v_abs
    q_hat = (qr * c_ref) / (2.0 * v_abs)
    p_hat = (pr * b_ref) / (2.0 * v_abs)
    r_hat = (rr * b_ref) / (2.0 * v_abs)

    mx = q_dyn * s_ref * b_ref * (_CL_P * p_hat)
    my = q_dyn * s_ref * c_ref * (_cm_alpha(alpha_deg) + _CM_Q * q_hat)
    mz = q_dyn * s_ref * b_ref * (_CN_R * r_hat)

    f_aero_body = r_wb @ (f_drag + f_lift)
    m_body = np.array([mx, my, mz]) + np.cross(r_cp, f_aero_body)

    omega = np.array([pr, qr, rr])
    d_omega = i_inv @ (m_body - np.cross(omega, i_mat @ omega))

    dqw = 0.5 * (-pr*qx - qr*qy - rr*qz)
    dqx = 0.5 * (pr*qw + rr*qy - qr*qz)
    dqy = 0.5 * (qr*qw - rr*qx + pr*qz)
    dqz = 0.5 * (rr*qw + qr*qx - pr*qy)

    return np.array([
        vx, vy, vz,
        a_lin[0], a_lin[1], a_lin[2],
        dqw, dqx, dqy, dqz,
        d_omega[0], d_omega[1], d_omega[2],
    ])


def _simulate_one(
    state0: np.ndarray,
    sim_params: dict,
    wind_field: DynamicWindField,
    aero: tuple,
    dt: float,
    max_steps: int,
) -> tuple:
    """Интегрирует один полет методом RK4 с учетом W(t,x,y,z)."""
    state = state0.copy()
    time_s = 0.0
    traj = [state[:3].copy()]
    times = [time_s]

    for _ in range(max_steps):
        prev_state = state.copy()
        prev_time = time_s

        k1 = _derivatives(state, prev_time, sim_params, wind_field, aero)
        k2 = _derivatives(state + 0.5 * dt * k1, prev_time + 0.5 * dt, sim_params, wind_field, aero)
        k3 = _derivatives(state + 0.5 * dt * k2, prev_time + 0.5 * dt, sim_params, wind_field, aero)
        k4 = _derivatives(state + dt * k3, prev_time + dt, sim_params, wind_field, aero)

        state = state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        time_s = prev_time + dt

        if state[1] <= 0.0 and len(traj) > 2:
            y_prev = float(prev_state[1])
            dy = y_prev - float(state[1])
            frac = y_prev / dy if abs(dy) > 1e-12 else 0.0
            frac = float(np.clip(frac, 0.0, 1.0))
            landing = prev_state[:3] + frac * (state[:3] - prev_state[:3])
            landing[1] = 0.0
            traj.append(landing)
            times.append(prev_time + frac * dt)
            break

        traj.append(state[:3].copy())
        times.append(time_s)

    return np.array(traj), np.array(times)


# ---------------------------------------------------------------------------
# Данные для анимации
# ---------------------------------------------------------------------------

def _make_visual_wind_frames(
    wind_field: DynamicWindField,
    frame_times: np.ndarray,
    trajectories: list,
) -> tuple:
    """Готовит подвижные потоковые стрелки для 3D-анимации."""
    all_points = np.vstack(trajectories)
    x_high = min(wind_field.x_axis[-1] * 0.92, max(float(np.max(all_points[:, 0])) * 1.08, 800.0))
    y_high = min(wind_field.y_axis[-1] * 0.85, max(float(np.max(all_points[:, 1])) * 1.12, 600.0))
    z_high = min(abs(wind_field.z_axis[-1]) * 0.75, max(float(np.max(np.abs(all_points[:, 2]))) * 1.50, 180.0))

    x_points = np.linspace(0.0, x_high, 7)
    y_points = np.linspace(80.0, y_high, 4)
    z_points = np.linspace(-z_high, z_high, 5)
    xx, yy, zz = np.meshgrid(x_points, y_points, z_points, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    frames = np.zeros((len(frame_times), len(points), 3), dtype=float)
    for i, t in enumerate(frame_times):
        frames[i] = wind_field.sample_points(float(t), points)

    rng = np.random.default_rng(wind_field.seed + 5000)
    particle_count = 160
    particles = np.column_stack([
        rng.uniform(0.0, x_high, particle_count),
        rng.uniform(80.0, y_high, particle_count),
        rng.uniform(-z_high, z_high, particle_count),
    ])
    particle_frames = np.zeros((len(frame_times), particle_count, 3), dtype=float)
    particle_vectors = np.zeros_like(particle_frames)
    particle_frames[0] = particles
    particle_vectors[0] = wind_field.sample_points(float(frame_times[0]), particles)

    for i in range(1, len(frame_times)):
        t_prev = float(frame_times[i - 1])
        t_curr = float(frame_times[i])
        dt_frame = max(t_curr - t_prev, 0.05)
        vectors = wind_field.sample_points(t_prev, particles)

        # Увеличенный визуальный шаг делает движение потока читаемым на масштабе
        # километровой траектории, не меняя физический расчет снаряда.
        particles = particles + vectors * dt_frame * 18.0
        particles[:, 0] = _wrap(particles[:, 0], 0.0, x_high)
        particles[:, 1] = _wrap(particles[:, 1], 80.0, y_high)
        particles[:, 2] = _wrap(particles[:, 2], -z_high, z_high)

        particle_frames[i] = particles
        particle_vectors[i] = wind_field.sample_points(t_curr, particles)

    return points, frames, particle_frames, particle_vectors


def _wrap(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Циклический перенос координат потоковых частиц внутри области."""
    width = high - low
    return low + np.mod(values - low, width)


def _trajectory_position_at_time(
    trajectory: np.ndarray,
    times: np.ndarray,
    time_s: float,
) -> np.ndarray:
    """Интерполирует положение снаряда в момент времени кадра."""
    if time_s <= float(times[0]):
        return trajectory[0].copy()
    if time_s >= float(times[-1]):
        return trajectory[-1].copy()

    idx = int(np.searchsorted(times, time_s, side="right"))
    t0 = float(times[idx - 1])
    t1 = float(times[idx])
    frac = (time_s - t0) / max(t1 - t0, 1e-12)
    return trajectory[idx - 1] + frac * (trajectory[idx] - trajectory[idx - 1])


def _make_projectile_wind_frames(
    wind_fields: list,
    frame_times: np.ndarray,
    trajectories: list,
    times_list: list,
) -> tuple:
    """Готовит положение снаряда и локальный ветер на каждом кадре записи."""
    positions = np.zeros((len(trajectories), len(frame_times), 3), dtype=float)
    winds = np.zeros_like(positions)
    for run_i, (trajectory, times) in enumerate(zip(trajectories, times_list)):
        wind_field = wind_fields[run_i]
        for frame_i, time_s in enumerate(frame_times):
            pos = _trajectory_position_at_time(trajectory, times, float(time_s))
            positions[run_i, frame_i] = pos
            winds[run_i, frame_i] = wind_field.sample(float(time_s), pos)
    return positions, winds


def _wind_ensemble_spread(scenario: str) -> float:
    """Насколько сильно отличаются метеокарты между пусками."""
    if scenario == "storm":
        return 1.35
    if scenario == "vortex":
        return 1.25
    if scenario == "turbulent":
        return 1.10
    if scenario == "shear":
        return 0.55
    return 0.20


# ---------------------------------------------------------------------------
# Публичный расчет
# ---------------------------------------------------------------------------

def calculate(params: dict) -> tuple:
    """
    Ансамблевый расчет итоговой модели.

    Returns:
        (result, True, label), где result — словарь со специальным форматом:
          kind          — "dynamic_6dof_wind_field";
          trajectories  — list[ndarray(K_i, 3)];
          times         — list[ndarray(K_i)];
          frame_times   — ndarray(F);
          wind_points   — ndarray(P, 3);
          wind_frames   — ndarray(F, P, 3).
    """
    n_runs = int(params.get("N", 24))
    v0 = float(params.get("v0", 250.0))
    v0_std = float(params.get("v0_std", 2.0))
    angle_deg = float(params.get("angle", 45.0))
    angle_std = float(params.get("angle_std", 0.25))
    azim_deg = float(params.get("azimuth", 0.0))
    azim_std = float(params.get("azimuth_std", 0.20))
    scenario = params.get("scenario", "storm")
    wind_strength = float(params.get("Wind_Strength", 10.0))
    turbulence = float(params.get("Turbulence", 3.0))
    vortex_x_shift = float(params.get("Vortex_X_Shift", 0.0))
    vortex_z_shift = float(params.get("Vortex_Z_Shift", 0.0))
    vortex_strength_factor = float(params.get("Vortex_Strength_Factor", 1.0))
    mass = float(params.get("m", 0.85))
    d_ref = float(params.get("D", 0.04))
    length = float(params.get("L", 0.18))
    seed = int(float(params.get("seed", 42)))

    rng = np.random.default_rng(seed)

    r = d_ref / 2.0
    s_ref = np.pi * r * r
    c_ref = length
    b_ref = d_ref
    r_cp = np.array([0.25 * length, 0.0, 0.0])
    i_mat, i_inv = _inertia_tensor(mass, d_ref, length)
    aero = (s_ref, c_ref, b_ref, r_cp, i_mat, i_inv)

    scenario_name = _normalize_scenario(scenario)
    t_field = 95.0
    field_kwargs = {
        "scenario": scenario_name,
        "strength": wind_strength,
        "turbulence": turbulence,
        "x_max": 8500.0,
        "y_max": 2600.0,
        "z_span": 1600.0,
        "t_max": t_field,
        "vortex_x_shift": vortex_x_shift,
        "vortex_z_shift": vortex_z_shift,
        "vortex_strength_factor": vortex_strength_factor,
    }
    spread = _wind_ensemble_spread(scenario_name)

    base_params = {"m": mass, "g": 9.81}
    dt = 0.02
    max_steps = int(t_field / dt)

    all_trajectories = []
    all_times = []
    wind_fields = []
    for run_i in range(n_runs):
        wind_field = DynamicWindField(
            **field_kwargs,
            seed=seed + 1000 + 37 * run_i,
            ensemble_index=run_i,
            ensemble_spread=spread,
        )
        wind_fields.append(wind_field)

        v0_i = max(1.0, float(rng.normal(v0, v0_std)))
        alpha = np.radians(float(rng.normal(angle_deg, angle_std)))
        beta = np.radians(float(rng.normal(azim_deg, azim_std)))

        vx0 = v0_i * np.cos(alpha) * np.cos(beta)
        vy0 = v0_i * np.sin(alpha)
        vz0 = v0_i * np.cos(alpha) * np.sin(beta)
        q0 = _initial_quaternion(alpha, beta)

        state0 = np.zeros(13)
        state0[0:3] = [0.0, 0.0, 0.0]
        state0[3:6] = [vx0, vy0, vz0]
        state0[6:10] = q0

        traj, times = _simulate_one(state0, base_params, wind_field, aero, dt, max_steps)
        all_trajectories.append(traj)
        all_times.append(times)

    reference_wind_field = wind_fields[0]
    max_time = max(float(t[-1]) for t in all_times)
    frame_times = np.linspace(0.0, max_time, 80)
    wind_points, wind_frames, wind_particles, wind_particle_vectors = (
        _make_visual_wind_frames(reference_wind_field, frame_times, all_trajectories)
    )
    projectile_positions, projectile_winds = _make_projectile_wind_frames(
        wind_fields, frame_times, all_trajectories, all_times
    )

    label = (
        f"Итоговая 6-DoF: N={n_runs}, поле={scenario_name}+МК, "
        f"Wind~{wind_strength:.0f}м/с, D={d_ref*1000:.0f}мм"
    )
    result = {
        "kind": "dynamic_6dof_wind_field",
        "trajectories": all_trajectories,
        "times": all_times,
        "frame_times": frame_times,
        "wind_points": wind_points,
        "wind_frames": wind_frames,
        "wind_particles": wind_particles,
        "wind_particle_vectors": wind_particle_vectors,
        "projectile_positions": projectile_positions,
        "projectile_winds": projectile_winds,
        "wind_field": reference_wind_field,
        "wind_fields": wind_fields,
        "wind_layers": WIND_LAYERS,
        "label": label,
        "metadata": {
            "scenario": scenario_name,
            "wind_strength": wind_strength,
            "turbulence": turbulence,
            "vortex_x_shift": vortex_x_shift,
            "vortex_z_shift": vortex_z_shift,
            "vortex_strength_factor": vortex_strength_factor,
            "seed": seed,
            "dt": dt,
            "wind_ensemble_spread": spread,
        },
    }
    return result, True, label
