"""
Модуль визуализации результатов Монте-Карло баллистики.

Экспортирует:
  render_monte_carlo_6dof()     — полный 3D-рендер 6-DoF МК (вызывается из main.py)
  compute_mean_trajectory()     — среднестатистическая 3D-траектория
  compute_mean_trajectory_2d()  — среднестатистическая 2D-траектория (для модели 3)
  compute_impact_ellipse()      — параметры эллипса рассеивания (XZ-плоскость)

Математика намеренно отделена от matplotlib-зависимостей,
чтобы её можно было тестировать независимо от GUI.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Вычислительные функции (без зависимости от matplotlib)
# ---------------------------------------------------------------------------

def compute_mean_trajectory(trajectories: list) -> np.ndarray:
    """
    Средняя 3D-траектория как E[r(t)] по ансамблю пусков.

    Траектории разной длины выравниваются до максимальной:
    короткие «замораживаются» в точке падения (тело лежит на земле).
    Это физически корректнее нулевого паддинга.

    Args:
        trajectories: Список ndarray формы (K_i, 3).

    Returns:
        ndarray (max_K, 3) — усреднённая траектория.
    """
    max_len = max(t.shape[0] for t in trajectories)
    acc = np.zeros((max_len, 3), dtype=float)
    for traj in trajectories:
        k = traj.shape[0]
        acc[:k] += traj
        if k < max_len:
            acc[k:] += traj[-1]   # заморозка в точке падения
    return acc / len(trajectories)


def compute_mean_trajectory_2d(trajectories: list) -> np.ndarray:
    """
    Средняя 2D-траектория (x, y) для модели 3 (Монте-Карло 2D).

    Алгоритм аналогичен compute_mean_trajectory, но для массивов (K_i, 2).

    Args:
        trajectories: Список ndarray формы (K_i, 2).

    Returns:
        ndarray (max_K, 2).
    """
    max_len = max(t.shape[0] for t in trajectories)
    acc = np.zeros((max_len, 2), dtype=float)
    for traj in trajectories:
        k = traj.shape[0]
        acc[:k] += traj
        if k < max_len:
            acc[k:] += traj[-1]
    return acc / len(trajectories)


def compute_impact_ellipse(
    trajectories: list,
    n_sigma: float,
    n_points: int = 200,
) -> tuple:
    """
    Параметры и контур эллипса рассеивания точек падения в плоскости XZ.

    Центр эллипса — точка падения средней траектории (физически: куда
    целится орудие). Полуоси = n_sigma * std по каждой оси.

    Args:
        trajectories: Список траекторий (K_i, 3).
        n_sigma:      Множитель СКО (1, 2 или 3).
        n_points:     Количество точек контура.

    Returns:
        (ex, ez, center_x, center_z, std_x, std_z)
    """
    impacts = np.array([t[-1] for t in trajectories])   # (N, 3)
    x_hits, z_hits = impacts[:, 0], impacts[:, 2]

    mean_traj = compute_mean_trajectory(trajectories)
    center_x = float(mean_traj[-1, 0])
    center_z = float(mean_traj[-1, 2])

    std_x = float(np.std(x_hits)) if len(x_hits) > 1 else 1.0
    std_z = float(np.std(z_hits)) if len(z_hits) > 1 else 1.0

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    ex = center_x + n_sigma * std_x * np.cos(theta)
    ez = center_z + n_sigma * std_z * np.sin(theta)

    return ex, ez, center_x, center_z, std_x, std_z


# ---------------------------------------------------------------------------
# Рендерер 6-DoF МК (зависит от matplotlib)
# ---------------------------------------------------------------------------

def render_monte_carlo_6dof(ax: 'Axes3D', trajectories: list) -> None:
    """
    Рисует полный результат 6-DoF Монте-Карло на 3D-осях matplotlib.

    Слои (снизу вверх):
      1. Облако реализаций (прозрачно-синие линии).
      2. Точки падения (серые маркеры на плоскости y=0).
      3. Эллипсы рассеивания 1σ / 2σ / 3σ в плоскости XZ (мишень).
      4. Средняя траектория (красная жирная линия).
      5. Крест-маркер центра рассеивания + подпись СКО.

    Порядок осей matplotlib 3D: plot(X, Y, Z) — у нас:
      X = дальность (data[:,0]), Y = боковой снос (data[:,2]), Z = высота (data[:,1])

    Args:
        ax:           Объект Axes3D.
        trajectories: Список ndarray (K_i, 3).
    """
    n = len(trajectories)

    # --- 1. Облако ---
    for traj in trajectories:
        ax.plot(
            traj[:, 0], traj[:, 2], traj[:, 1],
            color='dodgerblue', alpha=0.12, lw=0.7,
        )

    # --- 2. Точки падения ---
    impacts = np.array([t[-1] for t in trajectories])
    ax.scatter(
        impacts[:, 0], impacts[:, 2], np.zeros(n),
        color='steelblue', s=12, alpha=0.5, zorder=3,
        label=f"Точки падения (N={n})",
    )

    # --- 3. Эллипсы (мишень) ---
    for n_sig, color, ls, lw, pct in [
        (1, 'limegreen',  '-',  1.8, '68%'),
        (2, 'darkorange', '--', 2.0, '95%'),
        (3, 'crimson',    ':',  2.2, '99.7%'),
    ]:
        ex, ez, *_ = compute_impact_ellipse(trajectories, n_sig)
        ax.plot(ex, ez, np.zeros_like(ex),
                color=color, lw=lw, ls=ls,
                label=f"Эллипс {n_sig}σ ({pct})")

    # --- 4. Средняя траектория ---
    mean_traj = compute_mean_trajectory(trajectories)
    ax.plot(
        mean_traj[:, 0], mean_traj[:, 2], mean_traj[:, 1],
        color='red', lw=2.5, zorder=5,
        label="Средняя траектория (МК)",
    )

    # --- 5. Центр рассеивания ---
    _, _, cx, cz, sx, sz = compute_impact_ellipse(trajectories, 1)
    ax.scatter(
        [cx], [cz], [0.0],
        color='red', s=90, marker='X', zorder=6,
        label=f"Центр  σ_x={sx:.1f}м  σ_z={sz:.1f}м",
    )

    ax.set_xlabel("Дальность X, м")
    ax.set_ylabel("Боковой снос Z, м")
    ax.set_zlabel("Высота Y, м")
    ax.view_init(elev=20, azim=-55)