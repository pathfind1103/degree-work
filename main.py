"""
Главный модуль приложения для расчёта траекторий (Контроллер, MVC).

Форматы возврата из calculate():
  Модели 1, 2  → (x_arr, y_arr, label),    is_3d=False  — одна 2D-кривая
  Модель  3    → (list[ndarray(K,2)], None, label)       — 2D МК пучок
  Модели 4, 5  → (list[ndarray(K,3)], True, label)       — 3D пучок
  Модель  6    → (list[ndarray(K,3)], True, label)        — 6DoF + эллипсы

Критические исправления по сравнению с предыдущей версией:
  - Вместо main_fig.clf() используем безопасную замену subplot'а через
    figure.add_subplot() с предварительным удалением старых осей.
    clf() на Windows с Qt-бэкендом может вызвать 0xC0000409.
  - formula_fig.savefig() теперь использует FigureCanvasAgg явно —
    без привлечения pyplot-рендерера.
  - Роутинг форматов результата чётко разделён по типу данных.
"""

import sys
import importlib.util
import io
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt6.QtWidgets import QApplication, QLineEdit
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QTimer

from ui_main import MainWindowUI
from mc_renderer import (
    render_monte_carlo_6dof,
    compute_mean_trajectory,
    compute_mean_trajectory_2d,
    compute_impact_ellipse,
)


DEMO_SCENARIO_PRESETS = {
    "Базовый эталон": {
        "N": "40",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "calm",
        "Wind_Strength": "4.0",
        "Turbulence": "0.5",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Высотный сдвиг": {
        "N": "50",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "55.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "shear",
        "Wind_Strength": "22.0",
        "Turbulence": "1.5",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Турбулентный слой": {
        "N": "50",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "turbulent",
        "Wind_Strength": "18.0",
        "Turbulence": "7.0",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Штормовой фронт": {
        "N": "50",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "storm",
        "Wind_Strength": "35.0",
        "Turbulence": "5.0",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Вихревая зона": {
        "N": "40",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "vortex",
        "Wind_Strength": "30.0",
        "Turbulence": "3.0",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Ошибки начальных условий": {
        "N": "80",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "calm",
        "Wind_Strength": "3.0",
        "Turbulence": "0.5",
        "m": "0.85",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Лёгкий снаряд": {
        "N": "50",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "turbulent",
        "Wind_Strength": "20.0",
        "Turbulence": "5.0",
        "m": "0.50",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
    "Тяжёлый снаряд": {
        "N": "50",
        "v0": "250.0",
        "v0_std": "0.0",
        "angle": "45.0",
        "angle_std": "0.0",
        "azimuth": "0.0",
        "azimuth_std": "0.0",
        "scenario": "turbulent",
        "Wind_Strength": "20.0",
        "Turbulence": "5.0",
        "m": "2.00",
        "D": "0.04",
        "L": "0.18",
        "seed": "42",
    },
}


class TrajectoryApp(MainWindowUI):
    """
    Контроллер приложения.

    Управляет загрузкой моделей, обновлением UI и запуском расчётов.
    """

    def __init__(self):
        """Инициализация: подключаем сигналы и загружаем модели."""
        super().__init__()
        self.models: dict = {}
        self.param_widgets: dict = {}
        self.animation_data: dict | None = None
        self.animation_frame: int = 0
        self._resume_animation_after_drag = False
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(120)

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.scenario_preset_combo.currentIndexChanged.connect(self._on_scenario_preset_changed)
        self.calc_btn.clicked.connect(self.run_calculation)
        self.animation_timer.timeout.connect(self._advance_animation_frame)
        self.anim_play_btn.clicked.connect(self._toggle_animation)
        self.anim_prev_btn.clicked.connect(self._previous_animation_frame)
        self.anim_next_btn.clicked.connect(self._next_animation_frame)
        self.anim_slider.valueChanged.connect(self._on_animation_slider_changed)
        self.anim_run_combo.currentIndexChanged.connect(self._on_animation_run_changed)
        self.anim_zoom_slider.valueChanged.connect(self._on_animation_zoom_changed)
        self.anim_density_combo.currentIndexChanged.connect(self._on_animation_density_changed)
        self.main_canvas.mpl_connect("button_press_event", self._on_canvas_mouse_press)
        self.main_canvas.mpl_connect("button_release_event", self._on_canvas_mouse_release)

        self._load_models()

    # ------------------------------------------------------------------
    # Загрузка моделей
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """
        Сканирует папку models/ и динамически загружает *.py модули.

        Требования к модулю: get_name(), get_info(), get_params(), calculate().
        Файлы без этих функций пропускаются с диагностикой в консоль.
        """
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)

        loaded_models = []
        for file_path in sorted(models_dir.glob("*.py")):
            if file_path.name == "__init__.py":
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    file_path.stem, file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                model_name = module.get_name()
                loaded_models.append((model_name, module))
            except (ImportError, AttributeError) as err:
                print(f"[Загрузчик] Пропущен {file_path.name}: {err}")

        for model_name, module in sorted(loaded_models, key=self._model_sort_key):
            self.models[model_name] = module
            self.model_combo.addItem(model_name)

    @staticmethod
    def _model_sort_key(item: tuple) -> tuple:
        """
        Сортирует модели по числовому префиксу в get_name().

        Если префикса нет, такая модель уходит в конец списка, но всё равно
        остаётся доступной.
        """
        model_name, _ = item
        prefix = model_name.split(".", 1)[0]
        try:
            return int(prefix), model_name
        except ValueError:
            return 10_000, model_name

    # ------------------------------------------------------------------
    # Обновление UI при смене модели
    # ------------------------------------------------------------------

    def _on_model_changed(self) -> None:
        """
        При выборе новой модели обновляет описание, формулу и параметры.

        Рендеринг формулы: Figure → FigureCanvasAgg → BytesIO → QPixmap.
        FigureCanvasAgg используется явно — без pyplot — чтобы исключить
        краш нативного рендерера на Windows (0xC0000409).
        """
        model_name = self.model_combo.currentText()
        if not model_name or model_name not in self.models:
            return

        self._disable_animation_controls()

        info = self.models[model_name].get_info()

        # 1. Текстовое описание + расшифровка переменных
        vars_lines = "\n".join(
            f"• {k} — {v}"
            for k, v in info.get("parameters_info", {}).items()
        )
        self.info_display.setText(
            f"{info.get('description', '')}\n\nОбозначения:\n{vars_lines}"
        )

        # 2. Рендеринг LaTeX через Agg → PNG → QPixmap.
        #    ВАЖНО: каждый раз создаём НОВУЮ временную Figure и новый
        #    FigureCanvasAgg. Никогда не привязываем Agg к self.formula_fig,
        #    которая уже принадлежит Qt-layout — это вызывает 0xC0000409.
        formula_text = info.get("formula", "")
        try:
            from matplotlib.figure import Figure as _Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as _Agg

            tmp_fig = _Figure(figsize=(3.8, 1.8), facecolor='#f0f0f0')
            tmp_canvas = _Agg(tmp_fig)
            tmp_ax = tmp_fig.add_subplot(111)
            tmp_ax.axis('off')
            tmp_ax.text(
                0.5, 0.5,
                formula_text,
                fontsize=11,
                ha='center',
                va='center',
                math_fontfamily='cm',
                transform=tmp_ax.transAxes,
            )
            buf = io.BytesIO()
            tmp_canvas.print_figure(
                buf,
                format='png',
                dpi=96,
                facecolor='#f0f0f0',
                bbox_inches='tight',
                pad_inches=0.12,
            )
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            buf.close()
            if not pixmap.isNull():
                self.formula_label.setPixmap(pixmap)
                self.formula_label.setText("")
            else:
                self.formula_label.setText(formula_text)
        except Exception as exc:
            # Fallback: показываем сырой текст — лучше, чем краш
            self.formula_label.setPixmap(QPixmap())
            self.formula_label.setText(formula_text)
            print(f"[Формула] {exc}")

        # 3. Пересборка формы параметров
        self._clear_params()
        for p_name, p_val in self.models[model_name].get_params().items():
            edit = QLineEdit(str(p_val))
            self.params_layout.addRow(p_name, edit)
            self.param_widgets[p_name] = edit

        self._configure_scenario_presets(model_name)

    def _clear_params(self) -> None:
        """Удаляет все виджеты параметров из формы."""
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.param_widgets.clear()

    def _configure_scenario_presets(self, model_name: str) -> None:
        """Показывает список демонстрационных сценариев только для модели 8."""
        is_final_model = model_name.startswith("8.")

        self.scenario_preset_combo.blockSignals(True)
        self.scenario_preset_combo.clear()
        if is_final_model:
            self.scenario_preset_combo.addItem("Выбрать сценарий...", None)
            for preset_name, values in DEMO_SCENARIO_PRESETS.items():
                self.scenario_preset_combo.addItem(preset_name, values)
        self.scenario_preset_combo.blockSignals(False)

        self.scenario_preset_label.setVisible(is_final_model)
        self.scenario_preset_combo.setVisible(is_final_model)

    def _on_scenario_preset_changed(self, index: int) -> None:
        """Заполняет параметры модели 8 выбранным демонстрационным сценарием."""
        preset = self.scenario_preset_combo.itemData(index)
        if not preset:
            return

        for name, value in preset.items():
            widget = self.param_widgets.get(name)
            if widget is not None:
                widget.setText(str(value))

    # ------------------------------------------------------------------
    # Запуск расчёта
    # ------------------------------------------------------------------

    def run_calculation(self) -> None:
        """
        Читает параметры из UI, вызывает calculate() активной модели,
        определяет формат результата и делегирует рендеринг.

        Безопасная замена осей: вместо clf() используем
        fig.clear() + add_subplot(), что не разрушает связь canvas↔figure.
        """
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        params = {
            name: widget.text()
            for name, widget in self.param_widgets.items()
        }

        self._disable_animation_controls()

        try:
            raw = self.models[model_name].calculate(params)
        except Exception as err:
            self.info_display.setText(f"ОШИБКА РАСЧЁТА:\n{err}")
            return

        # Определяем формат ответа модели:
        #   Модели 1, 2 → (x_arr, y_arr, label_str)         — 3 элемента, нет флага
        #   Модели 3..6 → (data, is_3d_flag, label_str)      — 3 элемента, флаг bool/None
        # Различаем по типу второго элемента: numpy array → 2D-кривая,
        # bool/None → МК или 3D режим.
        import numpy as np
        if len(raw) == 3 and isinstance(raw[1], np.ndarray):
            # Формат модулей 1 и 2: (x, y, label)
            result = (raw[0], raw[1])
            is_3d = False
            label = raw[2]
        else:
            result, is_3d, label = raw

        self._remove_main_axes()

        dynamic_render = False

        if is_3d:
            self.main_ax = self.main_fig.add_subplot(111, projection='3d')
            if isinstance(result, dict) and result.get("kind") == "dynamic_6dof_wind_field":
                dynamic_render = True
                self._render_dynamic_wind_model(result, label)
            elif model_name.startswith("6."):
                self._render_6dof_mc(result)
            else:
                self._render_3d_bundle(result, label)
        else:
            self.main_ax = self.main_fig.add_subplot(111)
            if isinstance(result, list):
                # Модель 3: 2D МК — список массивов (K_i, 2)
                self._render_2d_mc(result, label)
            else:
                # Модели 1, 2: одна кривая — кортеж (x_arr, y_arr)
                self._render_2d_single(result, label)

        if not dynamic_render:
            self.main_ax.set_title(f"Результат: {model_name}", pad=10)
        self.main_canvas.draw()

    # ------------------------------------------------------------------
    # Рендереры (приватные)
    # ------------------------------------------------------------------

    def _remove_main_axes(self) -> None:
        """
        Безопасно удаляет старые оси перед новым расчётом.

        Axes3D регистрирует собственные обработчики мыши в canvas. После
        ax.remove() эти callback'и могут остаться висеть и на button_release
        обратиться к уже удалённой оси. Поэтому перед удалением отключаем
        callback'и, привязанные именно к этой оси.
        """
        for ax in self.main_fig.axes[:]:
            self._disconnect_axis_callbacks(ax)
            ax.remove()

    def _disconnect_axis_callbacks(self, ax) -> None:
        """Отключает matplotlib callback'и, bound-method которых принадлежит ax."""
        callbacks = getattr(self.main_canvas, "callbacks", None)
        registry = getattr(callbacks, "callbacks", None)
        if callbacks is None or registry is None:
            return

        for signal in (
            "motion_notify_event",
            "button_press_event",
            "button_release_event",
        ):
            for cid, proxy in list(registry.get(signal, {}).items()):
                try:
                    func = proxy()
                except TypeError:
                    func = proxy
                if getattr(func, "__self__", None) is ax:
                    callbacks.disconnect(cid)

    def _render_2d_single(self, result: tuple, label: str) -> None:
        """
        Одна 2D-траектория (модели 1 и 2).

        Args:
            result: Кортеж (x_arr, y_arr) — numpy-массивы координат.
            label:  Строка для легенды.
        """
        x_arr, y_arr = result
        self.main_ax.plot(x_arr, y_arr, lw=2, color='royalblue', label=label)
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Высота Y, м")
        self.main_ax.legend()
        self.main_ax.grid(True, alpha=0.4)

    def _render_2d_mc(self, result: list, label: str) -> None:
        """
        Пучок 2D-траекторий Монте-Карло + средняя линия (модель 3).

        Args:
            result: Список массивов (K_i, 2).
            label:  Строка для легенды.
        """
        for traj in result:
            self.main_ax.plot(
                traj[:, 0], traj[:, 1],
                color='steelblue', alpha=0.25, lw=0.8,
            )
        mean = compute_mean_trajectory_2d(result)
        self.main_ax.plot(
            mean[:, 0], mean[:, 1],
            color='red', lw=2,
            label=f"Среднее ({label})",
        )
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Высота Y, м")
        self.main_ax.legend()
        self.main_ax.grid(True, alpha=0.4)

    def _render_3d_bundle(self, result: list, label: str) -> None:
        """
        3D-пучок траекторий (модели 4 и 5).

        Args:
            result: Список массивов (K_i, 3), столбцы: x, y, z.
            label:  Строка для легенды.
        """
        self.main_ax.view_init(elev=25, azim=-60)
        for traj in result:
            self.main_ax.plot(
                traj[:, 0], traj[:, 2], traj[:, 1],
                color='crimson', alpha=0.25, lw=0.8,
            )
        self.main_ax.plot([], [], [], color='crimson', lw=1.5, label=label)
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Боковой снос Z, м")
        self.main_ax.set_zlabel("Высота Y, м")
        self.main_ax.legend(fontsize=8)

    def _render_6dof_mc(self, result: list) -> None:
        """
        6-DoF Монте-Карло: облако, средняя траектория, эллипсы 1σ/2σ/3σ.

        Делегирует в mc_renderer.render_monte_carlo_6dof.

        Args:
            result: Список массивов (K_i, 3).
        """
        render_monte_carlo_6dof(self.main_ax, result)
        self.main_ax.legend(loc='upper left', fontsize=7)

    def _render_dynamic_wind_model(self, result: dict, label: str) -> None:
        """
        Итоговая модель: общий результат + запись выбранного пуска.

        Args:
            result: Словарь специального формата из final_dynamic_6dof.py.
            label:  Строка для заголовка/легенды.
        """
        self.animation_data = result
        self.animation_data["display_label"] = label
        self.animation_frame = 0
        frame_count = len(result["frame_times"])

        self.anim_run_combo.blockSignals(True)
        self.anim_run_combo.clear()
        self.anim_run_combo.addItem("Итог")
        for i in range(len(result["trajectories"])):
            self.anim_run_combo.addItem(f"Запуск {i + 1}")
        self.anim_run_combo.setCurrentIndex(0)
        self.anim_run_combo.blockSignals(False)

        self.anim_slider.blockSignals(True)
        self.anim_slider.setMinimum(0)
        self.anim_slider.setMaximum(max(frame_count - 1, 0))
        self.anim_slider.setValue(0)
        self.anim_slider.blockSignals(False)
        self.anim_zoom_slider.setValue(35)
        self.anim_zoom_label.setText("Zoom 35%")
        self.anim_density_combo.setCurrentText("80")

        self.animation_panel.setVisible(True)
        self.anim_play_btn.setText("Пуск")
        self._set_animation_replay_enabled(False)
        self._render_dynamic_wind_summary()

    def _render_dynamic_wind_summary(self) -> None:
        """Рисует итоговую картину: все пуски, средняя, мишень и эллипсы."""
        if not self.animation_data:
            return

        trajectories = self.animation_data["trajectories"]
        label = self.animation_data.get("display_label", "Итоговая модель")
        elev, azim = self._current_3d_view()

        self.main_ax.cla()
        self.main_ax.view_init(elev=elev, azim=azim)

        for traj in trajectories:
            self.main_ax.plot(
                traj[:, 0], traj[:, 2], traj[:, 1],
                color="dodgerblue", alpha=0.18, lw=0.8,
            )
        self.main_ax.plot([], [], [], color="dodgerblue", alpha=0.45, lw=1.2,
                          label=f"Пуски МК (N={len(trajectories)})")

        impacts = np.array([traj[-1] for traj in trajectories])
        self.main_ax.scatter(
            impacts[:, 0], impacts[:, 2], np.zeros(len(impacts)),
            color="steelblue", s=14, alpha=0.55, label="Точки падения",
        )

        for n_sig, color, ls, lw, pct in [
            (1, "limegreen", "-", 1.8, "68%"),
            (2, "darkorange", "--", 2.0, "95%"),
            (3, "crimson", ":", 2.2, "99.7%"),
        ]:
            ex, ez, *_ = compute_impact_ellipse(trajectories, n_sig)
            self.main_ax.plot(
                ex, ez, np.zeros_like(ex),
                color=color, lw=lw, ls=ls,
                label=f"Эллипс {n_sig}σ ({pct})",
            )

        mean = compute_mean_trajectory(trajectories)
        self.main_ax.plot(
            mean[:, 0], mean[:, 2], mean[:, 1],
            color="red", lw=2.4, label="Средняя траектория",
        )

        _, _, cx, cz, sx, sz = compute_impact_ellipse(trajectories, 1)
        self.main_ax.scatter(
            [cx], [cz], [0.0],
            color="red", s=80, marker="X",
            label=f"Центр σ_x={sx:.1f}м σ_z={sz:.1f}м",
        )

        xlim, ylim, zlim = self._set_dynamic_axes_limits(trajectories)
        self._draw_wind_layers(xlim, ylim, zlim, annotate=True)
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Боковой снос Z, м")
        self.main_ax.set_zlabel("Высота Y, м")
        self.main_ax.set_title(f"{label}\nИтоговый результат по всем пускам", pad=10)
        self.main_ax.legend(loc="upper left", fontsize=7)

        self.anim_time_label.setText("Итог")
        self.main_canvas.draw_idle()

    def _render_dynamic_wind_frame(self, frame_index: int) -> None:
        """Рисует один кадр записи выбранного пуска."""
        if not self.animation_data:
            return
        run_index = self.anim_run_combo.currentIndex() - 1
        if run_index < 0:
            self._render_dynamic_wind_summary()
            return

        frame_times = self.animation_data["frame_times"]
        frame_index = int(np.clip(frame_index, 0, len(frame_times) - 1))
        self.animation_frame = frame_index
        t_curr = float(frame_times[frame_index])

        trajectories = self.animation_data["trajectories"]
        times_list = self.animation_data["times"]
        traj = trajectories[run_index]
        times = times_list[run_index]
        current_pos = self.animation_data["projectile_positions"][run_index, frame_index]
        local_wind = self.animation_data["projectile_winds"][run_index, frame_index]
        wind_fields = self.animation_data.get("wind_fields")
        wind_field = wind_fields[run_index] if wind_fields else self.animation_data["wind_field"]
        elev, azim = self._current_3d_view()

        self.main_ax.cla()
        self.main_ax.view_init(elev=elev, azim=azim)

        # Полная выбранная траектория как полупрозрачная "запись на плёнке".
        self.main_ax.plot(
            traj[:, 0], traj[:, 2], traj[:, 1],
            color="lightsteelblue", alpha=0.35, lw=0.8, label="Полная траектория",
        )

        k = int(np.searchsorted(times, t_curr, side="right"))
        if k > 1:
            segment = traj[:k]
            self.main_ax.plot(
                segment[:, 0], segment[:, 2], segment[:, 1],
                color="navy", alpha=0.95, lw=2.0, label=f"Запуск {run_index + 1}",
            )
            self.main_ax.scatter(
                current_pos[0], current_pos[2], current_pos[1],
                color="crimson", s=42, alpha=0.9, label="Текущее положение",
            )

        if t_curr >= float(times[-1]):
            self.main_ax.scatter(
                [traj[-1, 0]], [traj[-1, 2]], [0.0],
                color="black", s=38, marker="x", label="Падение",
            )

        xlim, ylim, zlim = self._set_dynamic_axes_limits([traj], focus=current_pos)
        self._draw_wind_layers(xlim, ylim, zlim, annotate=False)
        wind_points, grid_step = self._build_wind_display_grid(xlim, ylim, zlim)
        wind_vectors = wind_field.sample_points(t_curr, wind_points)

        speed = np.linalg.norm(wind_vectors, axis=1)
        max_speed = max(float(np.max(speed)), 1e-9)
        colors = self._wind_colors(speed / max_speed)
        scaled_vectors = self._scale_wind_vectors(wind_vectors, grid_step)
        self.main_ax.quiver(
            wind_points[:, 0], wind_points[:, 2], wind_points[:, 1],
            scaled_vectors[:, 0], scaled_vectors[:, 2], scaled_vectors[:, 1],
            length=1.0,
            normalize=False,
            colors=colors,
            alpha=0.78,
            linewidths=1.6,
            arrow_length_ratio=0.38,
        )
        self.main_ax.quiver(
            current_pos[0], current_pos[2], current_pos[1],
            *self._scale_local_wind(local_wind, grid_step),
            length=1.0,
            normalize=False,
            color="limegreen",
            alpha=0.95,
            linewidths=4.0,
            arrow_length_ratio=0.42,
        )
        self.main_ax.plot([], [], [], color="limegreen", lw=2.6,
                          label=f"Ветер у снаряда |W|={np.linalg.norm(local_wind):.1f} м/с")
        self.main_ax.plot([], [], [], color="white", alpha=0.0,
                          label=f"Сетка ветра: {len(wind_points)} стрелок, |W|max={max_speed:.1f} м/с")
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Боковой снос Z, м")
        self.main_ax.set_zlabel("Высота Y, м")
        self.main_ax.set_title(
            f"{self.animation_data.get('display_label', 'Итоговая модель')}\n"
            f"Запуск {run_index + 1}: t = {t_curr:.2f} с",
            pad=10,
        )
        self.main_ax.legend(loc="upper left", fontsize=7)

        self.anim_time_label.setText(f"t = {t_curr:.2f} c")
        if self.anim_slider.value() != frame_index:
            self.anim_slider.blockSignals(True)
            self.anim_slider.setValue(frame_index)
            self.anim_slider.blockSignals(False)

        self.main_canvas.draw_idle()

    def _set_dynamic_axes_limits(self, trajectories: list, focus: np.ndarray | None = None) -> tuple:
        """Ставит устойчивые границы 3D-осей для итоговой модели."""
        points = np.vstack(trajectories)
        x_min = min(0.0, float(np.min(points[:, 0])))
        x_max = max(float(np.max(points[:, 0])) * 1.08, 1.0)
        y_max = max(float(np.max(points[:, 1])) * 1.18, 1.0)
        z_abs = max(float(np.max(np.abs(points[:, 2]))) * 1.35, 30.0)

        xlim = (x_min, x_max)
        ylim = (-z_abs, z_abs)
        zlim = (0.0, y_max)

        if focus is not None:
            zoom = self.anim_zoom_slider.value() / 100.0
            if zoom > 0.0:
                cx = float(focus[0])
                cy = float(focus[1])
                cz = float(focus[2])

                full_x_half = max(0.5 * (xlim[1] - xlim[0]), 1.0)
                full_height_half = max(0.5 * (zlim[1] - zlim[0]), 1.0)
                full_side_half = max(0.5 * (ylim[1] - ylim[0]), 1.0)
                x_half = full_x_half * (1.0 - zoom) + 70.0 * zoom
                height_half = full_height_half * (1.0 - zoom) + 55.0 * zoom
                side_half = full_side_half * (1.0 - zoom) + 65.0 * zoom

                xlim = (cx - x_half, cx + x_half)
                ylim = (cz - side_half, cz + side_half)
                zlim = (cy - height_half, cy + height_half)

        self.main_ax.set_xlim(*xlim)
        self.main_ax.set_ylim(*ylim)
        self.main_ax.set_zlim(*zlim)
        return xlim, ylim, zlim

    def _draw_wind_layers(
        self,
        xlim: tuple,
        ylim: tuple,
        zlim: tuple,
        annotate: bool,
    ) -> None:
        """Показывает высотные слои ветрового поля полупрозрачными плоскостями."""
        if not self.animation_data:
            return

        layers = self.animation_data.get("wind_layers", [])
        if not layers:
            return

        x = np.array([xlim[0], xlim[1]])
        z = np.array([ylim[0], ylim[1]])
        xx, zz = np.meshgrid(x, z)

        for layer in layers:
            bottom = float(layer["bottom"])
            top = float(layer["top"])
            visible_top = min(top, float(zlim[1]))
            visible_bottom = max(bottom, float(zlim[0]))
            if visible_top <= zlim[0] or visible_bottom >= zlim[1]:
                continue

            color = layer["color"]
            for y_level in [visible_bottom, visible_top]:
                if zlim[0] <= y_level <= zlim[1]:
                    yy = np.full_like(xx, y_level)
                    self.main_ax.plot_surface(
                        xx, zz, yy,
                        color=color,
                        shade=False,
                        linewidth=0,
                        antialiased=False,
                    )

            if annotate:
                y_mid = 0.5 * (visible_bottom + visible_top)
                self.main_ax.text(
                    xlim[0],
                    ylim[1],
                    y_mid,
                    layer["name"],
                    fontsize=7,
                    color=(0.15, 0.15, 0.15),
                )

    def _build_wind_display_grid(self, xlim: tuple, ylim: tuple, zlim: tuple) -> tuple:
        """Строит регулярную сетку стрелок в центрах ячеек видимого куба."""
        target = int(self.anim_density_combo.currentText())
        shapes = {
            40: (4, 5, 2),
            80: (4, 5, 4),
            160: (5, 8, 4),
            256: (8, 8, 4),
        }
        nx, ny, nz = shapes.get(target, (5, 4, 4))
        x_vals = self._cell_centers(xlim[0], xlim[1], nx)
        z_vals = self._cell_centers(ylim[0], ylim[1], nz)
        y_vals = self._cell_centers(zlim[0], zlim[1], ny)
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        dx = (xlim[1] - xlim[0]) / max(nx, 1)
        dy = (zlim[1] - zlim[0]) / max(ny, 1)
        dz = (ylim[1] - ylim[0]) / max(nz, 1)
        grid_step = max(min(dx, dy, dz), 1.0)
        return points, grid_step

    @staticmethod
    def _cell_centers(low: float, high: float, count: int) -> np.ndarray:
        """Возвращает центры count равных отрезков на оси."""
        width = float(high) - float(low)
        if count <= 1:
            return np.array([float(low) + 0.5 * width])
        return float(low) + (np.arange(count, dtype=float) + 0.5) * width / count

    def _scale_wind_vectors(self, vectors: np.ndarray, grid_step: float) -> np.ndarray:
        """Масштабирует векторы ветра для видимых, но не одинаковых стрелок."""
        speed = np.linalg.norm(vectors, axis=1)
        direction = vectors / np.maximum(speed[:, None], 1e-9)
        visual_len = 0.82 * grid_step * (
            0.35 + 0.65 * np.sqrt(speed / max(float(np.max(speed)), 1e-9))
        )
        return direction * visual_len[:, None]

    def _scale_local_wind(self, vector: np.ndarray, grid_step: float) -> tuple:
        """Возвращает компоненты крупной стрелки локального ветра в порядке X,Z,Y."""
        speed = float(np.linalg.norm(vector))
        direction = vector / max(speed, 1e-9)
        scaled = direction * (1.55 * grid_step)
        return float(scaled[0]), float(scaled[2]), float(scaled[1])

    @staticmethod
    def _wind_colors(values: np.ndarray) -> list:
        """Цвета стрелок ветра: от холодного голубого к теплому оранжевому."""
        colors = []
        for value in np.clip(values, 0.0, 1.0):
            r = 0.10 + 0.90 * float(value)
            g = 0.55 + 0.25 * (1.0 - abs(float(value) - 0.55) / 0.55)
            b = 1.00 - 0.85 * float(value)
            colors.append((r, g, b, 0.82))
        return colors

    def _set_animation_replay_enabled(self, enabled: bool) -> None:
        """Включает управление записью только для выбранного пуска."""
        self.anim_zoom_slider.setEnabled(enabled)
        self.anim_density_combo.setEnabled(enabled)
        self.anim_prev_btn.setEnabled(enabled)
        self.anim_play_btn.setEnabled(enabled)
        self.anim_next_btn.setEnabled(enabled)
        self.anim_slider.setEnabled(enabled)

    def _disable_animation_controls(self) -> None:
        """Отключает панель записи для всех обычных моделей."""
        self.animation_timer.stop()
        self.animation_data = None
        self.animation_frame = 0
        self.anim_play_btn.setText("Пуск")
        self.anim_run_combo.blockSignals(True)
        self.anim_run_combo.clear()
        self.anim_run_combo.blockSignals(False)
        self._set_animation_replay_enabled(False)
        self.animation_panel.setVisible(False)

    def _toggle_animation(self) -> None:
        """Запуск/пауза воспроизведения записи."""
        if not self.animation_data:
            return
        if self.animation_timer.isActive():
            self.animation_timer.stop()
            self.anim_play_btn.setText("Пуск")
            return

        if self.animation_frame >= len(self.animation_data["frame_times"]) - 1:
            self._render_dynamic_wind_frame(0)
        self.animation_timer.start()
        self.anim_play_btn.setText("Пауза")

    def _advance_animation_frame(self) -> None:
        """Переход к следующему кадру по таймеру."""
        if not self.animation_data:
            return
        last = len(self.animation_data["frame_times"]) - 1
        if self.animation_frame >= last:
            self.animation_timer.stop()
            self.anim_play_btn.setText("Пуск")
            return
        self._render_dynamic_wind_frame(self.animation_frame + 1)

    def _previous_animation_frame(self) -> None:
        """Ручной шаг записи назад."""
        if not self.animation_data:
            return
        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")
        self._render_dynamic_wind_frame(self.animation_frame - 1)

    def _next_animation_frame(self) -> None:
        """Ручной шаг записи вперед."""
        if not self.animation_data:
            return
        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")
        self._render_dynamic_wind_frame(self.animation_frame + 1)

    def _on_animation_slider_changed(self, value: int) -> None:
        """Переход к кадру по ползунку."""
        if not self.animation_data:
            return
        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")
        self._render_dynamic_wind_frame(value)

    def _on_animation_run_changed(self, index: int) -> None:
        """Переключает итоговый вид и запись конкретного пуска."""
        if not self.animation_data:
            return

        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")
        self.animation_frame = 0
        self.anim_slider.blockSignals(True)
        self.anim_slider.setValue(0)
        self.anim_slider.blockSignals(False)

        if index <= 0:
            self._set_animation_replay_enabled(False)
            self._render_dynamic_wind_summary()
        else:
            self._set_animation_replay_enabled(True)
            self._render_dynamic_wind_frame(0)

    def _on_animation_zoom_changed(self, value: int) -> None:
        """Перерисовывает текущий кадр при смене масштаба записи."""
        self.anim_zoom_label.setText(f"Zoom {value}%")
        if not self.animation_data:
            return
        if self.anim_run_combo.currentIndex() <= 0:
            return
        self._render_dynamic_wind_frame(self.animation_frame)

    def _on_animation_density_changed(self, _index: int) -> None:
        """Перерисовывает текущий кадр при смене плотности стрелок."""
        if not self.animation_data:
            return
        if self.anim_run_combo.currentIndex() <= 0:
            return
        self._render_dynamic_wind_frame(self.animation_frame)

    def _current_3d_view(self) -> tuple:
        """Возвращает текущий ракурс 3D-оси, чтобы кадры не сбрасывали поворот."""
        elev = getattr(self.main_ax, "elev", 22)
        azim = getattr(self.main_ax, "azim", -58)
        return elev, azim

    def _on_canvas_mouse_press(self, event) -> None:
        """
        Даёт пользователю спокойно вращать 3D-график мышью.

        Если запись проигрывалась, на время drag ставим таймер на паузу.
        """
        if not self.animation_data or event.inaxes is not self.main_ax:
            return
        if self.anim_run_combo.currentIndex() <= 0:
            return

        self._resume_animation_after_drag = self.animation_timer.isActive()
        if self._resume_animation_after_drag:
            self.animation_timer.stop()

    def _on_canvas_mouse_release(self, event) -> None:
        """После вращения продолжает запись с новым сохранённым ракурсом."""
        if not self.animation_data:
            return
        if self.anim_run_combo.currentIndex() <= 0:
            return

        self._render_dynamic_wind_frame(self.animation_frame)
        if self._resume_animation_after_drag:
            self.animation_timer.start()
            self.anim_play_btn.setText("Пауза")
        self._resume_animation_after_drag = False


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrajectoryApp()
    window.show()
    sys.exit(app.exec())
