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
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt6.QtWidgets import QApplication, QLineEdit, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QTimer

from ui_main import MainWindowUI
from mc_renderer import (
    render_monte_carlo_6dof,
    compute_mean_trajectory,
    compute_mean_trajectory_2d,
    compute_impact_ellipse,
)


VORTEX_ONLY_PARAMS = {
    "Vortex_X_Shift",
    "Vortex_Z_Shift",
    "Vortex_Strength_Factor",
}


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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        "Vortex_X_Shift": "0.0",
        "Vortex_Z_Shift": "0.0",
        "Vortex_Strength_Factor": "1.0",
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
        self.param_labels: dict = {}
        self.animation_data: dict | None = None
        self.basic_export_data: dict | None = None
        self.model4_export_data: dict | None = None
        self.animation_frame: int = 0
        self._resume_animation_after_drag = False
        self._presentation_export_active = False
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(120)

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.scenario_preset_combo.currentIndexChanged.connect(self._on_scenario_preset_changed)
        self.calc_btn.clicked.connect(self.run_calculation)
        self.export_basic_graph_btn.clicked.connect(self._on_primary_export_clicked)
        self.export_compare_2d_btn.clicked.connect(self._on_secondary_export_clicked)
        self.export_diploma_graph_btn.clicked.connect(self._export_model4_diploma_png)
        self.animation_timer.timeout.connect(self._advance_animation_frame)
        self.anim_play_btn.clicked.connect(self._toggle_animation)
        self.anim_prev_btn.clicked.connect(self._previous_animation_frame)
        self.anim_next_btn.clicked.connect(self._next_animation_frame)
        self.anim_export_gif_btn.clicked.connect(self._export_animation_gif)
        self.anim_export_png_btn.clicked.connect(self._export_current_graph_png)
        self.anim_export_presentation_btn.clicked.connect(self._export_presentation_graph_png)
        self.anim_slider.valueChanged.connect(self._on_animation_slider_changed)
        self.anim_run_combo.currentIndexChanged.connect(self._on_animation_run_changed)
        self.anim_zoom_slider.valueChanged.connect(self._on_animation_zoom_changed)
        self.anim_density_combo.currentIndexChanged.connect(self._on_animation_density_changed)
        self.anim_camera_combo.currentIndexChanged.connect(self._on_animation_view_changed)
        self.anim_region_combo.currentIndexChanged.connect(self._on_animation_view_changed)
        self.anim_view_combo.currentIndexChanged.connect(self._on_animation_view_changed)
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
        self._disable_basic_export()
        self._configure_static_export_buttons(model_name)

        info = self.models[model_name].get_info()

        # 1. Текстовое описание + расшифровка переменных
        vars_lines = "\n".join(
            f"• {k} — {v}"
            for k, v in info.get("parameters_info", {}).items()
        )
        self.info_display.setText(
            f"{info.get('description', '')}\n\nОбозначения:\n{vars_lines}"
        )
        self._fit_info_display_to_content()

        # 2. Рендеринг LaTeX через Agg → PNG → QPixmap.
        #    ВАЖНО: каждый раз создаём НОВУЮ временную Figure и новый
        #    FigureCanvasAgg. Никогда не привязываем Agg к self.formula_fig,
        #    которая уже принадлежит Qt-layout — это вызывает 0xC0000409.
        formula_text = info.get("formula", "")
        try:
            from matplotlib.figure import Figure as _Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as _Agg

            line_count = max(1, formula_text.count("\n") + 1)
            fig_height = 2.15 if line_count <= 2 else min(3.0, 1.25 + 0.45 * line_count)
            font_size = 16 if line_count <= 2 else 13
            tmp_fig = _Figure(figsize=(4.5, fig_height), facecolor='#f0f0f0')
            tmp_canvas = _Agg(tmp_fig)
            tmp_ax = tmp_fig.add_subplot(111)
            tmp_ax.axis('off')
            tmp_ax.text(
                0.5, 0.5,
                formula_text,
                fontsize=font_size,
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
            self.param_labels[p_name] = self.params_layout.labelForField(edit)
            if p_name == "scenario":
                edit.textChanged.connect(self._update_vortex_param_visibility)

        self._configure_scenario_presets(model_name)
        self._update_vortex_param_visibility()

    def _fit_info_display_to_content(self) -> None:
        """Подбирает высоту описания под раскрытый текст с разумным верхним пределом."""
        doc = self.info_display.document()
        doc.setTextWidth(max(self.info_display.viewport().width(), 260))
        height = int(doc.size().height()) + 24
        self.info_display.setMinimumHeight(max(190, min(height, 520)))

    def _clear_params(self) -> None:
        """Удаляет все виджеты параметров из формы."""
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.param_widgets.clear()
        self.param_labels.clear()

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
        self._update_vortex_param_visibility()

    def _update_vortex_param_visibility(self) -> None:
        """Показывает параметры управления вихрем только для scenario=vortex."""
        scenario_widget = self.param_widgets.get("scenario")
        scenario = scenario_widget.text().strip().lower() if scenario_widget else ""
        show_vortex_params = scenario in {"vortex", "вихрь"}

        for name in VORTEX_ONLY_PARAMS:
            widget = self.param_widgets.get(name)
            label = self.param_labels.get(name)
            if widget is not None:
                widget.setVisible(show_vortex_params)
            if label is not None:
                label.setVisible(show_vortex_params)

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
        self._disable_basic_export()

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
            elif model_name.startswith("4."):
                self._render_monte_carlo_3d_model(result, label, params)
                self._prepare_model4_export(model_name, label, result, params)
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
                if model_name.startswith(("1.", "2.")):
                    self._prepare_basic_export(model_name, label, result, params)

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

    def _disable_basic_export(self) -> None:
        """Отключает экспорт презентационного 2D-графика до нового расчёта."""
        self.basic_export_data = None
        self.model4_export_data = None
        self.export_basic_graph_btn.setEnabled(False)
        self.export_compare_2d_btn.setEnabled(False)
        self.export_diploma_graph_btn.setEnabled(False)

    def _configure_static_export_buttons(self, model_name: str) -> None:
        """Показывает статические кнопки экспорта только у моделей, где они полезны."""
        is_model1 = model_name.startswith("1.")
        is_model2 = model_name.startswith("2.")
        is_model4 = model_name.startswith("4.")

        self.export_basic_graph_btn.setVisible(is_model1 or is_model2 or is_model4)
        self.export_compare_2d_btn.setVisible(is_model2 or is_model4)
        self.export_diploma_graph_btn.setVisible(is_model4)
        self.export_basic_graph_btn.setText("ЭКСПОРТ ГРАФИКА")
        self.export_compare_2d_btn.setText("GIF" if is_model4 else "СРАВНИТЬ С ВАКУУМОМ")

    def _on_primary_export_clicked(self) -> None:
        """Первая статическая кнопка экспорта: 2D-график или PNG модели 4."""
        model_name = self.model_combo.currentText()
        if model_name.startswith("4."):
            self._export_model4_png()
        else:
            self._export_basic_2d_graph()

    def _on_secondary_export_clicked(self) -> None:
        """Вторая статическая кнопка: сравнение модели 2 или GIF модели 4."""
        model_name = self.model_combo.currentText()
        if model_name.startswith("4."):
            self._export_model4_orbit_gif()
        elif model_name.startswith("2."):
            self._export_model2_comparison()

    def _prepare_basic_export(
        self,
        model_name: str,
        label: str,
        result: tuple,
        params: dict,
    ) -> None:
        """Сохраняет данные последнего расчёта модели 1/2 для экспорта."""
        x_arr, y_arr = result
        self.basic_export_data = {
            "model_name": model_name,
            "label": label,
            "x": np.asarray(x_arr, dtype=float),
            "y": np.asarray(y_arr, dtype=float),
            "params": dict(params),
        }
        self.export_basic_graph_btn.setEnabled(True)
        self.export_compare_2d_btn.setEnabled(model_name.startswith("2."))

    def _prepare_model4_export(
        self,
        model_name: str,
        label: str,
        trajectories: list,
        params: dict,
    ) -> None:
        """Сохраняет данные модели 4 для PNG/GIF экспорта."""
        self.model4_export_data = {
            "model_name": model_name,
            "label": label,
            "trajectories": [np.asarray(t, dtype=float) for t in trajectories],
            "params": dict(params),
        }
        self.export_basic_graph_btn.setEnabled(True)
        self.export_compare_2d_btn.setEnabled(True)
        self.export_diploma_graph_btn.setEnabled(True)

    def _export_basic_2d_graph(self) -> None:
        """Экспортирует модели 1/2 как готовый презентационный PNG с метриками."""
        if not self.basic_export_data:
            QMessageBox.information(
                self,
                "Экспорт графика",
                "Сначала выполните расчёт модели 1 или 2.",
            )
            return

        try:
            from matplotlib.figure import Figure
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Экспорт графика",
                f"Не удалось подготовить Matplotlib Figure:\n{exc}",
            )
            return

        data = self.basic_export_data
        model_name = data["model_name"]
        label = data["label"]
        x_arr = data["x"]
        y_arr = data["y"]
        params = data["params"]
        metrics = self._basic_2d_metrics(model_name, x_arr, y_arr, params)
        param_lines = self._basic_2d_param_lines(model_name, params)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        safe_model = "model1" if model_name.startswith("1.") else "model2"
        export_path = export_dir / f"{safe_model}_summary_{timestamp}.png"

        fig = Figure(figsize=(12.5, 6.8), facecolor="white", constrained_layout=True)
        export_canvas = FigureCanvasAgg(fig)
        grid = fig.add_gridspec(1, 2, width_ratios=[2.45, 0.85])
        ax = fig.add_subplot(grid[0, 0])
        info_ax = fig.add_subplot(grid[0, 1])
        info_ax.axis("off")

        is_model1 = model_name.startswith("1.")
        color = "royalblue" if is_model1 else "firebrick"
        plot_label = "Идеальная траектория" if is_model1 else "Траектория в атмосфере"
        export_title = "Модель идеального полёта" if is_model1 else "Модель движения в атмосфере"
        ax.plot(x_arr, y_arr, lw=3.0, color=color, label=plot_label)
        ax.scatter([x_arr[-1]], [y_arr[-1]], color=color, s=55, zorder=4)
        ax.set_title(export_title, pad=10, fontsize=15)
        ax.set_xlabel("Дальность X, м")
        ax.set_ylabel("Высота Y, м")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")
        ax.set_xlim(left=0.0, right=max(float(np.max(x_arr)) * 1.06, 1.0))
        ax.set_ylim(bottom=0.0, top=max(float(np.max(y_arr)) * 1.16, 1.0))

        info_ax.text(
            0.0, 0.98,
            "Итоги расчёта",
            fontsize=15,
            fontweight="bold",
            va="top",
            transform=info_ax.transAxes,
        )
        y = 0.87
        for title, value in metrics:
            info_ax.text(0.0, y, title, fontsize=10.5, color="#555555", transform=info_ax.transAxes)
            info_ax.text(0.64, y, value, fontsize=12, fontweight="bold", transform=info_ax.transAxes)
            y -= 0.078

        y -= 0.035
        info_ax.text(
            0.0, y,
            "Основные параметры",
            fontsize=15,
            fontweight="bold",
            va="top",
            transform=info_ax.transAxes,
        )
        y -= 0.085
        for line in param_lines:
            info_ax.text(0.0, y, line, fontsize=11.0, transform=info_ax.transAxes)
            y -= 0.064

        try:
            export_canvas.print_figure(export_path, dpi=220, facecolor="white")
            QMessageBox.information(
                self,
                "Экспорт графика",
                f"График сохранён:\n{export_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Экспорт графика",
                f"Не удалось сохранить график:\n{exc}",
            )

    def _export_model2_comparison(self) -> None:
        """Экспортирует сравнение модели 2 с идеальной моделью при тех же v0/angle/g."""
        if not self.basic_export_data or not self.basic_export_data["model_name"].startswith("2."):
            QMessageBox.information(
                self,
                "Сравнение моделей",
                "Сначала выполните расчёт модели 2.",
            )
            return

        try:
            from matplotlib.figure import Figure
            from matplotlib.patches import Rectangle
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Сравнение моделей",
                f"Не удалось подготовить Matplotlib Figure:\n{exc}",
            )
            return

        data = self.basic_export_data
        params = data["params"]
        x_atm = data["x"]
        y_atm = data["y"]
        x_vac, y_vac = self._ideal_curve_from_params(params)

        metrics_vac = self._basic_2d_metrics("1. Идеальный полёт", x_vac, y_vac, params)
        metrics_atm = self._basic_2d_metrics("2. Атмосфера", x_atm, y_atm, params)

        range_vac = float(x_vac[-1])
        range_atm = float(x_atm[-1])
        height_vac = float(np.max(y_vac))
        height_atm = float(np.max(y_atm))
        range_loss = range_vac - range_atm
        range_loss_pct = 100.0 * range_loss / max(range_vac, 1e-9)
        height_delta = height_atm - height_vac
        height_loss = height_vac - height_atm
        height_loss_pct = 100.0 * height_loss / max(height_vac, 1e-9)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"model2_vs_vacuum_{timestamp}.png"

        fig = Figure(figsize=(13.4, 7.2), facecolor="white", constrained_layout=True)
        export_canvas = FigureCanvasAgg(fig)
        grid = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.05])
        ax = fig.add_subplot(grid[0, 0])
        info_ax = fig.add_subplot(grid[0, 1])
        info_ax.axis("off")

        ax.plot(x_vac, y_vac, lw=3.0, color="royalblue", label="Идеальный полёт")
        ax.plot(x_atm, y_atm, lw=3.0, color="firebrick", label="Полёт в атмосфере")
        ax.scatter([x_vac[-1]], [0.0], color="royalblue", s=55, zorder=4)
        ax.scatter([x_atm[-1]], [0.0], color="firebrick", s=55, zorder=4)
        ax.set_title("Сравнение полёта в вакууме и атмосфере", pad=10, fontsize=14)
        ax.set_xlabel("Дальность X, м")
        ax.set_ylabel("Высота Y, м")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")
        ax.set_xlim(left=0.0, right=max(float(np.max(x_vac)), float(np.max(x_atm))) * 1.06)
        ax.set_ylim(bottom=0.0, top=max(float(np.max(y_vac)), float(np.max(y_atm))) * 1.16)

        info_ax.text(
            0.0, 0.98,
            "Итоги",
            fontsize=16,
            fontweight="bold",
            va="top",
            transform=info_ax.transAxes,
        )
        table_left, table_right = 0.0, 0.98
        table_top = 0.89
        row_h = 0.058
        col_x = [0.02, 0.47, 0.73]
        table_rows = [
            ("Показатель", "Вакуум", "Атмосфера"),
            ("Дальность", metrics_vac[0][1], metrics_atm[0][1]),
            ("Макс. высота", metrics_vac[1][1], metrics_atm[1][1]),
            ("Время", metrics_vac[2][1], metrics_atm[2][1]),
            ("Потеря дальности", "—", f"{range_loss:.1f} м"),
            ("Потеря высоты", "—", f"{height_loss:.1f} м"),
        ]
        for i, row in enumerate(table_rows):
            y0 = table_top - i * row_h
            bg = "#f1f3f6" if i == 0 else ("#ffffff" if i % 2 else "#f8f9fb")
            info_ax.add_patch(
                Rectangle(
                    (table_left, y0 - row_h + 0.006),
                    table_right - table_left,
                    row_h,
                    transform=info_ax.transAxes,
                    facecolor=bg,
                    edgecolor="#d3d7de",
                    linewidth=0.6,
                )
            )
            for j, text in enumerate(row):
                info_ax.text(
                    col_x[j],
                    y0 - 0.036,
                    text,
                    fontsize=9.0 if i == 0 else 9.4,
                    fontweight="bold" if (i == 0 or j > 0) else "normal",
                    color="#222222" if i == 0 else ("#111111" if j > 0 else "#555555"),
                    transform=info_ax.transAxes,
                )

        info_ax.text(
            0.02, 0.515,
            f"Потеря дальности: {range_loss_pct:.1f}%",
            fontsize=10.2,
            fontweight="bold",
            color="#333333",
            transform=info_ax.transAxes,
        )
        info_ax.text(
            0.02, 0.475,
            f"Потеря высоты: {height_loss_pct:.1f}%",
            fontsize=10.2,
            fontweight="bold",
            color="#333333",
            transform=info_ax.transAxes,
        )

        info_ax.text(
            0.0, 0.43,
            "Общие условия",
            fontsize=14,
            fontweight="bold",
            va="top",
            transform=info_ax.transAxes,
        )
        common_lines = [
            f"v₀: {params.get('v0', '—')} м/с",
            f"Угол запуска: {params.get('angle', '—')}°",
        ]
        y = 0.36
        for line in common_lines:
            info_ax.text(0.0, y, line, fontsize=11.4, transform=info_ax.transAxes)
            y -= 0.058

        info_ax.text(
            0.0, 0.22,
            "Параметры модели 2",
            fontsize=14,
            fontweight="bold",
            va="top",
            transform=info_ax.transAxes,
        )
        model2_lines = [
            f"Масса: {params.get('m', '—')} кг",
            f"Площадь: {params.get('S', '—')} м²",
            f"Cx: {params.get('Cx', '—')}, Cy: {params.get('Cy', '—')}",
            f"ρ₀: {params.get('rho0', '—')} кг/м³",
        ]
        y = 0.155
        for line in model2_lines:
            info_ax.text(0.0, y, line, fontsize=10.2, color="#333333", transform=info_ax.transAxes)
            y -= 0.039

        try:
            export_canvas.print_figure(export_path, dpi=220, facecolor="white")
            QMessageBox.information(
                self,
                "Сравнение моделей",
                f"Сравнительный график сохранён:\n{export_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Сравнение моделей",
                f"Не удалось сохранить график:\n{exc}",
            )

    def _export_model4_png(self) -> None:
        """Экспортирует модель 4 в PNG с выбранным ракурсом камеры."""
        if not self.model4_export_data:
            QMessageBox.information(
                self,
                "Экспорт графика",
                "Сначала выполните расчёт модели 4.",
            )
            return

        camera_choice = QMessageBox.question(
            self,
            "Экспорт графика",
            (
                "Какой ракурс камеры использовать?\n\n"
                "Да — оптимальный презентационный ракурс.\n"
                "Нет — текущий ракурс, который вы выставили мышью."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if camera_choice == QMessageBox.StandardButton.Yes:
            elev, azim = 22, -58
        else:
            elev, azim = self._current_3d_view()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"model4_monte_carlo_3d_{timestamp}.png"

        try:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8.4, 7.2), facecolor="white", constrained_layout=True)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, projection="3d")
            self._render_model4_scene(
                ax,
                self.model4_export_data["trajectories"],
                self.model4_export_data["label"],
                self.model4_export_data["params"],
                add_panel=True,
                elev=elev,
                azim=azim,
            )
            canvas.print_figure(
                export_path,
                dpi=210,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.25,
            )
            QMessageBox.information(
                self,
                "Экспорт графика",
                f"График сохранён:\n{export_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Экспорт графика",
                f"Не удалось сохранить график:\n{exc}",
            )

    def _export_model4_diploma_png(self) -> None:
        """Экспортирует модель 4 в более лаконичном виде для вставки в диплом."""
        if not self.model4_export_data:
            QMessageBox.information(
                self,
                "График для диплома",
                "Сначала выполните расчёт модели 4.",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"model4_diploma_{timestamp}.png"

        try:
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8.8, 7.2), facecolor="white", constrained_layout=True)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111, projection="3d")
            self._render_model4_diploma_scene(
                ax,
                self.model4_export_data["trajectories"],
                elev=22,
                azim=-58,
            )
            canvas.print_figure(
                export_path,
                dpi=230,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.18,
            )
            QMessageBox.information(
                self,
                "График для диплома",
                f"График сохранён:\n{export_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "График для диплома",
                f"Не удалось сохранить график:\n{exc}",
            )

    def _export_model4_orbit_gif(self) -> None:
        """Экспортирует модель 4 как GIF полного оборота вокруг 3D-сцены."""
        if not self.model4_export_data:
            QMessageBox.information(
                self,
                "Экспорт GIF",
                "Сначала выполните расчёт модели 4.",
            )
            return

        try:
            from PIL import Image
            from matplotlib.figure import Figure
        except ImportError as exc:
            QMessageBox.warning(
                self,
                "Экспорт GIF",
                f"Для экспорта GIF нужен Pillow и Matplotlib:\n{exc}",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"model4_monte_carlo_3d_orbit_{timestamp}.gif"

        old_text = self.export_compare_2d_btn.text()
        frames = []
        frame_count = 72
        elev = 22
        azimuths = np.linspace(-58, 302, frame_count, endpoint=False)
        try:
            self.export_compare_2d_btn.setEnabled(False)
            for i, azim in enumerate(azimuths, start=1):
                fig = Figure(figsize=(8.4, 7.2), facecolor="white", constrained_layout=True)
                canvas = FigureCanvasAgg(fig)
                ax = fig.add_subplot(111, projection="3d")
                self._render_model4_scene(
                    ax,
                    self.model4_export_data["trajectories"],
                    self.model4_export_data["label"],
                    self.model4_export_data["params"],
                    add_panel=True,
                    elev=elev,
                    azim=float(azim),
                )
                buf = io.BytesIO()
                canvas.print_figure(
                    buf,
                    format="png",
                    dpi=125,
                    facecolor="white",
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
                buf.seek(0)
                palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE")
                frames.append(Image.open(buf).convert("P", palette=palette).copy())
                buf.close()
                self.export_compare_2d_btn.setText(f"GIF {i}/{frame_count}")
                QApplication.processEvents()

            if frames:
                frames[0].save(
                    export_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=190,
                    loop=0,
                    optimize=True,
                )
                QMessageBox.information(
                    self,
                    "Экспорт GIF",
                    f"GIF сохранён:\n{export_path}",
                )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Экспорт GIF",
                f"Не удалось создать GIF:\n{exc}",
            )
        finally:
            self.export_compare_2d_btn.setText(old_text)
            self.export_compare_2d_btn.setEnabled(True)

    @staticmethod
    def _ideal_curve_from_params(params: dict) -> tuple:
        """Строит идеальную траекторию по параметрам модели 2."""
        v0 = float(params.get("v0", 100.0))
        angle = np.radians(float(params.get("angle", 45.0)))
        gravity = float(params.get("g", 9.81))
        t_max = 2.0 * v0 * np.sin(angle) / max(gravity, 1e-9)
        t = np.linspace(0.0, t_max, 250)
        x = v0 * np.cos(angle) * t
        y = v0 * np.sin(angle) * t - 0.5 * gravity * t ** 2
        return x, np.maximum(y, 0.0)

    def _basic_2d_metrics(
        self,
        model_name: str,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        params: dict,
    ) -> list:
        """Считает ключевые итоги для презентационного экспорта моделей 1/2."""
        range_m = float(x_arr[-1])
        max_height_m = float(np.max(y_arr))
        if model_name.startswith("1."):
            v0 = float(params.get("v0", 0.0))
            angle = np.radians(float(params.get("angle", 0.0)))
            gravity = float(params.get("g", 9.81))
            flight_time = 2.0 * v0 * np.sin(angle) / max(gravity, 1e-9)
        else:
            flight_time = max(len(x_arr) - 1, 0) * 0.005

        return [
            ("Дальность", f"{range_m:.1f} м"),
            ("Макс. высота", f"{max_height_m:.1f} м"),
            ("Время полёта", f"{flight_time:.2f} с"),
            ("Точек расчёта", f"{len(x_arr)}"),
        ]

    @staticmethod
    def _basic_2d_param_lines(model_name: str, params: dict) -> list:
        """Формирует короткий список параметров, общий для сравнения моделей."""
        lines = [
            f"v₀: {params.get('v0', '—')} м/с",
            f"Угол запуска: {params.get('angle', '—')}°",
            f"g: {params.get('g', '9.81')} м/с²",
        ]
        if model_name.startswith("2."):
            lines.extend([
                f"Масса: {params.get('m', '—')} кг",
                f"Площадь: {params.get('S', '—')} м²",
                f"Cx: {params.get('Cx', '—')}, Cy: {params.get('Cy', '—')}",
                f"ρ₀: {params.get('rho0', '—')} кг/м³",
            ])
        return lines

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

    def _render_monte_carlo_3d_model(
        self,
        trajectories: list,
        label: str,
        params: dict,
    ) -> None:
        """Рисует модель 4 в презентационном виде."""
        self._render_model4_scene(self.main_ax, trajectories, label, params, add_panel=True)

    def _render_model4_scene(
        self,
        ax,
        trajectories: list,
        label: str,
        params: dict,
        add_panel: bool,
        elev: float = 22,
        azim: float = -58,
    ) -> None:
        """Общий рендер модели 4 для экрана, PNG и GIF."""
        stats = self._model4_stats(trajectories)
        ax.view_init(elev=elev, azim=azim)

        for traj in trajectories:
            ax.plot(
                traj[:, 0], traj[:, 2], traj[:, 1],
                color="dodgerblue", alpha=0.16, lw=0.8,
            )
        ax.plot([], [], [], color="dodgerblue", alpha=0.45, lw=1.2,
                label=f"Пуски МК (N={len(trajectories)})")

        impacts = np.array([traj[-1] for traj in trajectories])
        ax.scatter(
            impacts[:, 0], impacts[:, 2], np.zeros(len(impacts)),
            color="steelblue", s=16, alpha=0.60, label="Точки падения",
        )

        for n_sig, color, ls, lw, pct in [
            (1, "limegreen", "-", 1.8, "68%"),
            (2, "darkorange", "--", 2.0, "95%"),
            (3, "crimson", ":", 2.2, "99.7%"),
        ]:
            ex, ez, *_ = compute_impact_ellipse(trajectories, n_sig)
            ax.plot(
                ex, ez, np.zeros_like(ex),
                color=color, lw=lw, ls=ls,
                label=f"Эллипс {n_sig}σ ({pct})",
            )

        mean = compute_mean_trajectory(trajectories)
        ax.plot(
            mean[:, 0], mean[:, 2], mean[:, 1],
            color="red", lw=2.6, label="Средняя траектория",
        )
        ax.scatter(
            [stats["center_x"]], [stats["center_z"]], [0.0],
            color="red", s=88, marker="X",
            label=f"Мишень σx={stats['std_x']:.1f}м σz={stats['std_z']:.1f}м",
        )

        all_points = np.vstack(trajectories)
        x_max = max(float(np.max(all_points[:, 0])) * 1.08, 1.0)
        y_max = max(float(np.max(all_points[:, 1])) * 1.18, 1.0)
        z_abs = max(float(np.max(np.abs(all_points[:, 2]))) * 1.35, 20.0)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(-z_abs, z_abs)
        ax.set_zlim(0.0, y_max)

        ax.set_xlabel("Дальность X, м")
        ax.set_ylabel("Боковой снос Z, м")
        ax.set_zlabel("Высота Y, м")
        ax.set_title(f"Результат: {label}", pad=10)
        ax.legend(loc="upper left", fontsize=7)
        if add_panel:
            self._draw_model4_stats_panel(ax, stats, params)

    def _render_model4_diploma_scene(
        self,
        ax,
        trajectories: list,
        elev: float = 22,
        azim: float = -58,
    ) -> None:
        """Рисует модель 4 в лаконичном виде для дипломной иллюстрации."""
        stats = self._model4_stats(trajectories)
        ax.view_init(elev=elev, azim=azim)

        for traj in trajectories:
            ax.plot(
                traj[:, 0], traj[:, 2], traj[:, 1],
                color="lightskyblue", alpha=0.22, lw=0.8,
            )
        ax.plot(
            [], [], [],
            color="lightskyblue", alpha=0.70, lw=1.4,
            label=f"Траектории запусков (N={len(trajectories)})",
        )

        impacts = np.array([traj[-1] for traj in trajectories])
        ax.scatter(
            impacts[:, 0], impacts[:, 2], np.zeros(len(impacts)),
            color="steelblue", s=18, alpha=0.65, label="Точки падения",
        )

        for n_sig, color, ls, lw, label in [
            (1, "limegreen", "-", 1.8, "Эллипс 1σ"),
            (2, "darkorange", "--", 2.0, "Эллипс 2σ"),
            (3, "crimson", ":", 2.2, "Эллипс 3σ"),
        ]:
            ex, ez, *_ = compute_impact_ellipse(trajectories, n_sig)
            ax.plot(ex, ez, np.zeros_like(ex), color=color, lw=lw, ls=ls, label=label)

        mean = compute_mean_trajectory(trajectories)
        ax.plot(
            mean[:, 0], mean[:, 2], mean[:, 1],
            color="red", lw=2.8, label="Средняя траектория",
        )
        ax.scatter(
            [stats["center_x"]], [stats["center_z"]], [0.0],
            color="red", s=92, marker="X", label="Центр рассеивания",
        )

        all_points = np.vstack(trajectories)
        x_max = max(float(np.max(all_points[:, 0])) * 1.08, 1.0)
        y_max = max(float(np.max(all_points[:, 1])) * 1.18, 1.0)
        z_abs = max(float(np.max(np.abs(all_points[:, 2]))) * 1.35, 20.0)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(-z_abs, z_abs)
        ax.set_zlim(0.0, y_max)

        ax.set_xlabel("Дальность X, м")
        ax.set_ylabel("Боковой снос Z, м")
        ax.set_zlabel("Высота Y, м")
        ax.set_title("Трёхмерное моделирование методом Монте-Карло", pad=10)
        ax.legend(loc="upper left", fontsize=7.5)

        panel = (
            "Итоги расчёта\n"
            f"N = {stats['n']}\n"
            f"Средняя дальность: {stats['mean_range']:.1f} м\n"
            f"Средняя высота: {stats['mean_height']:.1f} м\n"
            f"Разброс: σx={stats['std_x']:.1f} м, σz={stats['std_z']:.1f} м"
        )
        ax.text2D(
            0.985, 0.965,
            panel,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.6,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": (1.0, 1.0, 1.0, 0.90),
                "edgecolor": (0.70, 0.70, 0.70, 0.75),
                "linewidth": 0.8,
            },
        )

    @staticmethod
    def _model4_stats(trajectories: list) -> dict:
        """Считает статистику для модели 4."""
        impacts = np.array([traj[-1] for traj in trajectories])
        max_heights = np.array([float(np.max(traj[:, 1])) for traj in trajectories])
        times = np.array([max(len(traj) - 1, 0) * 0.05 for traj in trajectories])
        mean = compute_mean_trajectory(trajectories)
        _, _, center_x, center_z, std_x, std_z = compute_impact_ellipse(trajectories, 1)
        return {
            "n": len(trajectories),
            "center_x": center_x,
            "center_z": center_z,
            "std_x": std_x,
            "std_z": std_z,
            "mean_range": float(np.mean(impacts[:, 0])),
            "mean_side": float(np.mean(impacts[:, 2])),
            "mean_height": float(np.mean(max_heights)),
            "mean_time": float(np.mean(times)),
            "mean_traj_range": float(mean[-1, 0]),
        }

    @staticmethod
    def _draw_model4_stats_panel(ax, stats: dict, params: dict) -> None:
        """Показывает краткую статистику модели 4 поверх 3D-графика."""
        panel = (
            "3D МОНТЕ-КАРЛО\n"
            f"N: {stats['n']}\n"
            f"Средняя дальность: {stats['mean_range']:.1f} м\n"
            f"Средняя высота max: {stats['mean_height']:.1f} м\n"
            f"Среднее время: {stats['mean_time']:.2f} с\n"
            f"Мишень: X={stats['center_x']:.1f} м, Z={stats['center_z']:.1f} м\n"
            f"Разброс: σx={stats['std_x']:.1f} м, σz={stats['std_z']:.1f} м\n"
            f"Ветер: Wx~{params.get('wx_mean', '—')} м/с, "
            f"Wz~{params.get('wz_mean', '—')}±{params.get('wind_std', '—')} м/с"
        )
        ax.text2D(
            0.985, 0.985,
            panel,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": (1.0, 1.0, 1.0, 0.88),
                "edgecolor": (0.65, 0.65, 0.65, 0.70),
                "linewidth": 0.8,
            },
        )

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
        self.anim_camera_combo.setCurrentText("Следить за снарядом")
        self.anim_region_combo.setCurrentText("Куб средний")
        self.anim_view_combo.setCurrentText("3D поле")

        self.animation_panel.setVisible(True)
        self.anim_export_gif_btn.setEnabled(True)
        self.anim_export_png_btn.setEnabled(True)
        self.anim_export_presentation_btn.setEnabled(True)
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

        ellipse_limit_points = []
        for n_sig, color, ls, lw, pct in [
            (1, "limegreen", "-", 1.8, "68%"),
            (2, "darkorange", "--", 2.0, "95%"),
            (3, "crimson", ":", 2.2, "99.7%"),
        ]:
            ex, ez, *_ = compute_impact_ellipse(trajectories, n_sig)
            ellipse_limit_points.append(
                np.column_stack([ex, np.zeros_like(ex), ez])
            )
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

        summary_limit_points = np.vstack([
            impacts,
            np.array([[cx, 0.0, cz]], dtype=float),
            *ellipse_limit_points,
        ])
        xlim, ylim, zlim = self._set_dynamic_axes_limits(
            trajectories,
            extra_points=summary_limit_points,
        )
        self._draw_wind_layers(xlim, ylim, zlim, annotate=True)
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Боковой снос Z, м")
        self.main_ax.set_zlabel("Высота Y, м")
        self.main_ax.set_title(f"{label}\nИтоговый результат по всем пускам", pad=10)
        self.main_ax.legend(loc="upper left", fontsize=7)
        self._draw_dynamic_summary_panel(trajectories)

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
        view_mode = self.anim_view_combo.currentText()
        wind_xylim, wind_ylim, wind_zlim = self._wind_region_limits(xlim, ylim, zlim, current_pos)
        if view_mode == "Вертикальный срез":
            self._draw_vertical_slice_plane(wind_xylim, wind_zlim, float(current_pos[2]))
            wind_points, grid_step = self._build_wind_slice_grid(
                wind_xylim, wind_zlim, float(current_pos[2])
            )
        else:
            wind_points, grid_step = self._build_wind_display_grid(
                wind_xylim, wind_ylim, wind_zlim
            )
        wind_vectors = wind_field.sample_points(t_curr, wind_points)

        speed = np.linalg.norm(wind_vectors, axis=1)
        max_speed = max(float(np.max(speed)), 1e-9)
        colors = self._wind_colors(speed)
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
        local_arrow = self._scale_local_wind(local_wind, grid_step)
        local_end = (
            current_pos[0] + local_arrow[0],
            current_pos[2] + local_arrow[1],
            current_pos[1] + local_arrow[2],
        )
        self.main_ax.quiver(
            current_pos[0], current_pos[2], current_pos[1],
            *local_arrow,
            length=1.0,
            normalize=False,
            color="black",
            alpha=1.0,
            linewidths=7.0,
            arrow_length_ratio=0.46,
            zorder=102,
        )
        self.main_ax.quiver(
            current_pos[0], current_pos[2], current_pos[1],
            *local_arrow,
            length=1.0,
            normalize=False,
            color="limegreen",
            alpha=1.0,
            linewidths=4.2,
            arrow_length_ratio=0.46,
            zorder=103,
        )
        self.main_ax.scatter(
            [local_end[0]], [local_end[1]], [local_end[2]],
            color="black", s=62, marker=">", alpha=1.0, zorder=104,
        )
        self.main_ax.scatter(
            [local_end[0]], [local_end[1]], [local_end[2]],
            color="limegreen", s=38, marker=">", alpha=1.0, zorder=105,
        )
        self.main_ax.text(
            local_end[0], local_end[1], local_end[2],
            " W",
            color="black",
            fontsize=10,
            weight="bold",
            zorder=106,
        )
        self.main_ax.plot([], [], [], color="limegreen", lw=2.6,
                          label="Локальный ветер у снаряда")
        self.main_ax.plot([], [], [], color="white", alpha=0.0,
                          label=f"Ветровая сетка: {len(wind_points)} стрелок, |W|max={max_speed:.1f} м/с")
        self.main_ax.plot([], [], [], color="white", alpha=0.0,
                          label="Цвет: синий <5, зелёный 5-15, оранжевый 15-30, красный >30 м/с")
        self.main_ax.set_xlabel("Дальность X, м")
        self.main_ax.set_ylabel("Боковой снос Z, м")
        self.main_ax.set_zlabel("Высота Y, м")
        self.main_ax.set_title(
            f"{self.animation_data.get('display_label', 'Итоговая модель')}\n"
            f"Запуск {run_index + 1}: t = {t_curr:.2f} с",
            pad=10,
        )
        self.main_ax.legend(loc="upper left", fontsize=7)
        self._draw_frame_mini_panel(
            current_pos,
            local_wind,
            t_curr,
            view_mode,
            self.anim_camera_combo.currentText(),
            self.anim_region_combo.currentText(),
        )

        self.anim_time_label.setText(f"t = {t_curr:.2f} c")
        if self.anim_slider.value() != frame_index:
            self.anim_slider.blockSignals(True)
            self.anim_slider.setValue(frame_index)
            self.anim_slider.blockSignals(False)

        self.main_canvas.draw_idle()

    def _set_dynamic_axes_limits(
        self,
        trajectories: list,
        focus: np.ndarray | None = None,
        extra_points: np.ndarray | None = None,
    ) -> tuple:
        """Ставит устойчивые границы 3D-осей для итоговой модели."""
        points = np.vstack(trajectories)
        if extra_points is not None and len(extra_points) > 0:
            points = np.vstack([points, extra_points])
        x_min = min(0.0, float(np.min(points[:, 0])))
        x_max = max(float(np.max(points[:, 0])) * 1.08, 1.0)
        y_max = max(float(np.max(points[:, 1])) * 1.18, 1.0)
        z_abs = max(float(np.max(np.abs(points[:, 2]))) * 1.35, 30.0)

        xlim = (x_min, x_max)
        ylim = (-z_abs, z_abs)
        zlim = (0.0, y_max)

        if focus is not None:
            region_mode = self.anim_region_combo.currentText()
            camera_mode = self.anim_camera_combo.currentText()
            zoom = self.anim_zoom_slider.value() / 100.0
            if camera_mode == "Вся траектория":
                pass
            elif region_mode != "Вся область":
                size_factor = self._zoom_size_factor()
                x_half, height_half, side_half = self._local_region_halves(region_mode)
                x_half *= size_factor
                height_half *= size_factor
                side_half *= size_factor
                cx = float(focus[0])
                cy = float(focus[1])
                cz = float(focus[2])

                xlim = (cx - x_half, cx + x_half)
                ylim = (cz - side_half, cz + side_half)
                zlim = (cy - height_half, cy + height_half)
            elif camera_mode.startswith("Следить"):
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

    @staticmethod
    def _local_region_halves(region_mode: str) -> tuple:
        """Размеры локального окна вокруг снаряда: X, высота, боковой снос."""
        sizes = {
            "Куб близкий": (160.0, 90.0, 90.0),
            "Куб средний": (360.0, 180.0, 180.0),
            "Куб дальний": (750.0, 350.0, 350.0),
        }
        return sizes.get(region_mode, sizes["Куб средний"])

    def _zoom_size_factor(self) -> float:
        """Плавная поправка размера окна: 35% примерно соответствует базовому кубу."""
        zoom = self.anim_zoom_slider.value()
        return float(np.clip(1.35 - 0.01 * zoom, 0.35, 1.35))

    def _wind_region_limits(
        self,
        xlim: tuple,
        ylim: tuple,
        zlim: tuple,
        focus: np.ndarray,
    ) -> tuple:
        """Возвращает область, внутри которой размещаются стрелки ветра."""
        region_mode = self.anim_region_combo.currentText()
        if region_mode == "Вся область":
            return xlim, ylim, zlim

        size_factor = self._zoom_size_factor()
        x_half, height_half, side_half = self._local_region_halves(region_mode)
        x_half *= size_factor
        height_half *= size_factor
        side_half *= size_factor

        rx = self._clip_interval(float(focus[0]) - x_half, float(focus[0]) + x_half, xlim)
        ry = self._clip_interval(float(focus[2]) - side_half, float(focus[2]) + side_half, ylim)
        rz = self._clip_interval(float(focus[1]) - height_half, float(focus[1]) + height_half, zlim)
        return rx, ry, rz

    @staticmethod
    def _clip_interval(low: float, high: float, bounds: tuple) -> tuple:
        """Обрезает интервал по границам оси, сохраняя ненулевую ширину."""
        clipped_low = max(float(bounds[0]), float(low))
        clipped_high = min(float(bounds[1]), float(high))
        if clipped_high - clipped_low < 1e-6:
            center = 0.5 * (float(bounds[0]) + float(bounds[1]))
            half = max(0.5 * (float(bounds[1]) - float(bounds[0])), 1.0)
            return center - half, center + half
        return clipped_low, clipped_high

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
            40: (4, 4, 3),
            80: (5, 5, 4),
            160: (6, 6, 5),
            256: (8, 7, 6),
            384: (8, 8, 7),
            512: (9, 8, 8),
            1024: (12, 9, 10),
            2048: (16, 12, 11),
            4096: (16, 16, 16),
        }
        nx, ny, nz = shapes.get(target, (5, 4, 4))
        x_vals = self._cell_centers(xlim[0], xlim[1], nx)
        z_vals = self._cell_centers(ylim[0], ylim[1], nz)
        y_vals = self._cell_centers(zlim[0], zlim[1], ny)
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        points = points[points[:, 1] >= 0.0]

        dx = (xlim[1] - xlim[0]) / max(nx, 1)
        dy = (zlim[1] - zlim[0]) / max(ny, 1)
        dz = (ylim[1] - ylim[0]) / max(nz, 1)
        grid_step = max(min(dx, dy, dz), 1.0)
        return points, grid_step

    def _build_wind_slice_grid(self, xlim: tuple, zlim: tuple, side_z: float) -> tuple:
        """Строит сетку стрелок на вертикальном X-Y срезе через снаряд."""
        target = int(self.anim_density_combo.currentText())
        shapes = {
            40: (8, 5),
            80: (10, 8),
            160: (16, 10),
            256: (16, 16),
            384: (24, 16),
            512: (32, 16),
            1024: (40, 26),
            2048: (56, 36),
            4096: (80, 52),
        }
        nx, ny = shapes.get(target, (10, 8))
        x_vals = self._cell_centers(xlim[0], xlim[1], nx)
        y_vals = self._cell_centers(zlim[0], zlim[1], ny)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")
        zz = np.full_like(xx, side_z)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        points = points[points[:, 1] >= 0.0]

        dx = (xlim[1] - xlim[0]) / max(nx, 1)
        dy = (zlim[1] - zlim[0]) / max(ny, 1)
        return points, max(min(dx, dy), 1.0)

    def _draw_vertical_slice_plane(self, xlim: tuple, zlim: tuple, side_z: float) -> None:
        """Показывает полупрозрачную плоскость вертикального среза."""
        x = np.array([xlim[0], xlim[1]])
        y = np.array([zlim[0], zlim[1]])
        xx, yy = np.meshgrid(x, y)
        zz = np.full_like(xx, side_z)
        self.main_ax.plot_surface(
            xx, zz, yy,
            color=(0.15, 0.45, 0.95, 0.08),
            shade=False,
            linewidth=0,
            antialiased=False,
        )

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
    def _wind_colors(speeds: np.ndarray) -> list:
        """Абсолютная шкала цвета ветра: м/с, а не относительный максимум кадра."""
        colors = []
        stops = [
            (0.0, np.array([0.18, 0.62, 1.00])),
            (5.0, np.array([0.22, 0.78, 0.84])),
            (15.0, np.array([0.36, 0.74, 0.32])),
            (30.0, np.array([1.00, 0.62, 0.12])),
            (50.0, np.array([0.92, 0.12, 0.10])),
        ]
        for speed in np.clip(speeds, 0.0, 50.0):
            for (s0, c0), (s1, c1) in zip(stops[:-1], stops[1:]):
                if speed <= s1:
                    frac = (float(speed) - s0) / max(s1 - s0, 1e-9)
                    rgb = c0 * (1.0 - frac) + c1 * frac
                    colors.append((float(rgb[0]), float(rgb[1]), float(rgb[2]), 0.84))
                    break
        return colors

    @staticmethod
    def _local_wind_label(vector: np.ndarray) -> str:
        """Короткая инженерная подпись локального ветра у снаряда."""
        speed = float(np.linalg.norm(vector))
        return (
            f"Ветер у снаряда |W|={speed:.1f} м/с "
            f"(Wx={vector[0]:+.1f}, Wy={vector[1]:+.1f}, Wz={vector[2]:+.1f})"
        )

    def _draw_dynamic_summary_panel(self, trajectories: list) -> None:
        """Показывает ключевые параметры и результаты итогового расчёта модели 8."""
        if not self.animation_data:
            return

        metadata = self.animation_data.get("metadata", {})
        times_list = self.animation_data.get("times", [])
        impacts = np.array([traj[-1] for traj in trajectories])
        max_heights = np.array([float(np.max(traj[:, 1])) for traj in trajectories])
        flight_times = np.array([float(times[-1]) for times in times_list]) if times_list else np.array([0.0])
        _, _, cx, cz, sx, sz = compute_impact_ellipse(trajectories, 1)

        panel = (
            "ИТОГ РАСЧЁТА\n"
            f"N: {len(trajectories)}\n"
            f"Сценарий: {metadata.get('scenario', '—')}\n"
            f"Wind: {float(metadata.get('wind_strength', 0.0)):.1f} м/с\n"
            f"Turbulence: {float(metadata.get('turbulence', 0.0)):.1f}\n"
            f"Средняя дальность: {float(np.mean(impacts[:, 0])):.1f} м\n"
            f"Средняя высота max: {float(np.mean(max_heights)):.1f} м\n"
            f"Среднее время: {float(np.mean(flight_times)):.2f} с\n"
            f"Мишень: X={cx:.1f} м, Z={cz:.1f} м\n"
            f"Разброс: σx={sx:.1f} м, σz={sz:.1f} м"
        )
        if self._presentation_export_active:
            panel = (
                "Итоги расчёта\n"
                f"Сценарий: {metadata.get('scenario', '—')}\n"
                f"N = {len(trajectories)}\n"
                f"Средняя дальность: {float(np.mean(impacts[:, 0])):.1f} м\n"
                f"Средняя макс. высота: {float(np.mean(max_heights)):.1f} м\n"
                f"Среднее время: {float(np.mean(flight_times)):.2f} с\n"
                f"Разброс: σx={sx:.1f} м, σz={sz:.1f} м"
            )
        self.main_ax.text2D(
            0.985, 0.985,
            panel,
            transform=self.main_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            color=(0.05, 0.05, 0.05),
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": (1.0, 1.0, 1.0, 0.90),
                "edgecolor": (0.65, 0.65, 0.65, 0.75),
                "linewidth": 0.9,
            },
        )

    def _draw_frame_mini_panel(
        self,
        position: np.ndarray,
        wind: np.ndarray,
        time_s: float,
        view_mode: str,
        camera_mode: str,
        region_mode: str,
    ) -> None:
        """Показывает численные параметры текущего кадра поверх графика."""
        metadata = self.animation_data.get("metadata", {}) if self.animation_data else {}
        speed = float(np.linalg.norm(wind))
        run_range = None
        run_max_height = None
        run_index = self.anim_run_combo.currentIndex() - 1
        if self.animation_data and run_index >= 0:
            trajectories = self.animation_data.get("trajectories", [])
            if run_index < len(trajectories):
                traj = trajectories[run_index]
                run_range = float(traj[-1, 0])
                run_max_height = float(np.max(traj[:, 1]))

        run_stats = ""
        if run_range is not None and run_max_height is not None:
            run_stats = (
                f"Дальность запуска: {run_range:.1f} м\n"
                f"Макс. высота запуска: {run_max_height:.1f} м\n"
            )
        panel = (
            "КАДР РАСЧЁТА\n"
            f"Сценарий: {metadata.get('scenario', '—')}\n"
            f"t: {time_s:.2f} c\n"
            f"Снаряд: X={position[0]:.1f} м, Y={position[1]:.1f} м, Z={position[2]:.1f} м\n"
            f"{run_stats}"
            f"Локальный ветер: |W|={speed:.1f} м/с\n"
            f"  Wx дальность: {wind[0]:+.1f} м/с\n"
            f"  Wy вертикаль: {wind[1]:+.1f} м/с\n"
            f"  Wz боковой:  {wind[2]:+.1f} м/с\n"
            f"Камера: {camera_mode}\n"
            f"Стрелки: {region_mode}, {view_mode}"
        )
        if self._presentation_export_active:
            panel = (
                "Кадр расчёта\n"
                f"Сценарий: {metadata.get('scenario', '—')}\n"
                f"t = {time_s:.2f} с\n"
                f"Положение: X={position[0]:.1f} м, Y={position[1]:.1f} м, Z={position[2]:.1f} м\n"
                f"{run_stats}"
                f"Локальный ветер: |W|={speed:.1f} м/с\n"
                f"Wx={wind[0]:+.1f}, Wy={wind[1]:+.1f}, Wz={wind[2]:+.1f} м/с"
            )
        self.main_ax.text2D(
            0.985, 0.985,
            panel,
            transform=self.main_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            color=(0.05, 0.05, 0.05),
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": (1.0, 1.0, 1.0, 0.90),
                "edgecolor": (0.65, 0.65, 0.65, 0.75),
                "linewidth": 0.9,
            },
        )

    def _set_animation_replay_enabled(self, enabled: bool) -> None:
        """Включает управление записью только для выбранного пуска."""
        self.anim_zoom_slider.setEnabled(enabled)
        self.anim_density_combo.setEnabled(enabled)
        self.anim_camera_combo.setEnabled(enabled)
        self.anim_region_combo.setEnabled(enabled)
        self.anim_view_combo.setEnabled(enabled)
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
        self.anim_export_gif_btn.setEnabled(False)
        self.anim_export_png_btn.setEnabled(False)
        self.anim_export_presentation_btn.setEnabled(False)
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

    def _on_animation_view_changed(self, _index: int) -> None:
        """Перерисовывает запись при смене камеры, области или режима среза."""
        if not self.animation_data:
            return
        if self.anim_run_combo.currentIndex() <= 0:
            return
        self._render_dynamic_wind_frame(self.animation_frame)

    def _export_animation_gif(self) -> None:
        """Экспортирует запись выбранного пуска в GIF для презентации."""
        if not self.animation_data:
            return

        try:
            from PIL import Image
        except ImportError:
            QMessageBox.warning(
                self,
                "Экспорт GIF",
                "Для экспорта GIF нужен пакет Pillow. В текущем Python он не найден.",
            )
            return

        trajectories = self.animation_data.get("trajectories", [])
        if not trajectories:
            return

        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")

        old_run_index = self.anim_run_combo.currentIndex()
        old_frame = self.animation_frame
        old_camera = self.anim_camera_combo.currentText()
        old_region = self.anim_region_combo.currentText()
        old_view = self.anim_view_combo.currentText()
        old_density = self.anim_density_combo.currentText()
        old_zoom = self.anim_zoom_slider.value()
        old_elev, old_azim = self._current_3d_view()

        camera_choice = QMessageBox.question(
            self,
            "Экспорт GIF",
            (
                "Какой ракурс камеры использовать?\n\n"
                "Да — оптимальный презентационный ракурс.\n"
                "Нет — текущий ракурс, который вы выставили мышью."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if camera_choice == QMessageBox.StandardButton.Yes:
            export_elev, export_azim = 24, -58
        else:
            export_elev, export_azim = old_elev, old_azim

        export_run_index = old_run_index if old_run_index > 0 else 1
        run_number = export_run_index
        metadata = self.animation_data.get("metadata", {})
        scenario = str(metadata.get("scenario", "wind"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"{scenario}_run_{run_number}_{timestamp}.gif"

        frame_count = len(self.animation_data["frame_times"])
        export_count = min(96, frame_count)
        export_frames = np.unique(
            np.linspace(0, frame_count - 1, export_count, dtype=int)
        )

        frames = []
        try:
            self.anim_export_gif_btn.setEnabled(False)
            self.anim_export_gif_btn.setText("GIF...")
            if old_run_index <= 0:
                self.anim_run_combo.setCurrentIndex(export_run_index)
                self.anim_camera_combo.setCurrentText("Следить за снарядом")
                self.anim_region_combo.setCurrentText("Куб средний")
                self.anim_view_combo.setCurrentText("3D поле")
                self.anim_density_combo.setCurrentText("80")
                self.anim_zoom_slider.setValue(35)

            self.main_ax.view_init(elev=export_elev, azim=export_azim)
            for i, frame_index in enumerate(export_frames, start=1):
                self._render_dynamic_wind_frame(int(frame_index))
                self.main_ax.view_init(elev=export_elev, azim=export_azim)
                self.main_canvas.draw()

                buf = io.BytesIO()
                self.main_fig.savefig(
                    buf,
                    format="png",
                    dpi=100,
                    facecolor="white",
                    bbox_inches="tight",
                )
                buf.seek(0)
                palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE")
                frames.append(Image.open(buf).convert("P", palette=palette).copy())
                buf.close()

                self.anim_export_gif_btn.setText(f"GIF {i}/{len(export_frames)}")
                QApplication.processEvents()

            if frames:
                frames[0].save(
                    export_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=130,
                    loop=0,
                    optimize=True,
                )
                self.info_display.setText(
                    f"{self.info_display.toPlainText()}\n\nGIF экспортирован:\n{export_path}"
                )
                QMessageBox.information(
                    self,
                    "Экспорт GIF",
                    f"GIF сохранён:\n{export_path}",
                )
        except Exception as exc:
            QMessageBox.warning(self, "Экспорт GIF", f"Не удалось создать GIF:\n{exc}")
        finally:
            self.anim_run_combo.setCurrentIndex(old_run_index)
            self.anim_camera_combo.setCurrentText(old_camera)
            self.anim_region_combo.setCurrentText(old_region)
            self.anim_view_combo.setCurrentText(old_view)
            self.anim_density_combo.setCurrentText(old_density)
            self.anim_zoom_slider.setValue(old_zoom)
            self.animation_frame = old_frame
            if old_run_index <= 0:
                self._render_dynamic_wind_summary()
            else:
                self._render_dynamic_wind_frame(old_frame)
                self.main_ax.view_init(elev=old_elev, azim=old_azim)
                self.main_canvas.draw_idle()
            self.anim_export_gif_btn.setText("GIF")
            self.anim_export_gif_btn.setEnabled(True)

    def _export_presentation_graph_png(self) -> None:
        """Сохраняет текущий вид модели 8 в лаконичном виде для презентации/диплома."""
        if not self.animation_data:
            return

        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")

        metadata = self.animation_data.get("metadata", {})
        scenario = str(metadata.get("scenario", "wind"))
        run_index = self.anim_run_combo.currentIndex()
        suffix = "summary" if run_index <= 0 else f"run_{run_index}_frame_{self.animation_frame:04d}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"{scenario}_{suffix}_presentation_{timestamp}.png"

        old_elev, old_azim = self._current_3d_view()
        camera_choice = QMessageBox.question(
            self,
            "Экспорт для презентации",
            (
                "Какой ракурс камеры использовать?\n\n"
                "Да — оптимальный презентационный ракурс.\n"
                "Нет — текущий ракурс, который вы выставили мышью."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if camera_choice == QMessageBox.StandardButton.Yes:
            export_elev, export_azim = 24, -58
        else:
            export_elev, export_azim = old_elev, old_azim

        old_text = self.anim_export_presentation_btn.text()
        try:
            self.anim_export_presentation_btn.setEnabled(False)
            self.anim_export_presentation_btn.setText("PNG...")
            self._presentation_export_active = True

            if run_index <= 0:
                self._render_dynamic_wind_summary()
            else:
                self._render_dynamic_wind_frame(self.animation_frame)
            self.main_ax.view_init(elev=export_elev, azim=export_azim)
            self.main_canvas.draw()
            self.main_fig.savefig(
                export_path,
                format="png",
                dpi=230,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.16,
            )
            QMessageBox.information(
                self,
                "Экспорт для презентации",
                f"График сохранён:\n{export_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Экспорт для презентации",
                f"Не удалось сохранить график:\n{exc}",
            )
        finally:
            self._presentation_export_active = False
            if run_index <= 0:
                self._render_dynamic_wind_summary()
            else:
                self._render_dynamic_wind_frame(self.animation_frame)
            self.main_ax.view_init(elev=old_elev, azim=old_azim)
            self.main_canvas.draw_idle()
            self.anim_export_presentation_btn.setText(old_text)
            self.anim_export_presentation_btn.setEnabled(True)

    def _export_current_graph_png(self) -> None:
        """Сохраняет текущий график в PNG с высоким разрешением."""
        if not self.animation_data:
            return

        self.animation_timer.stop()
        self.anim_play_btn.setText("Пуск")

        old_elev, old_azim = self._current_3d_view()
        camera_choice = QMessageBox.question(
            self,
            "Экспорт скриншота",
            (
                "Какой ракурс камеры использовать?\n\n"
                "Да — оптимальный презентационный ракурс.\n"
                "Нет — текущий ракурс, который вы выставили мышью."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if camera_choice == QMessageBox.StandardButton.Yes:
            export_elev, export_azim = 24, -58
        else:
            export_elev, export_azim = old_elev, old_azim

        metadata = self.animation_data.get("metadata", {})
        scenario = str(metadata.get("scenario", "wind"))
        run_index = self.anim_run_combo.currentIndex()
        frame_suffix = "summary"
        if run_index > 0:
            frame_suffix = f"run_{run_index}_frame_{self.animation_frame:04d}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / f"{scenario}_{frame_suffix}_{timestamp}.png"

        try:
            self.anim_export_png_btn.setEnabled(False)
            self.anim_export_png_btn.setText("PNG...")
            self.main_ax.view_init(elev=export_elev, azim=export_azim)
            self.main_canvas.draw()
            self.main_fig.savefig(
                export_path,
                format="png",
                dpi=220,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.12,
            )
            self.info_display.setText(
                f"{self.info_display.toPlainText()}\n\nСкриншот графика сохранён:\n{export_path}"
            )
            QMessageBox.information(
                self,
                "Экспорт скриншота",
                f"Скриншот сохранён:\n{export_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Экспорт скриншота",
                f"Не удалось сохранить скриншот:\n{exc}",
            )
        finally:
            self.main_ax.view_init(elev=old_elev, azim=old_azim)
            self.main_canvas.draw_idle()
            self.anim_export_png_btn.setText("Скриншот")
            self.anim_export_png_btn.setEnabled(True)

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
