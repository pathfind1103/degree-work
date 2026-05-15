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

from ui_main import MainWindowUI
from mc_renderer import render_monte_carlo_6dof, compute_mean_trajectory_2d


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

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.calc_btn.clicked.connect(self.run_calculation)

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
                self.models[model_name] = module
                self.model_combo.addItem(model_name)
            except (ImportError, AttributeError) as err:
                print(f"[Загрузчик] Пропущен {file_path.name}: {err}")

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

    def _clear_params(self) -> None:
        """Удаляет все виджеты параметров из формы."""
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.param_widgets.clear()

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

        # Безопасная замена осей на Windows:
        # НЕ используем figure.clear() / figure.clf() — они разрушают
        # внутреннюю связь Figure<->FigureCanvasQTAgg и вызывают 0xC0000409.
        # Вместо этого удаляем каждую ось через ax.remove() и создаём новую.
        for ax in self.main_fig.axes[:]:
            ax.remove()

        if is_3d:
            self.main_ax = self.main_fig.add_subplot(111, projection='3d')
            if model_name.startswith("6."):
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

        self.main_ax.set_title(f"Результат: {model_name}", pad=10)
        self.main_canvas.draw()

    # ------------------------------------------------------------------
    # Рендереры (приватные)
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrajectoryApp()
    window.show()
    sys.exit(app.exec())