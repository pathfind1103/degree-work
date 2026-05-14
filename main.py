"""
Главный модуль приложения для расчета траекторий.
Выполняет роль контроллера, связывая UI-представление с математическими моделями.
"""

import sys
import importlib.util
import io
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import QApplication, QLineEdit
from PyQt6.QtGui import QPixmap

from ui_main import MainWindowUI


class TrajectoryApp(MainWindowUI):
    """
    Класс-контроллер приложения.
    Управляет жизненным циклом программы, загрузкой моделей и обработкой событий.
    """

    def __init__(self):
        """Инициализация приложения и подключение обработчиков событий."""
        super().__init__()
        self.models = {}
        self.param_widgets = {}

        # Подключение обработчиков сигналов интерфейса
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.calc_btn.clicked.connect(self.run_calculation)

        # Начальная загрузка доступных расчетных модулей
        self._load_models()

    def _load_models(self):
        """
        Динамически загружает файлы моделей из директории 'models'.
        """
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)

        for file_path in models_dir.glob("*.py"):
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
                print(f"Ошибка при загрузке модели {file_path.name}: {err}")

    def _on_model_changed(self):
        """
        Обновляет интерфейс при выборе другой модели.
        Использует QPixmap для стабильного отображения LaTeX без вылетов.
        """
        model_name = self.model_combo.currentText()
        if not model_name or model_name not in self.models:
            return

        model = self.models[model_name]
        info = model.get_info()

        # 1. Обновление текстового описания
        vars_info = info.get("parameters_info", {})
        vars_text = "\n".join([f"• {k} — {v}" for k, v in vars_info.items()])
        description = info.get("description", "Описание отсутствует.")
        self.info_display.setText(f"{description}\n\nОбозначения:\n{vars_text}")

        # 2. Безопасный рендеринг формулы (фикс вылетов и обрезки)
        try:
            self.formula_ax.clear()
            self.formula_ax.axis('off')

            # Рисуем текст формулы
            self.formula_ax.text(
                0.5, 0.5,
                info.get("formula", ""),
                fontsize=13,
                ha='center',
                va='center',
                math_fontfamily='cm'
            )

            # Вместо tight_layout используем bbox_inches при сохранении
            buf = io.BytesIO()
            self.formula_fig.savefig(
                buf,
                format='png',
                dpi=110,
                facecolor='#f0f0f0',
                bbox_inches='tight', # Убирает пустые поля вокруг формулы
                pad_inches=0.1
            )
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            self.formula_label.setPixmap(pixmap)
            buf.close()
        except Exception as e:
            self.formula_label.setText("Ошибка рендеринга формулы")
            print(f"Render error: {e}")

        # 3. Пересоздание динамической формы параметров
        self._clear_params()
        for p_name, p_val in model.get_params().items():
            edit = QLineEdit(str(p_val))
            self.params_layout.addRow(p_name, edit)
            self.param_widgets[p_name] = edit

    def _clear_params(self):
        """Удаляет все текущие виджеты параметров из формы."""
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.param_widgets.clear()

    def run_calculation(self):
        """Выполнение расчета и отрисовка графика (поддерживает 2D и интерактивный 3D режимы)."""
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        params = {name: widget.text() for name, widget in self.param_widgets.items()}

        try:
            # Выполнение физического расчета.
            # Теперь мы считываем еще и флаг is_3d, который возвращает модель
            result, is_3d, label = self.models[model_name].calculate(params)

            # Пересоздаем оси графика в зависимости от размерности (2D или 3D)
            self.main_fig.clf()  # Полностью очищаем фигуру

            if is_3d:
                self.main_fig.clf()
                self.main_ax = self.main_fig.add_subplot(111, projection='3d')

                # Логика для 6-DoF баллистики снаряда
                if model_name.startswith("6."):
                    all_trajs = result  # Теперь тут просто список массивов

                    # 1. Отрисовка случайных реализаций (фоновое облако)
                    for traj in all_trajs:
                        self.main_ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color='dodgerblue', alpha=0.15, lw=1)

                        # 2. МАТЕМАТИЧЕСКИЙ РАСЧЕТ ИСТИННОЙ ПРЕДСКАЗАННОЙ ТРАЕКТОРИИ (Мат. ожидание)
                        # Находим максимальное количество шагов среди всех полетов
                        max_len = max(t.shape[0] for t in all_trajs)
                        mean_trajectory = []

                        for step in range(max_len):
                            points_at_step = []
                            for t in all_trajs:
                                # Проверяем количество строк (шагов) в массиве через t.shape[0]
                                if step < t.shape[0]:
                                    points_at_step.append(t[step, :])
                                else:
                                    points_at_step.append(t[-1, :])

                            # Считаем среднее арифметическое координат X, Y, Z на данном шаге
                            mean_trajectory.append(np.mean(points_at_step, axis=0))

                        mean_trajectory = np.array(mean_trajectory)

                    # 3. СБОР ТОЧЕК ПАДЕНИЯ И РАСЧЕТ ХАРАКТЕРИСТИК РАССЕИВАНИЯ
                    impacts = np.array([t[-1, :] for t in all_trajs])
                    x_hits = impacts[:, 0]
                    z_hits = impacts[:, 2]

                    # Центр эллипса — это строго точка приземления усредненной траектории
                    center_x = mean_trajectory[-1, 0]
                    center_z = mean_trajectory[-1, 2]

                    # Дисперсия (СКО) разброса вокруг центра предсказания
                    std_x = np.std(x_hits) if len(x_hits) > 1 else 10.0
                    std_z = np.std(z_hits) if len(z_hits) > 1 else 10.0

                    # Генерация оранжевого контура эллипса (2-сигма)
                    theta = np.linspace(0, 2 * np.pi, 100)
                    ellipse_x = center_x + 2 * std_x * np.cos(theta)
                    ellipse_z = center_z + 2 * std_z * np.sin(theta)
                    ellipse_y = np.zeros_like(theta)

                    # Отрисовка контура эллипса
                    self.main_ax.plot(ellipse_x, ellipse_z, ellipse_y, color='darkorange', lw=2, ls='--',
                                      label="Эллипс рассеивания (2σ)")

                    # Отрисовка Итоговой Предсказанной Траектории метода Монте-Карло
                    self.main_ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 2], mean_trajectory[:, 1], color='red',
                                      lw=3, label="Результат Монте-Карло (Истинное предсказание)")

                    # Маркер центра падения предсказания
                    self.main_ax.scatter([center_x], [center_z], color = 'red', s = 60, marker = 'X', zorder = 5)
                    self.main_ax.plot([], [], [], color='dodgerblue', lw=1,
                                      label=f"Случайные выстрелы (N={len(all_trajs)})")
                else:
                    # Стандартный 3D режим для модели №4
                    self.main_ax.view_init(elev=25, azim=-60)
                    for traj in result:
                        self.main_ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], color='crimson', alpha=0.2, lw=1)
                    self.main_ax.plot([], [], [], color='crimson', lw=1.5, label=label)

                self.main_ax.set_xlabel("Дальность X (м)")
                self.main_ax.set_ylabel("Боковой снос Z (м)")
                self.main_ax.set_zlabel("Высота Y (м)")
                self.main_ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.05))

            self.main_ax.set_title(f"Результат: {model_name}")
            self.main_ax.legend()
            self.main_canvas.draw()

        except Exception as err:
            self.info_display.setText(f"ОШИБКА РАСЧЕТА:\n{err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrajectoryApp()
    window.show()
    sys.exit(app.exec())
