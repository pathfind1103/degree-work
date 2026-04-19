"""
Главный модуль приложения для расчета траекторий.
Выполняет роль контроллера, связывая UI-представление с математическими моделями.
"""

import sys
import importlib.util
import io
from pathlib import Path

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
        """Выполнение расчета и отрисовка графика."""
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        params = {name: widget.text() for name, widget in self.param_widgets.items()}

        try:
            x, y, label = self.models[model_name].calculate(params)

            self.main_ax.clear()
            self.main_ax.plot(x, y, 'b-', lw=2, label=label)
            self.main_ax.set_title(f"Результат: {model_name}")
            self.main_ax.set_xlabel("Дистанция (м)")
            self.main_ax.set_ylabel("Высота (м)")
            self.main_ax.grid(True, ls='--', alpha=0.6)
            self.main_ax.legend()
            self.main_canvas.draw()
        except Exception as err:
            self.info_display.setText(f"ОШИБКА РАСЧЕТА:\n{err}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrajectoryApp()
    window.show()
    sys.exit(app.exec())
