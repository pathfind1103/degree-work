import sys
import os
import importlib.util

# Настройка для работы PyQt6 с Matplotlib
os.environ["QT_API"] = "pyqt6"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QComboBox, QFormLayout, QLineEdit,
                             QPushButton, QLabel, QTextEdit, QFrame)
from PyQt6.QtCore import Qt


class ThesisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Программный комплекс: Расчет траекторий (Диплом)")
        self.resize(1200, 800)

        self.models = {}  # Словарь для хранения подгруженных модулей

        # Основной виджет и слой
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- ЛЕВАЯ ПАНЕЛЬ (Управление) ---
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)
        left_panel.setSpacing(10)

        # 1. Выбор модели
        left_panel.addWidget(QLabel("<b>Выберите математическую модель:</b>"))
        self.model_select = QComboBox()
        self.model_select.currentIndexChanged.connect(self.update_ui_for_model)
        left_panel.addWidget(self.model_select)

        # 2. Описание (Текстовое поле)
        left_panel.addWidget(QLabel("<b>Описание:</b>"))
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(120)
        self.info_box.setStyleSheet("background-color: #fdfdfd; border: 1px solid #ccc; padding: 5px;")
        left_panel.addWidget(self.info_box)

        # 3. Формула (Matplotlib canvas для LaTeX)
        left_panel.addWidget(QLabel("<b>Математический аппарат:</b>"))
        self.formula_fig = plt.figure(figsize=(3, 1), facecolor='#f0f0f0')
        self.formula_ax = self.formula_fig.add_subplot(111)
        self.formula_ax.axis('off')
        self.formula_canvas = FigureCanvas(self.formula_fig)
        self.formula_canvas.setMaximumHeight(100)
        left_panel.addWidget(self.formula_canvas)

        # Разделитель
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_panel.addWidget(line)

        # 4. Параметры (Форма ввода)
        left_panel.addWidget(QLabel("<b>Параметры системы:</b>"))
        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        left_panel.addWidget(self.params_container)

        # 5. Кнопка запуска
        self.start_btn = QPushButton("ЗАПУСТИТЬ РАСЧЕТ")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_btn.clicked.connect(self.run_calculation)
        left_panel.addWidget(self.start_btn)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # --- ЦЕНТРАЛЬНАЯ ПАНЕЛЬ (График) ---
        self.main_figure, self.main_ax = plt.subplots(figsize=(8, 6))
        self.main_canvas = FigureCanvas(self.main_figure)
        main_layout.addWidget(self.main_canvas, 3)

        # Инициализация
        self.load_models_from_folder()

    def load_models_from_folder(self):
        """Динамическая загрузка файлов из папки models"""
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for filename in os.listdir(models_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                module_name = filename[:-3]
                path = os.path.join(models_dir, filename)

                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                display_name = module.get_name()
                self.models[display_name] = module
                self.model_select.addItem(display_name)

    def update_ui_for_model(self):
        """Обновление описания, формулы и полей ввода при смене модели"""
        model_name = self.model_select.currentText()
        if not model_name: return

        current_model = self.models[model_name]
        info = current_model.get_info()

        # 1. Обновляем текст описания (делаем его черным и добавляем описание переменных)
        self.info_box.clear()
        self.info_box.setTextColor(Qt.GlobalColor.black)  # Принудительно черный цвет

        description_text = info.get("description", "")
        params_info = info.get("parameters_info", {})

        # Формируем расширенный текст
        full_text = f"{description_text}\n\nОбозначения переменных:\n"
        for var, desc in params_info.items():
            full_text += f"• {var} — {desc}\n"

        self.info_box.setText(full_text)

        # 2. Отрисовка LaTeX формулы без осей и рамок
        self.formula_ax.clear()
        # Убираем все линии и деления осей
        self.formula_ax.set_axis_off()

        # Пишем текст по центру (0.5, 0.5)
        self.formula_ax.text(0.5, 0.5, info.get("formula", ""),
                             fontsize=14, ha='center', va='center',
                             transform=self.formula_ax.transAxes)

        # Убираем лишние отступы вокруг формулы
        self.formula_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.formula_canvas.draw()

        # 3. Пересоздаем поля ввода (этот блок оставляем как был)
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        params_config = current_model.get_params()
        for p_name, p_val in params_config.items():
            input_field = QLineEdit(str(p_val))
            self.params_layout.addRow(p_name, input_field)

    def run_calculation(self):
        """Сбор данных из полей и вызов метода расчета модели"""
        model_name = self.model_select.currentText()
        if not model_name: return

        current_model = self.models[model_name]

        # Собираем значения из QLineEdit
        params = {}
        for i in range(self.params_layout.rowCount()):
            label_widget = self.params_layout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget()
            input_widget = self.params_layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget()
            params[label_widget.text()] = input_widget.text()

        try:
            x, y, label = current_model.calculate(params)

            # Отрисовка на главном графике
            self.main_ax.clear()
            self.main_ax.plot(x, y, 'b-', linewidth=2, label=label)
            self.main_ax.set_xlabel("Дистанция (м)")
            self.main_ax.set_ylabel("Высота (м)")
            self.main_ax.set_title(f"Результат расчета: {model_name}")
            self.main_ax.grid(True, linestyle='--', alpha=0.7)
            self.main_ax.legend()
            self.main_canvas.draw()

        except Exception as e:
            self.info_box.setText(f"Ошибка в расчетах:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThesisApp()
    window.show()
    sys.exit(app.exec())
