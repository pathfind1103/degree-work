"""
Модуль графического интерфейса (View).
Содержит описание структуры окна и визуальных компонентов приложения.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QFormLayout, QTextEdit, QFrame,
    QPushButton, QLabel
)

# --- КОНСТАНТЫ ОФОРМЛЕНИЯ (PEP 8) ---
STYLE_INFO_BOX = (
    "background-color: #fdfdfd; "
    "border: 1px solid #ccc; "
    "color: black; "
    "padding: 5px;"
)
STYLE_CALC_BUTTON = (
    "background-color: #4CAF50; "
    "color: white; "
    "font-weight: bold; "
    "border-radius: 4px;"
)
FORMULA_PANEL_COLOR = '#f0f0f0'


class MainWindowUI(QMainWindow):
    """
    Базовый класс интерфейса главного окна.
    Отвечает исключительно за создание виджетов и их размещение (Layout).
    """

    def __init__(self):
        """Инициализация окна и базовых параметров оформления."""
        super().__init__()
        self.setWindowTitle("Программный комплекс: Расчет траекторий")
        self.resize(1200, 800)

        # Инициализация структурных компонентов
        self._setup_ui()

    def _setup_ui(self):
        """Создание и компоновка всех элементов управления интерфейсом."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        # Создание функциональных зон
        self._create_left_panel()
        self._create_chart_area()

    def _create_left_panel(self):
        """Создание боковой панели управления (настройки, описание, параметры)."""
        self.left_panel = QVBoxLayout()
        self.left_panel.setContentsMargins(15, 15, 15, 15)
        self.left_panel.setSpacing(12)

        # Блок выбора математической модели
        self.left_panel.addWidget(QLabel("<b>Выберите модель:</b>"))
        self.model_combo = QComboBox()
        self.left_panel.addWidget(self.model_combo)

        # Блок текстового описания физической задачи
        self.left_panel.addWidget(QLabel("<b>Описание задачи:</b>"))
        self.info_display = QTextEdit(readOnly=True)
        self.info_display.setMaximumHeight(180)
        self.info_display.setStyleSheet(STYLE_INFO_BOX)
        self.left_panel.addWidget(self.info_display)

        # Область отображения формулы (Matplotlib Canvas)
        self.left_panel.addWidget(QLabel("<b>Математический аппарат (LaTeX):</b>"))
        self.formula_fig, self.formula_ax = plt.subplots(
            figsize=(3, 1),
            facecolor=FORMULA_PANEL_COLOR
        )
        self.formula_ax.axis('off')
        self.formula_canvas = FigureCanvas(self.formula_fig)
        self.formula_canvas.setMaximumHeight(100)
        self.left_panel.addWidget(self.formula_canvas)

        # Разделительная линия
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.left_panel.addWidget(separator)

        # Динамическая область параметров (Form Layout)
        self.left_panel.addWidget(QLabel("<b>Параметры системы:</b>"))
        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        self.params_layout.setSpacing(10)
        self.left_panel.addWidget(self.params_container)

        # Кнопка запуска расчета
        self.calc_btn = QPushButton("ЗАПУСТИТЬ РАСЧЕТ", minimumHeight=50)
        self.calc_btn.setStyleSheet(STYLE_CALC_BUTTON)
        self.left_panel.addWidget(self.calc_btn)

        # Прижимаем элементы к верху и добавляем панель в главный слой
        self.left_panel.addStretch()
        self.main_layout.addLayout(self.left_panel, 1)

    def _create_chart_area(self):
        """Создание области графиков в центральной части окна."""
        # Создаем фигуру Matplotlib для основного графика
        self.main_fig, self.main_ax = plt.subplots(figsize=(8, 6))
        self.main_canvas = FigureCanvas(self.main_fig)

        # Добавляем холст в главный слой (соотношение сторон 3 к 1)
        self.main_layout.addWidget(self.main_canvas, 3)
