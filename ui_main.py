"""
Модуль графического интерфейса (View).

Архитектурные решения:
  - QSplitter между левой панелью и областью графика:
    пользователь может перетаскивать разделитель мышью.
  - formula_fig — отдельная Figure(), НЕ привязанная ни к какому Qt-канвасу.
    Рендеринг выполняется через FigureCanvasAgg изолированно, результат
    передаётся в QLabel как PNG-байты. Это исключает конфликт рендереров
    на Windows (0xC0000409).
  - main_fig — Figure(), привязанная к FigureCanvasQTAgg. Создаётся
    без участия pyplot, чтобы не было конфликта двух event loop.
"""

import io

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QFormLayout, QTextEdit, QFrame,
    QPushButton, QLabel, QSizePolicy, QSplitter, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap


# ---------------------------------------------------------------------------
# Константы оформления
# ---------------------------------------------------------------------------

STYLE_INFO_BOX = (
    "background-color: #fdfdfd;"
    "border: 1px solid #ccc;"
    "color: black;"
    "padding: 5px;"
)
STYLE_CALC_BUTTON = (
    "background-color: #4CAF50;"
    "color: white;"
    "font-weight: bold;"
    "border-radius: 4px;"
)
STYLE_FORMULA_LABEL = (
    "background-color: #f0f0f0;"
    "border: 1px solid #ccc;"
    "border-radius: 2px;"
    "min-height: 80px;"
)
FORMULA_BG_COLOR = '#f0f0f0'


class MainWindowUI(QMainWindow):
    """
    Базовый класс интерфейса (View в MVC).

    Создаёт все виджеты и размещает их. Бизнес-логика — в TrajectoryApp.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Программный комплекс: Расчёт траекторий")
        self.resize(1300, 820)
        self._setup_ui()

    # ------------------------------------------------------------------
    # Сборка интерфейса
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Создаёт корневой виджет с QSplitter внутри."""
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # QSplitter — горизонтальный, позволяет тянуть границу мышью
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(5)
        self.splitter.setChildrenCollapsible(False)   # панели не схлопываются

        self._create_left_panel()    # добавляет виджет в splitter
        self._create_chart_area()    # добавляет виджет в splitter

        # Начальное соотношение: левая панель 300 px, остальное — график
        self.splitter.setSizes([300, 1000])

        root_layout.addWidget(self.splitter)

    def _create_left_panel(self) -> None:
        """
        Создаёт прокручиваемую левую панель управления.

        Оборачивается в QScrollArea, чтобы при узкой панели
        содержимое не обрезалось, а появлялась полоса прокрутки.
        """
        inner = QWidget()
        inner.setMinimumWidth(220)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Выбор модели
        layout.addWidget(QLabel("<b>Выберите модель:</b>"))
        self.model_combo = QComboBox()
        layout.addWidget(self.model_combo)

        # Описание задачи
        layout.addWidget(QLabel("<b>Описание задачи:</b>"))
        self.info_display = QTextEdit(readOnly=True)
        self.info_display.setMaximumHeight(160)
        self.info_display.setStyleSheet(STYLE_INFO_BOX)
        layout.addWidget(self.info_display)

        # Формула (LaTeX → PNG → QLabel)
        layout.addWidget(QLabel("<b>Математический аппарат:</b>"))
        self.formula_label = QLabel()
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.setStyleSheet(STYLE_FORMULA_LABEL)
        self.formula_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.formula_label.setWordWrap(True)
        layout.addWidget(self.formula_label)

        # Отдельная Figure для рендеринга формул — НЕ привязана к Qt-канвасу.
        # FigureCanvasAgg создаётся каждый раз заново при рендеринге
        # (в методе render_formula контроллера), чтобы не держать лишний канвас.
        self.formula_fig = Figure(figsize=(3.8, 1.8), facecolor=FORMULA_BG_COLOR)
        self.formula_ax = self.formula_fig.add_subplot(111)
        self.formula_ax.axis('off')

        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Параметры модели
        layout.addWidget(QLabel("<b>Параметры системы:</b>"))
        self.params_container = QWidget()
        self.params_layout = QFormLayout(self.params_container)
        self.params_layout.setSpacing(7)
        layout.addWidget(self.params_container)

        # Кнопка расчёта
        self.calc_btn = QPushButton("ЗАПУСТИТЬ РАСЧЁТ", minimumHeight=48)
        self.calc_btn.setStyleSheet(STYLE_CALC_BUTTON)
        layout.addWidget(self.calc_btn)

        layout.addStretch(1)

        # Прокручиваемая обёртка
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.splitter.addWidget(scroll)

    def _create_chart_area(self) -> None:
        """
        Создаёт область графика.

        Figure создаётся через Figure() без pyplot и вручную привязывается
        к FigureCanvasQTAgg — нет конфликта двух event loop на Windows.
        """
        self.main_fig = Figure(figsize=(9, 7))
        self.main_fig.set_tight_layout(True)
        self.main_ax = self.main_fig.add_subplot(111)

        self.main_canvas = FigureCanvas(self.main_fig)
        self.main_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.splitter.addWidget(self.main_canvas)

    # ------------------------------------------------------------------
    # Публичный метод рендеринга формулы
    # ------------------------------------------------------------------

    def render_formula(self, formula_text: str) -> None:
        """
        Рендерит LaTeX-строку в formula_label через Agg (без Qt-рендерера).

        Создаёт временный FigureCanvasAgg, рендерит в BytesIO, грузит в QPixmap.
        Временный канвас сразу уничтожается — нет утечки ресурсов.

        Поддерживаются только команды mathtext matplotlib:
          \\frac, \\sqrt, \\cdot, \\times, \\mathbf, \\dot, \\alpha и т.д.
        НЕ поддерживаются: \\bigl, \\bigr, \\boldsymbol, \\left, \\right.

        Args:
            formula_text: Строка формулы (может содержать \\n\\n как разделитель).
        """
        try:
            self.formula_ax.clear()
            self.formula_ax.axis('off')
            self.formula_ax.text(
                0.5, 0.5,
                formula_text,
                fontsize=11,
                ha='center',
                va='center',
                math_fontfamily='cm',
                transform=self.formula_ax.transAxes,
                wrap=False,
            )

            # Временный Agg-канвас — полностью изолирован от Qt
            agg = FigureCanvasAgg(self.formula_fig)
            buf = io.BytesIO()
            agg.print_figure(
                buf,
                format='png',
                dpi=96,
                facecolor=FORMULA_BG_COLOR,
                bbox_inches='tight',
                pad_inches=0.15,
            )
            buf.seek(0)

            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            buf.close()

            if not pixmap.isNull():
                self.formula_label.setPixmap(pixmap)
                self.formula_label.setText("")
            else:
                self.formula_label.setText("Ошибка загрузки PNG")

        except Exception as exc:
            # Показываем текст формулы как fallback — лучше, чем пустота
            self.formula_label.setPixmap(QPixmap())
            self.formula_label.setText(formula_text)
            print(f"[Формула] {exc}")