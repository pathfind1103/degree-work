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
    QSlider, QToolButton,
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
    "font-size: 12px;"
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
    "min-height: 110px;"
)
FORMULA_BG_COLOR = '#f0f0f0'
STYLE_SECTION_TOGGLE = (
    "QToolButton {"
    "color: white;"
    "font-weight: bold;"
    "border: none;"
    "padding: 4px 0;"
    "text-align: left;"
    "}"
)


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
        inner.setMinimumWidth(260)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        # Выбор модели
        layout.addWidget(QLabel("<b>Выберите модель:</b>"))
        self.model_combo = QComboBox()
        layout.addWidget(self.model_combo)

        self.scenario_preset_label = QLabel("<b>Сценарий демонстрации:</b>")
        self.scenario_preset_combo = QComboBox()
        self.scenario_preset_label.setVisible(False)
        self.scenario_preset_combo.setVisible(False)
        layout.addWidget(self.scenario_preset_label)
        layout.addWidget(self.scenario_preset_combo)

        # Описание задачи
        self.info_toggle_btn, self.info_section = self._create_collapsible_section(
            "Описание задачи",
            expanded=True,
        )
        layout.addWidget(self.info_toggle_btn)
        self.info_display = QTextEdit(readOnly=True)
        self.info_display.setMinimumHeight(190)
        self.info_display.setStyleSheet(STYLE_INFO_BOX)
        self.info_section.layout().addWidget(self.info_display)
        layout.addWidget(self.info_section)

        # Формула (LaTeX → PNG → QLabel)
        self.formula_toggle_btn, self.formula_section = self._create_collapsible_section(
            "Математический аппарат",
            expanded=True,
        )
        layout.addWidget(self.formula_toggle_btn)
        self.formula_label = QLabel()
        self.formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formula_label.setStyleSheet(STYLE_FORMULA_LABEL)
        self.formula_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.formula_label.setWordWrap(True)
        formula_scroll = QScrollArea()
        formula_scroll.setWidget(self.formula_label)
        formula_scroll.setWidgetResizable(True)
        formula_scroll.setMinimumHeight(185)
        formula_scroll.setMaximumHeight(260)
        formula_scroll.setFrameShape(QFrame.Shape.NoFrame)
        formula_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        formula_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.formula_section.layout().addWidget(formula_scroll)
        layout.addWidget(self.formula_section)

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
        self.params_layout.setSpacing(8)
        self.params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.params_container.setStyleSheet(
            "QLabel { font-size: 12px; }"
            "QLineEdit { min-height: 24px; padding: 2px 6px; font-size: 12px; }"
        )
        layout.addWidget(self.params_container)

        # Кнопка расчёта
        self.calc_btn = QPushButton("ЗАПУСТИТЬ РАСЧЁТ", minimumHeight=48)
        self.calc_btn.setStyleSheet(STYLE_CALC_BUTTON)
        layout.addWidget(self.calc_btn)

        self.export_basic_graph_btn = QPushButton("ЭКСПОРТ ГРАФИКА", minimumHeight=36)
        self.export_basic_graph_btn.setEnabled(False)
        layout.addWidget(self.export_basic_graph_btn)

        self.export_compare_2d_btn = QPushButton("СРАВНИТЬ С ВАКУУМОМ", minimumHeight=36)
        self.export_compare_2d_btn.setEnabled(False)
        layout.addWidget(self.export_compare_2d_btn)

        self.export_diploma_graph_btn = QPushButton("ГРАФИК ДЛЯ ДИПЛОМА", minimumHeight=36)
        self.export_diploma_graph_btn.setEnabled(False)
        self.export_diploma_graph_btn.setVisible(False)
        layout.addWidget(self.export_diploma_graph_btn)

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
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        self.main_fig = Figure(figsize=(9, 7))
        self.main_fig.set_tight_layout(True)
        self.main_ax = self.main_fig.add_subplot(111)

        self.main_canvas = FigureCanvas(self.main_fig)
        self.main_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        chart_layout.addWidget(self.main_canvas, 1)

        self.animation_panel = QWidget()
        self.animation_panel.setVisible(False)
        anim_layout = QVBoxLayout(self.animation_panel)
        anim_layout.setContentsMargins(10, 6, 10, 6)
        anim_layout.setSpacing(5)
        anim_top_layout = QHBoxLayout()
        anim_top_layout.setSpacing(8)
        anim_bottom_layout = QHBoxLayout()
        anim_bottom_layout.setSpacing(8)

        self.anim_run_combo = QComboBox()
        self.anim_run_combo.setMinimumWidth(130)
        self.anim_camera_combo = QComboBox()
        self.anim_camera_combo.addItems(["Следить за снарядом", "Вся траектория"])
        self.anim_camera_combo.setMinimumWidth(145)
        self.anim_region_combo = QComboBox()
        self.anim_region_combo.addItems(["Куб средний", "Куб близкий", "Куб дальний", "Вся область"])
        self.anim_region_combo.setMinimumWidth(110)
        self.anim_view_combo = QComboBox()
        self.anim_view_combo.addItems(["3D поле", "Вертикальный срез"])
        self.anim_view_combo.setMinimumWidth(130)
        self.anim_density_combo = QComboBox()
        self.anim_density_combo.addItems(["40", "80", "160", "256", "384", "512", "1024", "2048", "4096"])
        self.anim_density_combo.setCurrentText("80")
        self.anim_density_combo.setMinimumWidth(82)
        self.anim_zoom_label = QLabel("Zoom 35%")
        self.anim_zoom_label.setMinimumWidth(76)
        self.anim_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.anim_zoom_slider.setMinimum(0)
        self.anim_zoom_slider.setMaximum(100)
        self.anim_zoom_slider.setValue(35)
        self.anim_zoom_slider.setFixedWidth(130)
        self.anim_prev_btn = QPushButton("Назад")
        self.anim_play_btn = QPushButton("Пуск")
        self.anim_next_btn = QPushButton("Вперёд")
        self.anim_export_gif_btn = QPushButton("GIF")
        self.anim_export_gif_btn.setMinimumWidth(54)
        self.anim_export_png_btn = QPushButton("Скриншот")
        self.anim_export_png_btn.setMinimumWidth(82)
        self.anim_export_collage_btn = QPushButton("4 КАДРА")
        self.anim_export_collage_btn.setMinimumWidth(78)
        self.anim_export_collage_btn.setToolTip(
            "Сохранить четыре стадии выбранного запуска одним изображением"
        )
        self.anim_export_presentation_btn = QPushButton("Презентация")
        self.anim_export_presentation_btn.setMinimumWidth(100)
        self.anim_slider = QSlider(Qt.Orientation.Horizontal)
        self.anim_slider.setMinimum(0)
        self.anim_slider.setMaximum(0)
        self.anim_time_label = QLabel("t = 0.00 c")
        self.anim_time_label.setMinimumWidth(110)
        self.anim_time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        anim_top_layout.addWidget(self.anim_run_combo)
        anim_top_layout.addWidget(self.anim_camera_combo)
        anim_top_layout.addWidget(self.anim_region_combo)
        anim_top_layout.addWidget(self.anim_view_combo)
        anim_top_layout.addWidget(QLabel("Стрелки"))
        anim_top_layout.addWidget(self.anim_density_combo)
        anim_top_layout.addWidget(self.anim_zoom_label)
        anim_top_layout.addWidget(self.anim_zoom_slider)
        anim_top_layout.addStretch(1)
        anim_top_layout.addWidget(self.anim_export_png_btn)
        anim_top_layout.addWidget(self.anim_export_collage_btn)
        anim_top_layout.addWidget(self.anim_export_presentation_btn)
        anim_top_layout.addWidget(self.anim_export_gif_btn)

        anim_bottom_layout.addWidget(self.anim_prev_btn)
        anim_bottom_layout.addWidget(self.anim_play_btn)
        anim_bottom_layout.addWidget(self.anim_next_btn)
        anim_bottom_layout.addWidget(self.anim_slider, 1)
        anim_bottom_layout.addWidget(self.anim_time_label)

        anim_layout.addLayout(anim_top_layout)
        anim_layout.addLayout(anim_bottom_layout)

        chart_layout.addWidget(self.animation_panel)
        self.splitter.addWidget(chart_widget)

    def _create_collapsible_section(self, title: str, expanded: bool) -> tuple:
        """Создаёт простую разворачиваемую секцию левой панели."""
        button = QToolButton()
        button.setText(f"{'▼' if expanded else '▶'} {title}")
        button.setCheckable(True)
        button.setChecked(expanded)
        button.setStyleSheet(STYLE_SECTION_TOGGLE)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body.setVisible(expanded)

        def toggle(checked: bool) -> None:
            body.setVisible(checked)
            button.setText(f"{'▼' if checked else '▶'} {title}")

        button.toggled.connect(toggle)
        return button, body

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
