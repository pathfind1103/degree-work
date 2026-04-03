import numpy as np

def get_name():
    return "1. Идеальный полет"

def get_info():
    return {
        "description": ("Движение тела в однородном поле тяжести. "
                        "Сопротивление среды отсутствует. Траектория представляет собой "
                        "симметричную параболу."),
        # Формула в формате LaTeX
        "formula": r"$y = x \cdot \tan(\alpha) - \frac{g \cdot x^2}{2 \cdot v_0^2 \cdot \cos^2(\alpha)}$",
        "parameters_info": {
            "v0": "Начальная скорость вылета тела (м/с)",
            "angle": "Угол между вектором скорости и горизонтом (градусы)",
            "g": "Ускорение свободного падения (стандарт 9.81)"
        }
    }

def get_params():
    return {"v0": "50", "angle": "45", "g": "9.81"}

def calculate(params):
    v0, alpha_deg, g = float(params['v0']), float(params['angle']), float(params['g'])
    alpha = np.radians(alpha_deg)
    t_max = (2 * v0 * np.sin(alpha)) / g
    t = np.linspace(0, t_max, 100)
    x = v0 * t * np.cos(alpha)
    y = v0 * t * np.sin(alpha) - 0.5 * g * t**2
    return x, y, "Идеальная траектория"
