import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# Форматирование даты
def format_region(day, month):
    day_i = (day - 1) // 7 + 3
    day_j = ((day - 1) % 7) + 1
    month_i = (month - 1) // 6 + 1
    month_j = ((month - 1) % 6) + 1
    return day_i, day_j, month_i, month_j


# Форма календаря
def create_area():
    area = np.zeros((9, 9), dtype=int)
    area[:, [0, -1]] = -1
    area[[0, -1], :] = -1
    area[0, -2] = area[1, -2] = area[2, -2] = -1
    area[-2, 4:7] = -1
    area[-2, -2] = -1
    return area


# Поворот фигуры
def rotate(matrix):
    return np.rot90(matrix, k=-1)


# Отзеркаливание фигуры
def reflect(matrix):
    return np.flip(matrix, axis=1)


# Проверка возможности размещения фигуры
def can_place(area, figure, start_i, start_j):
    for i, j in product(range(figure.shape[0]), range(figure.shape[1])):
        if figure[i][j] == 1:
            if (start_i + i >= area.shape[0]) or (start_j + j >= area.shape[1]) or (area[start_i + i][start_j + j] != 0):
                return False
    return True


# Размещение фигуры на поле
def place_figure(area, figure, start_i, start_j, k):
    for i, j in product(range(figure.shape[0]), range(figure.shape[1])):
        if figure[i][j] == 1:
            area[start_i + i][start_j + j] = k


# Все уникальные варианты расположения фигуры
def get_unique_variants(figure):
    variants = set()
    current = figure
    for _ in range(4):
        current_tuple = tuple(map(tuple, current))
        variants.add(current_tuple)
        reflected = reflect(current)
        reflected_tuple = tuple(map(tuple, reflected))
        variants.add(reflected_tuple)
        current = rotate(current)
    return [np.array(v) for v in variants]


def backtrack(area, figures):
    def solve(fig_index):
        if fig_index >= len(figures):
            return True
        for variant in get_unique_variants(figures[fig_index]):
            for i, j in product(range(area.shape[0] - variant.shape[0] + 1),
                                range(area.shape[1] - variant.shape[1] + 1)):
                if can_place(area, variant, i, j):
                    place_figure(area, variant, i, j, fig_index + 1)
                    if solve(fig_index + 1):
                        return True
                    place_figure(area, variant, i, j, 0)
        return False

    solve(0)


# Ввод корректных дня и месяца
while True:
    try:
        day = int(input('Введите день (1-31): '))
        month = int(input('Введите месяц (1-12): '))
        if not (1 <= day <= 31) or not (1 <= month <= 12):
            raise ValueError
        break
    except ValueError:
        print("Пожалуйста, введите корректные значения для дня и месяца.")


# Формируем поле и размещаем даты
area = create_area()
day_i, day_j, month_i, month_j = format_region(day, month)

area[day_i][day_j] = 9
area[month_i][month_j] = 9

# Определение фигур
figures = [
    np.array([[1, 0, 1], [1, 1, 1]]),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0]]),
    np.array([[1, 1, 1], [1, 1, 1]]),
    np.array([[1, 1, 0], [1, 1, 1]]),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1]]),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1]]),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
]

# Размещение фигур
backtrack(area, figures)

# Определение цветовой палитры
cmap = {
    -1: [0, 0, 0],
    0: [0.1, 0.1, 0.1],
    1: [0.2, 0.2, 0.2],
    2: [0.3, 0.3, 0.3],
    3: [0.4, 0.4, 0.4],
    4: [0.5, 0.5, 0.5],
    5: [0.6, 0.6, 0.6],
    6: [0.7, 0.7, 0.7],
    7: [0.8, 0.8, 0.8],
    8: [0.9, 0.9, 0.9],     
    9: [1, 0, 0]
}


# Создаем массив цветов на основе cmap
color_data = np.array([[cmap[value] for value in row] for row in area])

# Отображаем изображение
plt.imshow(color_data, interpolation="nearest")
plt.axis("off")
plt.show()
