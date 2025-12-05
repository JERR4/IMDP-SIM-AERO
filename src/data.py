"""
Исходные данные по операциям, участкам и типам самолетов.

ПРИМЕЧАНИЕ: Так как Excel-файл с фактическими данными не предоставлен,
все времена выполнения операций и их распределение по участкам заданы
на основе разумных допущений, описанных в комментариях ниже.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

# Типы самолетов
AIRCRAFT_TYPES = ["SA101", "SA102", "SA103"]

# Количество рабочих участков
NUM_STATIONS = 5

# Количество механиков на участок
MECHANICS_PER_STATION = 8

# ===== КЛЮЧЕВАЯ ПРАВКА: время цикла =====
# По расчету критического пути:
# - Для SA101 суммарное время по участку 5 ≈ 16 ч
# - Для SA102 суммарное время по участку 5 ≈ 17.7 ч (бутылочное горлышко)
# Поэтому задаем время цикла на уровне узкого места, чтобы линия работала "на пределе"
# и различия между сценариями (буферы, capacity, гибкая рабочая сила) были заметны.

# Время цикла (в часах)
CYCLE_TIME_75 = 18.0   # партия 75 самолётов
CYCLE_TIME_100 = 18.0  # партия 100 самолётов

# График работы смен (в часах от начала недели)
# Первая смена: 6:00-10:00, 10:30-14:30 (пн-пт)
# Вторая смена: 14:30-18:30, 19:00-22:00 (пн-пт)
SHIFT_SCHEDULE = {
    "shift1": {
        "monday": [(6, 10), (10.5, 14.5)],
        "tuesday": [(6, 10), (10.5, 14.5)],
        "wednesday": [(6, 10), (10.5, 14.5)],
        "thursday": [(6, 10), (10.5, 14.5)],
        "friday": [(6, 10), (10.5, 14.5)],
    },
    "shift2": {
        "monday": [(14.5, 18.5), (19, 22)],
        "tuesday": [(14.5, 18.5), (19, 22)],
        "wednesday": [(14.5, 18.5), (19, 22)],
        "thursday": [(14.5, 18.5), (19, 22)],
        "friday": [(14.5, 18.5), (19, 22)],
    }
}

# Рабочие часы в неделю (для упрощения моделирования)
# Пн-Пт, первая смена: 6:00-10:00, 10:30-14:30 = 8 часов
# Пн-Пт, вторая смена: 14:30-18:30, 19:00-22:00 = 7.5 часов
# Итого: 15.5 часов в день * 5 дней = 77.5 часов в неделю
WORK_HOURS_PER_DAY = 15.5
WORK_DAYS_PER_WEEK = 5


@dataclass
class Operation:
    """Операция на рабочем участке."""
    name: str
    station_id: int  # Номер участка (1-5)
    sequence: int  # Номер последовательности (1, 2, 3, ...)
    predecessors: List[str]  # Список имен операций-предшественников
    processing_time_by_type: Dict[str, float]  # Время выполнения в часах для каждого типа самолета
    mechanics_required: int = 1  # Количество механиков, необходимых для операции


# ДОПУЩЕНИЯ:
# 1. Каждый участок имеет от 5 до 20 операций (как указано в задании)
# 2. Времена выполнения для SA101 и SA102 различаются (SA102 немного сложнее)
# 3. Для SA103 времена совпадают с SA101 (как указано в задании)
# 4. Операции распределены по последовательностям (Sequence) от 1 до 5-7
# 5. Времена выполнения варьируются от 0.5 до 8 часов на операцию
# 6. Некоторые операции требуют больше механиков (2-3 человека)

OPERATIONS: Dict[str, Operation] = {}

# Участок 1: Подготовка корпуса
OPERATIONS.update({
    "OP1_1": Operation(
        name="Установка опорных конструкций",
        station_id=1,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 2.0, "SA102": 2.3, "SA103": 2.0},
        mechanics_required=2
    ),
    "OP1_2": Operation(
        name="Проверка геометрии корпуса",
        station_id=1,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 1.5, "SA102": 1.8, "SA103": 1.5},
        mechanics_required=1
    ),
    "OP1_3": Operation(
        name="Подготовка крепежных элементов",
        station_id=1,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 1.0, "SA102": 1.2, "SA103": 1.0},
        mechanics_required=1
    ),
    "OP1_4": Operation(
        name="Установка внутренних перегородок",
        station_id=1,
        sequence=2,
        predecessors=["OP1_1", "OP1_2"],
        processing_time_by_type={"SA101": 3.0, "SA102": 3.5, "SA103": 3.0},
        mechanics_required=3
    ),
    "OP1_5": Operation(
        name="Монтаж систем крепления",
        station_id=1,
        sequence=2,
        predecessors=["OP1_1", "OP1_3"],
        processing_time_by_type={"SA101": 2.5, "SA102": 2.8, "SA103": 2.5},
        mechanics_required=2
    ),
    "OP1_6": Operation(
        name="Проверка качества сборки",
        station_id=1,
        sequence=3,
        predecessors=["OP1_4", "OP1_5"],
        processing_time_by_type={"SA101": 1.5, "SA102": 1.7, "SA103": 1.5},
        mechanics_required=1
    ),
})

# Участок 2: Установка систем
OPERATIONS.update({
    "OP2_1": Operation(
        name="Прокладка электрических кабелей",
        station_id=2,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 4.0, "SA102": 4.5, "SA103": 4.0},
        mechanics_required=2
    ),
    "OP2_2": Operation(
        name="Установка гидравлических систем",
        station_id=2,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 3.5, "SA102": 4.0, "SA103": 3.5},
        mechanics_required=2
    ),
    "OP2_3": Operation(
        name="Монтаж систем кондиционирования",
        station_id=2,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 3.0, "SA102": 3.3, "SA103": 3.0},
        mechanics_required=2
    ),
    "OP2_4": Operation(
        name="Подключение систем к центральным узлам",
        station_id=2,
        sequence=2,
        predecessors=["OP2_1", "OP2_2", "OP2_3"],
        processing_time_by_type={"SA101": 2.5, "SA102": 2.8, "SA103": 2.5},
        mechanics_required=3
    ),
    "OP2_5": Operation(
        name="Тестирование систем",
        station_id=2,
        sequence=3,
        predecessors=["OP2_4"],
        processing_time_by_type={"SA101": 2.0, "SA102": 2.2, "SA103": 2.0},
        mechanics_required=2
    ),
    "OP2_6": Operation(
        name="Документирование установки",
        station_id=2,
        sequence=4,
        predecessors=["OP2_5"],
        processing_time_by_type={"SA101": 0.5, "SA102": 0.6, "SA103": 0.5},
        mechanics_required=1
    ),
})

# Участок 3: Установка интерьера
OPERATIONS.update({
    "OP3_1": Operation(
        name="Установка пассажирских кресел",
        station_id=3,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 5.0, "SA102": 5.5, "SA103": 5.0},
        mechanics_required=4
    ),
    "OP3_2": Operation(
        name="Монтаж багажных полок",
        station_id=3,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 2.5, "SA102": 2.8, "SA103": 2.5},
        mechanics_required=2
    ),
    "OP3_3": Operation(
        name="Установка освещения салона",
        station_id=3,
        sequence=2,
        predecessors=["OP3_1"],
        processing_time_by_type={"SA101": 2.0, "SA102": 2.2, "SA103": 2.0},
        mechanics_required=2
    ),
    "OP3_4": Operation(
        name="Монтаж кухонного оборудования",
        station_id=3,
        sequence=2,
        predecessors=["OP3_1"],
        processing_time_by_type={"SA101": 3.5, "SA102": 4.0, "SA103": 3.5},
        mechanics_required=3
    ),
    "OP3_5": Operation(
        name="Установка туалетных модулей",
        station_id=3,
        sequence=3,
        predecessors=["OP3_2", "OP3_3"],
        processing_time_by_type={"SA101": 2.5, "SA102": 2.8, "SA103": 2.5},
        mechanics_required=2
    ),
    "OP3_6": Operation(
        name="Финишная отделка интерьера",
        station_id=3,
        sequence=4,
        predecessors=["OP3_4", "OP3_5"],
        processing_time_by_type={"SA101": 3.0, "SA102": 3.3, "SA103": 3.0},
        mechanics_required=2
    ),
})

# Участок 4: Установка двигателей и крыльев
OPERATIONS.update({
    "OP4_1": Operation(
        name="Подготовка точек крепления двигателей",
        station_id=4,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 2.0, "SA102": 2.3, "SA103": 2.0},
        mechanics_required=2
    ),
    "OP4_2": Operation(
        name="Установка двигателей",
        station_id=4,
        sequence=2,
        predecessors=["OP4_1"],
        processing_time_by_type={"SA101": 6.0, "SA102": 6.5, "SA103": 6.0},
        mechanics_required=4
    ),
    "OP4_3": Operation(
        name="Подключение топливных систем",
        station_id=4,
        sequence=3,
        predecessors=["OP4_2"],
        processing_time_by_type={"SA101": 3.0, "SA102": 3.3, "SA103": 3.0},
        mechanics_required=2
    ),
    "OP4_4": Operation(
        name="Установка крыльев",
        station_id=4,
        sequence=2,
        predecessors=["OP4_1"],
        processing_time_by_type={"SA101": 5.0, "SA102": 5.5, "SA103": 5.0},
        mechanics_required=4
    ),
    "OP4_5": Operation(
        name="Подключение систем управления",
        station_id=4,
        sequence=4,
        predecessors=["OP4_3", "OP4_4"],
        processing_time_by_type={"SA101": 2.5, "SA102": 2.8, "SA103": 2.5},
        mechanics_required=2
    ),
    "OP4_6": Operation(
        name="Проверка силовой установки",
        station_id=4,
        sequence=5,
        predecessors=["OP4_5"],
        processing_time_by_type={"SA101": 1.5, "SA102": 1.7, "SA103": 1.5},
        mechanics_required=2
    ),
})

# Участок 5: Финальная сборка и испытания
OPERATIONS.update({
    "OP5_1": Operation(
        name="Установка шасси",
        station_id=5,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 3.0, "SA102": 3.3, "SA103": 3.0},
        mechanics_required=3
    ),
    "OP5_2": Operation(
        name="Монтаж навигационного оборудования",
        station_id=5,
        sequence=1,
        predecessors=[],
        processing_time_by_type={"SA101": 2.5, "SA102": 2.8, "SA103": 2.5},
        mechanics_required=2
    ),
    "OP5_3": Operation(
        name="Установка систем связи",
        station_id=5,
        sequence=2,
        predecessors=["OP5_2"],
        processing_time_by_type={"SA101": 2.0, "SA102": 2.2, "SA103": 2.0},
        mechanics_required=2
    ),
    "OP5_4": Operation(
        name="Покраска и маркировка",
        station_id=5,
        sequence=3,
        predecessors=["OP5_1"],
        processing_time_by_type={"SA101": 4.0, "SA102": 4.5, "SA103": 4.0},
        mechanics_required=2
    ),
    "OP5_5": Operation(
        name="Комплексные испытания",
        station_id=5,
        sequence=4,
        predecessors=["OP5_3", "OP5_4"],
        processing_time_by_type={"SA101": 5.0, "SA102": 5.5, "SA103": 5.0},
        mechanics_required=3
    ),
    "OP5_6": Operation(
        name="Финальная проверка и сертификация",
        station_id=5,
        sequence=5,
        predecessors=["OP5_5"],
        processing_time_by_type={"SA101": 2.0, "SA102": 2.2, "SA103": 2.0},
        mechanics_required=2
    ),
})


def get_operations_by_station(station_id: int) -> List[Operation]:
    """Возвращает список операций для указанного участка."""
    return [op for op in OPERATIONS.values() if op.station_id == station_id]


def get_operations_by_sequence(station_id: int, sequence: int) -> List[Operation]:
    """Возвращает список операций для указанного участка и последовательности."""
    return [
        op for op in OPERATIONS.values()
        if op.station_id == station_id and op.sequence == sequence
    ]


def get_total_processing_time(station_id: int, aircraft_type: str) -> float:
    """
    Вычисляет общее время обработки на участке для данного типа самолета.
    Учитывает параллельное выполнение операций одной последовательности.
    """
    operations = get_operations_by_station(station_id)
    if not operations:
        return 0.0

    # Находим максимальную последовательность
    max_sequence = max(op.sequence for op in operations)

    total_time = 0.0
    for seq in range(1, max_sequence + 1):
        seq_operations = get_operations_by_sequence(station_id, seq)
        if seq_operations:
            # Время последовательности = максимальное время среди параллельных операций
            seq_time = max(
                op.processing_time_by_type.get(aircraft_type, 0.0)
                for op in seq_operations
            )
            total_time += seq_time

    return total_time


# Параметры партий
BATCH_75 = {
    "total": 75,
    "distribution": {"SA101": 0.6, "SA102": 0.4},  # 60% SA101, 40% SA102
    "cycle_time": CYCLE_TIME_75,
}

BATCH_100 = {
    "total": 100,
    "distribution": {"SA101": 0.45, "SA102": 0.30, "SA103": 0.25},  # 45% SA101, 30% SA102, 25% SA103
    "cycle_time": CYCLE_TIME_100,
}
