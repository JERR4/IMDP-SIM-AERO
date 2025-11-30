"""
Определение сценариев экспериментов.
"""

from typing import Dict, Any
from .data import BATCH_75, BATCH_100


def get_scenario_1() -> Dict[str, Any]:
    """
    Сценарий 1: Буферы между участками отсутствуют (ёмкость 0).
    Каждый участок может обрабатывать 1 самолет одновременно.
    """
    return {
        "name": "Сценарий 1: Буферы отсутствуют",
        "description": "Между рабочими участками нет места для самолетов (нулевая емкость буферов)",
        "buffer_capacity": 0,
        "station_capacity": 1,
        "use_flexible_workforce": False,
    }


def get_scenario_2() -> Dict[str, Any]:
    """
    Сценарий 2: Буфер между участками - 1 самолет.
    Каждый участок может обрабатывать 1 самолет одновременно.
    """
    return {
        "name": "Сценарий 2: Буфер 1 самолет",
        "description": "Между рабочими участками есть место для одного самолета",
        "buffer_capacity": 1,
        "station_capacity": 1,
        "use_flexible_workforce": False,
    }


def get_scenario_3() -> Dict[str, Any]:
    """
    Сценарий 3: Буфер - неограниченный (условно большая capacity).
    Каждый участок может обрабатывать 1 самолет одновременно.
    """
    return {
        "name": "Сценарий 3: Неограниченный буфер",
        "description": "Между рабочими участками есть место для неограниченного количества самолетов",
        "buffer_capacity": 1000,  # Большое число для имитации неограниченного буфера
        "station_capacity": 1,
        "use_flexible_workforce": False,
    }


def get_scenario_4() -> Dict[str, Any]:
    """
    Сценарий 4: Каждый участок может обрабатывать 2 самолета одновременно.
    Буферы между участками отсутствуют.
    """
    return {
        "name": "Сценарий 4: Capacity 2, буферы отсутствуют",
        "description": "Каждый рабочий участок может работать над двумя самолетами одновременно, буферы отсутствуют",
        "buffer_capacity": 0,
        "station_capacity": 2,
        "use_flexible_workforce": False,
    }


def get_scenario_5() -> Dict[str, Any]:
    """
    Сценарий 5: Единый пул рабочих (гибкая рабочая сила).
    Буферы отсутствуют, capacity = 1.
    """
    return {
        "name": "Сценарий 5: Гибкая рабочая сила",
        "description": "Единый коллектив из 40 работников, где любой работник может выполнять любую задачу",
        "buffer_capacity": 0,
        "station_capacity": 1,
        "use_flexible_workforce": True,
    }


def get_all_scenarios() -> list[Dict[str, Any]]:
    """Возвращает список всех сценариев."""
    return [
        get_scenario_1(),
        get_scenario_2(),
        get_scenario_3(),
        get_scenario_4(),
        get_scenario_5(),
    ]


def get_batch_config(batch_type: str = "75") -> Dict[str, Any]:
    """
    Возвращает конфигурацию партии.
    
    Параметры:
    - batch_type: "75" для партии из 75 самолетов, "100" для партии из 100 самолетов
    """
    if batch_type == "75":
        return {
            "batch_size": BATCH_75["total"],
            "aircraft_distribution": BATCH_75["distribution"],
            "cycle_time": BATCH_75["cycle_time"],
            "use_cycle_time": True,  # Используем фиксированное время цикла
        }
    elif batch_type == "100":
        return {
            "batch_size": BATCH_100["total"],
            "aircraft_distribution": BATCH_100["distribution"],
            "cycle_time": BATCH_100["cycle_time"],
            "use_cycle_time": True,
        }
    else:
        raise ValueError(f"Неизвестный тип партии: {batch_type}")


def get_batch_config_no_cycle(batch_type: str = "75") -> Dict[str, Any]:
    """
    Возвращает конфигурацию партии БЕЗ использования фиксированного времени цикла.
    Самолеты перемещаются только после завершения работы на участках.
    """
    if batch_type == "75":
        return {
            "batch_size": BATCH_75["total"],
            "aircraft_distribution": BATCH_75["distribution"],
            "cycle_time": 0,  # Не используется
            "use_cycle_time": False,  # Не используем фиксированное время цикла
        }
    elif batch_type == "100":
        return {
            "batch_size": BATCH_100["total"],
            "aircraft_distribution": BATCH_100["distribution"],
            "cycle_time": 0,  # Не используется
            "use_cycle_time": False,
        }
    else:
        raise ValueError(f"Неизвестный тип партии: {batch_type}")

