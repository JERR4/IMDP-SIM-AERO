"""
Простой тест для проверки работы модели.
Запускает один небольшой эксперимент.
"""

from src.model import AssemblyLineModel
from src.scenarios import get_scenario_1, get_batch_config

def test_small_batch():
    """Тестирует модель с небольшой партией."""
    print("Тестирование модели с партией из 5 самолетов...")
    
    # Создаем модель
    model = AssemblyLineModel(
        buffer_capacity=1,
        station_capacity=1,
        use_flexible_workforce=False,
        seed=42
    )
    
    # Запускаем небольшую партию
    batch_config = {
        "batch_size": 5,
        "aircraft_distribution": {"SA101": 0.6, "SA102": 0.4},
        "cycle_time": 96,  # 4 дня
        "use_cycle_time": False  # Без фиксированного времени цикла для ускорения
    }
    
    model.env.process(
        model.run_batch(
            batch_size=batch_config["batch_size"],
            aircraft_distribution=batch_config["aircraft_distribution"],
            cycle_time=batch_config["cycle_time"],
            use_cycle_time=batch_config["use_cycle_time"]
        )
    )
    
    # Запускаем моделирование
    print("Запуск моделирования...")
    model.run()
    
    # Проверяем результаты jerry
    print(f"\nРезультаты:")
    print(f"  Время завершения партии: {model.stats.batch_completion_time:.2f} часов")
    print(f"  Количество самолетов: {len(model.aircraft_list)}")
    print(f"  Завершенных самолетов: {sum(1 for a in model.aircraft_list if a.exit_time is not None)}")
    
    # Вычисляем среднее время цикла
    if model.stats.cycle_times:
        avg_cycle_time = sum(model.stats.cycle_times) / len(model.stats.cycle_times)
        print(f"  Среднее время цикла: {avg_cycle_time:.2f} часов")
        print(f"  Количество измерений времени цикла: {len(model.stats.cycle_times)}")
    else:
        print(f"  Среднее время цикла: N/A (нет данных)")
    
    print(f"  Средний уровень НЗП: {model.stats.avg_wip:.2f}")
    
    print("\nЗагрузка участков:")
    for station_id in range(1, 6):
        stats = model.stats.station_stats[station_id]
        print(f"  Участок {station_id}: {stats.utilization:.2%} (обработано самолетов: {stats.aircraft_processed})")
    
    print("\n✓ Тест завершен успешно!")

if __name__ == "__main__":
    try:
        test_small_batch()
    except Exception as e:
        import traceback
        print(f"\n✗ Ошибка при тестировании:")
        print(traceback.format_exc())

