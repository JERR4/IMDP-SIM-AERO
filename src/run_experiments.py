"""
Запуск серий экспериментов и сохранение результатов.
"""

import os
import json
import pickle
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from .model import AssemblyLineModel
from .scenarios import get_all_scenarios, get_batch_config, get_batch_config_no_cycle


def run_single_experiment(
    scenario: Dict[str, Any],
    batch_config: Dict[str, Any],
    replication: int,
    seed: int = None
) -> Dict[str, Any]:
    """
    Запускает один эксперимент.
    
    Параметры:
    - scenario: конфигурация сценария
    - batch_config: конфигурация партии
    - replication: номер репликации
    - seed: seed для генератора случайных чисел
    """
    # Создаем модель
    model = AssemblyLineModel(
        buffer_capacity=scenario["buffer_capacity"],
        station_capacity=scenario["station_capacity"],
        use_flexible_workforce=scenario["use_flexible_workforce"],
        seed=seed if seed is not None else replication * 1000
    )
    
    # Запускаем партию
    model.env.process(
        model.run_batch(
            batch_size=batch_config["batch_size"],
            aircraft_distribution=batch_config["aircraft_distribution"],
            cycle_time=batch_config["cycle_time"],
            use_cycle_time=batch_config["use_cycle_time"]
        )
    )
    
    # Запускаем моделирование
    model.run()
    
    # Собираем результаты
    results = {
        "scenario_name": scenario["name"],
        "scenario_config": scenario,
        "batch_config": batch_config,
        "replication": replication,
        "seed": seed if seed is not None else replication * 1000,
        "batch_completion_time": model.stats.batch_completion_time,
        "avg_cycle_time": (
            sum(model.stats.cycle_times) / len(model.stats.cycle_times)
            if model.stats.cycle_times else 0.0
        ),
        "cycle_times": model.stats.cycle_times,
        "avg_wip": getattr(model.stats, "avg_wip", 0.0),
        "station_stats": {
            station_id: {
                "utilization": stats.utilization,
                "blocking_time": stats.blocking_time,
                "idle_time": stats.idle_time,
                "total_processing_time": stats.total_processing_time,
                "aircraft_processed": stats.aircraft_processed,
            }
            for station_id, stats in model.stats.station_stats.items()
        },
        "aircraft_data": [
            {
                "id": a.id,
                "type": a.aircraft_type,
                "entry_time": a.entry_time,
                "exit_time": a.exit_time,
                "total_time": a.exit_time - a.entry_time if a.exit_time else None,
                "station_times": {
                    str(k): {"start": v[0], "end": v[1]}
                    for k, v in a.station_times.items()
                }
            }
            for a in model.aircraft_list
        ],
        "wip_levels": model.stats.wip_levels,
    }
    
    return results


def run_experiment_series(
    scenarios: List[Dict[str, Any]],
    batch_configs: List[Dict[str, Any]],
    num_replications: int = 5,
    results_dir: str = "results"
) -> List[Dict[str, Any]]:
    """
    Запускает серию экспериментов.
    
    Параметры:
    - scenarios: список сценариев
    - batch_configs: список конфигураций партий
    - num_replications: количество репликаций для каждого эксперимента
    - results_dir: директория для сохранения результатов
    """
    # Создаем директорию для результатов
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for scenario in scenarios:
        for batch_config in batch_configs:
            scenario_name_clean = scenario["name"].replace(" ", "_").replace(":", "")
            batch_size = batch_config["batch_size"]
            
            print(f"\n{'='*60}")
            print(f"Сценарий: {scenario['name']}")
            print(f"Партия: {batch_size} самолетов")
            print(f"Репликаций: {num_replications}")
            print(f"{'='*60}\n")
            
            for replication in range(1, num_replications + 1):
                print(f"  Репликация {replication}/{num_replications}...", end=" ", flush=True)
                
                try:
                    result = run_single_experiment(
                        scenario=scenario,
                        batch_config=batch_config,
                        replication=replication
                    )
                    all_results.append(result)
                    
                    # Сохраняем отдельный файл для каждого эксперимента
                    filename = (
                        f"{scenario_name_clean}_batch{batch_size}_rep{replication}.pkl"
                    )
                    filepath = os.path.join(results_dir, filename)
                    with open(filepath, "wb") as f:
                        pickle.dump(result, f)
                    
                    print(f"✓ (время завершения: {result['batch_completion_time']:.2f} ч)")
                    
                except Exception as e:
                    import traceback
                    print(f"✗ Ошибка: {e}")
                    print(f"Детали: {traceback.format_exc()}")
                    continue
    
    # Сохраняем сводную таблицу результатов
    summary_data = []
    for result in all_results:
        summary_data.append({
            "scenario": result["scenario_name"],
            "batch_size": result["batch_config"]["batch_size"],
            "replication": result["replication"],
            "batch_completion_time": result["batch_completion_time"],
            "avg_cycle_time": result["avg_cycle_time"],
            "avg_wip": result["avg_wip"],
            **{
                f"station_{i}_utilization": result["station_stats"][i]["utilization"]
                for i in range(1, 6)
            }
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nСводная таблица сохранена: {summary_path}")
    
    # Сохраняем все результаты в один файл
    all_results_path = os.path.join(results_dir, "all_results.pkl")
    with open(all_results_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Все результаты сохранены: {all_results_path}")
    
    return all_results


def main():
    """Основная функция для запуска экспериментов."""
    # Получаем все сценарии
    scenarios = get_all_scenarios()
    
    # Конфигурации партий
    batch_configs = [
        get_batch_config("75"),  # Партия 75 самолетов с фиксированным временем цикла
        get_batch_config("100"),  # Партия 100 самолетов с фиксированным временем цикла
        get_batch_config_no_cycle("75"),  # Партия 75 без фиксированного времени цикла
        get_batch_config_no_cycle("100"),  # Партия 100 без фиксированного времени цикла
    ]
    
    # Запускаем эксперименты
    num_replications = 5  # Количество репликаций для каждого эксперимента
    
    results = run_experiment_series(
        scenarios=scenarios,
        batch_configs=batch_configs,
        num_replications=num_replications,
        results_dir="results"
    )
    
    print(f"\n{'='*60}")
    print(f"Всего экспериментов выполнено: {len(results)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

