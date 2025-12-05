"""
Анализ результатов экспериментов и построение графиков.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # рендер без GUI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ------ Общие настройки графиков ------

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Русский язык
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ------ Загрузка данных ------

def load_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    """Загружает сырые результаты экспериментов из all_results.pkl."""
    results_path = os.path.join(results_dir, "all_results.pkl")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Файл результатов не найден: {results_path}")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    return results


def load_summary(results_dir: str = "results") -> pd.DataFrame:
    """Загружает сводную таблицу результатов из summary.csv."""
    summary_path = os.path.join(results_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Файл сводной таблицы не найден: {summary_path}")

    if os.path.getsize(summary_path) == 0:
        raise ValueError(
            f"Файл сводной таблицы пуст: {summary_path}. "
            f"Сначала запустите run_experiments.py"
        )

    try:
        df = pd.read_csv(summary_path, encoding="utf-8-sig")
        if df.empty:
            raise ValueError(
                f"Файл сводной таблицы не содержит данных: {summary_path}. "
                f"Сначала запустите run_experiments.py"
            )
        return df
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Файл сводной таблицы не содержит данных: {summary_path}. "
            f"Сначала запустите run_experiments.py"
        )


# ------ Статистика ------

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет агрегированную статистику по сценариям."""
    df = df[df["use_cycle_time"] == True].copy()

    stats = (
        df.groupby(["scenario", "batch_size", "use_cycle_time"])
        .agg(
            {
                "batch_completion_time": ["mean", "std", "min", "max"],
                "avg_cycle_time": ["mean", "std"],
                "avg_wip": ["mean", "std"],
                **{f"station_{i}_utilization": ["mean", "std"] for i in range(1, 6)},
            }
        )
        .reset_index()
    )

    stats.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in stats.columns.values
    ]
    return stats



# ------ Графики ------

def plot_batch_completion_time(df: pd.DataFrame, output_dir: str = "report/figures"):
    """График времени завершения партии по сценариям."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = df[df["use_cycle_time"] == True].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, batch_size in enumerate([75, 100]):
        ax = axes[idx]
        data = df[df["batch_size"] == batch_size]

        scenario_data = (
            data.groupby("scenario")["batch_completion_time"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean")
        )

        x_pos = np.arange(len(scenario_data))
        ax.bar(
            x_pos,
            scenario_data["mean"],
            yerr=scenario_data["std"],
            capsize=5,
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_xlabel("Сценарий", fontsize=12)
        ax.set_ylabel("Время завершения партии (часы)", fontsize=12)
        ax.set_title(f"Партия {batch_size} самолетов", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            scenario_data["scenario"],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "batch_completion_time.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"График сохранен: {os.path.join(output_dir, 'batch_completion_time.png')}")


def plot_cycle_time(df: pd.DataFrame, output_dir: str = "report/figures"):
    """Boxplot среднего времени цикла по сценариям."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = df[df["use_cycle_time"] == True].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, batch_size in enumerate([75, 100]):
        ax = axes[idx]
        data = df[df["batch_size"] == batch_size]

        scenario_list = data["scenario"].unique()
        cycle_data = [
            data[data["scenario"] == s]["avg_cycle_time"].values
            for s in scenario_list
        ]

        bp = ax.boxplot(cycle_data, tick_labels=scenario_list, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)



        ax.set_xlabel("Сценарий", fontsize=12)
        ax.set_ylabel("Среднее время цикла (часы)", fontsize=12)
        ax.set_title(f"Партия {batch_size} самолетов", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "cycle_time.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"График сохранен: {os.path.join(output_dir, 'cycle_time.png')}")


def plot_station_utilization(df: pd.DataFrame, output_dir: str = "report/figures"):
    """Строит график загрузки станций по сценариям."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    batch_sizes = [75, 100]

    # 2 строки (по партии), 5 столбцов (по участкам)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for batch_idx, batch_size in enumerate(batch_sizes):
        data = df[df["batch_size"] == batch_size]

        for station_id in range(1, 6):
            ax = axes[batch_idx, station_id - 1]

            station_col = f"station_{station_id}_utilization"
            if station_col not in data.columns:
                ax.set_visible(False)
                continue

            scenario_means = (
                data.groupby("scenario")[station_col]
                .mean()
                .reset_index()
                .sort_values(station_col)
            )

            x_pos = np.arange(len(scenario_means))
            ax.bar(x_pos, scenario_means[station_col], alpha=0.7, edgecolor="black")

            ax.set_xlabel("Сценарий", fontsize=9)
            ax.set_ylabel("Загрузка", fontsize=9)
            ax.set_title(f"Участок {station_id}, партия {batch_size}",
                         fontsize=10, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                scenario_means["scenario"],
                rotation=45,
                ha="right",
                fontsize=8,
            )

            # ВАЖНО: динамический предел по Y
            max_util = scenario_means[station_col].max()
            ax.set_ylim(0, max_util * 1.1)   # небольшой запас сверху

            ax.grid(axis="y", alpha=0.3)
            ax.axhline(y=1.0, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "station_utilization.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {os.path.join(output_dir, 'station_utilization.png')}")


def plot_wip_levels(df: pd.DataFrame, output_dir: str = "report/figures"):
    """График среднего уровня НЗП по сценариям."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = df[df["use_cycle_time"] == True].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, batch_size in enumerate([75, 100]):
        ax = axes[idx]
        data = df[df["batch_size"] == batch_size]

        scenario_data = (
            data.groupby("scenario")["avg_wip"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean")
        )

        x_pos = np.arange(len(scenario_data))
        ax.bar(
            x_pos,
            scenario_data["mean"],
            yerr=scenario_data["std"],
            capsize=5,
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_xlabel("Сценарий", fontsize=12)
        ax.set_ylabel("Средний уровень НЗП (самолетов)", fontsize=12)
        ax.set_title(f"Партия {batch_size} самолетов", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            scenario_data["scenario"],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "wip_levels.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"График сохранен: {os.path.join(output_dir, 'wip_levels.png')}")


def plot_comparison_matrix(df: pd.DataFrame, output_dir: str = "report/figures"):
    """Матрица сравнения ключевых метрик по сценариям."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = df[df["use_cycle_time"] == True].copy()
    metrics = ["batch_completion_time", "avg_cycle_time", "avg_wip"]
    batch_sizes = [75, 100]

    fig, axes = plt.subplots(len(metrics), len(batch_sizes), figsize=(14, 10))

    for metric_idx, metric in enumerate(metrics):
        for batch_idx, batch_size in enumerate(batch_sizes):
            ax = axes[metric_idx, batch_idx]
            data = df[df["batch_size"] == batch_size]

            scenario_means = (
                data.groupby("scenario")[metric]
                .mean()
                .reset_index()
                .sort_values(metric)
            )

            x_pos = np.arange(len(scenario_means))
            ax.bar(x_pos, scenario_means[metric], alpha=0.7, edgecolor="black")

            ax.set_xlabel("Сценарий", fontsize=10)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
            ax.set_title(
                f"{metric.replace('_', ' ').title()}, партия {batch_size}",
                fontsize=11,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                scenario_means["scenario"],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "comparison_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"График сохранен: {os.path.join(output_dir, 'comparison_matrix.png')}")


# ------ Таблица для отчёта ------

def generate_report_table(df: pd.DataFrame, output_dir: str = "report") -> pd.DataFrame:
    """Генерирует агрегированную таблицу статистики для отчёта."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stats = calculate_statistics(df)

    stats_path = os.path.join(output_dir, "statistics.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"Таблица статистики сохранена: {stats_path}")

    return stats


# ------ Построение summary из сырых результатов ------

def create_summary_from_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Создаёт DataFrame summary из списка сырых результатов run_experiments."""
    summary_data = []

    for result in results:
        batch_cfg = result["batch_config"]
        summary_data.append(
            {
                "scenario": result["scenario_name"],
                "batch_size": batch_cfg["batch_size"],
                "use_cycle_time": batch_cfg.get("use_cycle_time", True),
                "cycle_time": batch_cfg.get("cycle_time", None),
                "replication": result["replication"],
                "batch_completion_time": result["batch_completion_time"],
                "avg_cycle_time": result["avg_cycle_time"],
                "avg_wip": result["avg_wip"],
                **{
                    f"station_{i}_utilization": result["station_stats"][i][
                        "utilization"
                    ]
                    for i in range(1, 6)
                },
            }
        )

    return pd.DataFrame(summary_data)


# ------ main ------

def main():
    print("Загрузка результатов...")
    df = None

    # Пробуем summary.csv
    try:
        df = load_summary()
        print(f"Загружено {len(df)} записей из summary.csv")
    except (FileNotFoundError, ValueError):
        print("summary.csv отсутствует или пуст. Пытаемся загрузить all_results.pkl...")
        try:
            results = load_results()
            print(f"Загружено {len(results)} результатов из all_results.pkl")
            df = create_summary_from_results(results)
            print(f"Создана сводная таблица с {len(df)} записями")

            summary_path = os.path.join("results", "summary.csv")
            df.to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"Сводная таблица сохранена: {summary_path}")
        except FileNotFoundError:
            print("\nОшибка: Файлы результатов не найдены.")
            print("Для запуска экспериментов выполните:")
            print("  python -m src.run_experiments")
            return

    if df is None or df.empty:
        print("\nОшибка: Нет данных для анализа.")
        print("Для запуска экспериментов выполните:")
        print("  python -m src.run_experiments")
        return

    print("\nВычисление статистики...")
    stats = generate_report_table(df)
    print("\nСтатистика по сценариям:")
    print(stats.to_string())

    print("\nПостроение графиков...")
    plot_batch_completion_time(df)
    plot_cycle_time(df)
    plot_station_utilization(df)
    plot_wip_levels(df)
    plot_comparison_matrix(df)

    print("\nАнализ завершен!")


if __name__ == "__main__":
    main()
