# src/streamlit_app.py

import os
import pickle
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


RESULTS_DIR = "results"
ALL_RESULTS_PATH = os.path.join(RESULTS_DIR, "/Users/jerry/BMSTU/ОПЖЦАСОИУ/2 sem/курсач/IMDP-SIM-AERO/results/all_results.pkl")


# ---------- Загрузка данных ----------

@st.cache_data
def load_all_results(path: str = ALL_RESULTS_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        st.error(f"Файл с результатами не найден: {path}")
        return []

    with open(path, "rb") as f:
        results = pickle.load(f)

    # На всякий случай приведём к списку
    if isinstance(results, dict):
        results = [results]

    return results


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Упрощённая таблица для фильтрации в интерфейсе."""
    rows = []
    for r in results:
        rows.append(
            {
                "scenario": r["scenario_name"],
                "batch_size": r["batch_config"]["batch_size"],
                "use_cycle_time": r["batch_config"]["use_cycle_time"],
                "replication": r["replication"],
            }
        )
    return pd.DataFrame(rows)


# ---------- Графики ----------

def prepare_gantt_data(exp: Dict[str, Any]) -> pd.DataFrame:
    """
    Делаем таблицу вида:
    [aircraft_id, station, start_time_hours, end_time_hours]
    для построения Gantt.
    """
    records = []

    for a in exp["aircraft_data"]:
        a_id = f"A{a['id']}"
        for station_str, times in a["station_times"].items():
            station_id = int(station_str)
            start = times["start"]
            end = times["end"]
            if start is None or end is None:
                continue

            records.append(
                {
                    "aircraft": a_id,
                    "station": f"Участок {station_id}",
                    "start_h": start,
                    "end_h": end,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Для plotly-timeline надо в datetime, иначе он рисует чушь по оси
    base = pd.Timestamp("1970-01-01")
    df["start_dt"] = base + pd.to_timedelta(df["start_h"], unit="h")
    df["end_dt"] = base + pd.to_timedelta(df["end_h"], unit="h")

    return df


def plot_gantt(exp: Dict[str, Any]):
    df = prepare_gantt_data(exp)
    st.subheader("Занятость участков во времени (Gantt)")

    if df.empty:
        st.info("Нет детальных данных по временам операций для выбранного эксперимента.")
        return

    fig = px.timeline(
        df,
        x_start="start_dt",
        x_end="end_dt",
        y="station",
        color="aircraft",
        hover_data={"start_h": True, "end_h": True},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=500,
        xaxis_title="Время (часы)",
        yaxis_title="Участок",
        legend_title="Самолёт",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_wip_curve(exp: Dict[str, Any]):
    st.subheader("Динамика WIP")

    wip_raw = exp.get("wip_levels", [])
    if not wip_raw:
        st.info("Нет данных WIP для выбранного эксперимента.")
        return

    # wip_levels = [(time, wip), ...]
    wip_raw = sorted(wip_raw, key=lambda x: x[0])
    times = [t for t, _ in wip_raw]
    levels = [w for _, w in wip_raw]

    if len(times) == 1:
        # один шаг – просто линия
        fig = px.line(x=times, y=levels)
    else:
        # строим ступенчатый график вручную:
        step_t = []
        step_y = []
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            w0 = levels[i]
            step_t.extend([t0, t1])
            step_y.extend([w0, w0])
        # добавим последнюю точку
        step_t.append(times[-1])
        step_y.append(levels[-1])

        fig = px.line(
            x=step_t,
            y=step_y,
        )

    fig.update_traces(mode="lines")
    fig.update_layout(
        xaxis_title="Время (часы)",
        yaxis_title="Количество самолётов",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_cycle_time_hist(exp: Dict[str, Any]):
    st.subheader("Распределение индивидуальных времён цикла самолётов")

    # считаем время цикла = exit_time - entry_time для каждого самолёта
    cycle_times = []
    for a in exp["aircraft_data"]:
        entry = a.get("entry_time")
        exit_t = a.get("exit_time")
        if entry is None or exit_t is None:
            continue
        cycle = exit_t - entry
        if cycle >= 0:
            cycle_times.append(cycle)

    if not cycle_times:
        st.info("Нет данных по времени цикла для выбранного эксперимента.")
        return

    ct_series = pd.Series(cycle_times)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Число самолётов", len(ct_series))
    with col2:
        st.metric("Среднее время цикла (ч)", f"{ct_series.mean():.2f}")
    with col3:
        st.metric("Ст. отклонение (ч)", f"{ct_series.std():.2f}")

    fig = px.histogram(
        x=cycle_times,
        nbins=20,
        labels={"x": "Время цикла (часы)", "y": "Количество самолётов"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ---------- UI ----------

def main():
    st.set_page_config(
        page_title="Имитационная модель авиазавода",
        layout="wide",
    )
    st.title("Анализ имитационных экспериментов")

    results = load_all_results()
    if not results:
        st.stop()

    df_index = results_to_dataframe(results)

    st.sidebar.header("Фильтр экспериментов")

    scenario_list = sorted(df_index["scenario"].unique())
    scenario = st.sidebar.selectbox("Сценарий", scenario_list)

    batch_list = sorted(df_index["batch_size"].unique())
    batch_size = st.sidebar.selectbox("Размер партии", batch_list)

    use_cycle_list = sorted(df_index["use_cycle_time"].unique())
    use_cycle_str_map = {True: "С фиксированным временем цикла", False: "Без фиксированного времени цикла"}
    use_cycle_choice = st.sidebar.selectbox(
        "Режим запуска",
        [use_cycle_str_map[v] for v in use_cycle_list],
    )
    # обратно в bool
    inv_map = {v: k for k, v in use_cycle_str_map.items()}
    use_cycle_time = inv_map[use_cycle_choice]

    subset = df_index[
        (df_index["scenario"] == scenario)
        & (df_index["batch_size"] == batch_size)
        & (df_index["use_cycle_time"] == use_cycle_time)
    ]

    if subset.empty:
        st.error("Нет экспериментов с такими параметрами.")
        st.stop()

    rep_list = sorted(subset["replication"].unique())
    replication = st.sidebar.selectbox("Репликация", rep_list)

    # находим нужный эксперимент
    exp = None
    for r in results:
        if (
            r["scenario_name"] == scenario
            and r["batch_config"]["batch_size"] == batch_size
            and r["batch_config"]["use_cycle_time"] == use_cycle_time
            and r["replication"] == replication
        ):
            exp = r
            break

    if exp is None:
        st.error("Не удалось найти данные выбранного эксперимента.")
        st.stop()

    st.markdown(
        f"**Сценарий:** {scenario}  \n"
        f"**Партия:** {batch_size} самолётов  \n"
        f"**Режим запуска:** {use_cycle_str_map[use_cycle_time]}  \n"
        f"**Репликация:** {replication}"
    )

    tab_gantt, tab_wip, tab_cycle = st.tabs(
        ["Gantt по участкам", "WIP(t)", "Распределение времён цикла"]
    )

    with tab_gantt:
        plot_gantt(exp)

    with tab_wip:
        plot_wip_curve(exp)

    with tab_cycle:
        plot_cycle_time_hist(exp)


if __name__ == "__main__":
    main()
