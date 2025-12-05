"""
Имитационная модель поточной линии окончательной сборки авиалайнеров.
"""

import simpy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import random

from .data import (
    OPERATIONS, get_operations_by_station, get_operations_by_sequence,
    NUM_STATIONS, MECHANICS_PER_STATION, WORK_HOURS_PER_DAY, WORK_DAYS_PER_WEEK,
    AIRCRAFT_TYPES
)


@dataclass
class Aircraft:
    """Представляет самолет в системе."""
    id: int
    aircraft_type: str
    entry_time: float = 0.0
    exit_time: Optional[float] = None
    station_times: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # {station_id: (start, end)}
    operations_completed: List[str] = field(default_factory=list)


@dataclass
class StationStats:
    """Статистика по рабочему участку."""
    utilization: float = 0.0  # Загрузка (0-1)
    blocking_time: float = 0.0  # Время блокировки
    idle_time: float = 0.0  # Время простоя
    total_processing_time: float = 0.0
    aircraft_processed: int = 0


@dataclass
class ModelStats:
    """Общая статистика модели."""
    cycle_times: List[float] = field(default_factory=list)  # Время между завершениями самолетов
    batch_completion_time: float = 0.0
    wip_levels: List[Tuple[float, int]] = field(default_factory=list)  # [(time, wip_count)]
    station_stats: Dict[int, StationStats] = field(default_factory=lambda: {
        i: StationStats() for i in range(1, NUM_STATIONS + 1)
    })
    total_mechanics_utilization: float = 0.0
    avg_wip: float = 0.0


class AssemblyLineModel:
    """
    Имитационная модель сборочной линии.
    
    Параметры:
    - buffer_capacity: емкость буферов между участками (0, 1, или большое число для неограниченного)
    - station_capacity: количество самолетов, которые могут обрабатываться одновременно на каждом участке (1 или 2)
    - use_flexible_workforce: если True, используется единый пул рабочих вместо отдельных по участкам
    """
    
    def __init__(
        self,
        buffer_capacity: int = 0,
        station_capacity: int = 1,
        use_flexible_workforce: bool = False,
        seed: Optional[int] = None
    ):
        self.env = simpy.Environment()
        self.buffer_capacity = buffer_capacity if buffer_capacity > 0 else 0
        self.station_capacity = station_capacity
        self.use_flexible_workforce = use_flexible_workforce
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        # Ресурсы: станции (участки)
        self.stations = {}
        for i in range(1, NUM_STATIONS + 1):
            self.stations[i] = simpy.Resource(self.env, capacity=station_capacity)
        
        # Буферы между станциями
        self.buffers = {}
        for i in range(1, NUM_STATIONS):
            if self.buffer_capacity > 0:
                self.buffers[i] = simpy.Store(self.env, capacity=self.buffer_capacity)
            else:
                # Буфер с нулевой емкостью - самолет не может покинуть станцию, пока следующая не освободится
                self.buffers[i] = None
        
        # Рабочие ресурсы
        if use_flexible_workforce:
            # Единый пул рабочих
            total_mechanics = NUM_STATIONS * MECHANICS_PER_STATION
            self.mechanics_pool = simpy.Resource(self.env, capacity=total_mechanics)
            self.station_mechanics = None
        else:
            # Отдельные пулы по участкам
            self.mechanics_pool = None
            self.station_mechanics = {}
            for i in range(1, NUM_STATIONS + 1):
                self.station_mechanics[i] = simpy.Resource(self.env, capacity=MECHANICS_PER_STATION)
        
        # Статистика
        self.stats = ModelStats()
        self.aircraft_list: List[Aircraft] = []
        self.current_wip = 0
        self.last_completion_time = None
        
        # Отслеживание времени работы станций
        self.station_busy_start: Dict[int, float] = {}
        self.station_idle_start: Dict[int, float] = {}
        for i in range(1, NUM_STATIONS + 1):
            self.station_idle_start[i] = 0.0
    
    def is_work_time(self, current_time: float) -> bool:
        """
        Проверяет, является ли текущее время рабочим.
        Упрощенная версия: считаем, что работа идет непрерывно в рамках моделирования.
        """
        return True
    
    def get_mechanics_resource(self, station_id: int) -> simpy.Resource:
        """Возвращает ресурс механиков для указанного участка."""
        if self.use_flexible_workforce:
            return self.mechanics_pool
        else:
            return self.station_mechanics[station_id]
    
    def aircraft_process(self, aircraft: Aircraft):
        """
        Процесс обработки самолета на сборочной линии.
        Самолет проходит через все участки последовательно.
        """
        self.current_wip += 1
        self.stats.wip_levels.append((self.env.now, self.current_wip))
        
        for station_id in range(1, NUM_STATIONS + 1):
            # Если есть буфер перед станцией, забираем самолет из буфера
            if station_id > 1 and self.buffers[station_id - 1] is not None:
                yield self.buffers[station_id - 1].get()
            
            # Ожидание доступа к станции
            with self.stations[station_id].request() as station_request:
                yield station_request
                
                # Начало обработки на станции
                start_time = self.env.now
                aircraft.station_times[station_id] = (start_time, None)
                
                # Обновление статистики станции
                if station_id in self.station_idle_start:
                    idle_duration = self.env.now - self.station_idle_start[station_id]
                    self.stats.station_stats[station_id].idle_time += idle_duration
                    del self.station_idle_start[station_id]
                self.station_busy_start[station_id] = self.env.now
                
                # Выполнение операций на станции
                yield from self.process_station_operations(aircraft, station_id)
                
                # Завершение обработки на станции
                # Завершение обработки на станции (конец чистой обработки)
                end_time = self.env.now
                aircraft.station_times[station_id] = (start_time, end_time)

                # Обновление статистики по обработке
                processing_duration = end_time - start_time
                station_stats = self.stats.station_stats[station_id]
                station_stats.total_processing_time += processing_duration
                station_stats.aircraft_processed += 1

                # Перемещение к следующей станции / возможная блокировка
                if station_id < NUM_STATIONS:
                    block_start = self.env.now  # начало возможной блокировки (после обработки)

                    if self.buffers[station_id] is not None:
                        # Есть буфер — ждем свободное место в буфере
                        yield self.buffers[station_id].put(aircraft)
                        block_end = self.env.now
                        blocking_duration = block_end - block_start
                        station_stats.blocking_time += blocking_duration
                    else:
                        # Буфера нет — ждем освобождения следующей станции, удерживая текущую
                        next_station_id = station_id + 1
                        with self.stations[next_station_id].request() as next_request:
                            yield next_request
                            block_end = self.env.now
                            blocking_duration = block_end - block_start
                            station_stats.blocking_time += blocking_duration
                        # Здесь ресурс текущей станции освободится при выходе из внешнего with

                    # После блокировки станция уходит в простой
                    if station_id in self.station_busy_start:
                        del self.station_busy_start[station_id]
                    self.station_idle_start[station_id] = self.env.now
                else:
                    # Последняя станция — после обработки сразу простаивает
                    if station_id in self.station_busy_start:
                        del self.station_busy_start[station_id]
                    self.station_idle_start[station_id] = self.env.now

        # Самолет завершил обработку
        aircraft.exit_time = self.env.now
        self.current_wip -= 1
        self.stats.wip_levels.append((self.env.now, self.current_wip))
        
        # Вычисление времени цикла
        if self.last_completion_time is not None:
            cycle_time = self.env.now - self.last_completion_time
            self.stats.cycle_times.append(cycle_time)
        self.last_completion_time = self.env.now
    
    def process_station_operations(self, aircraft: Aircraft, station_id: int):
        """
        Выполняет все операции на указанном участке для самолета.
        Учитывает последовательности (Sequence) и предшественников.
        """
        operations = get_operations_by_station(station_id)
        if not operations:
            return
        
        # Группируем операции по последовательностям
        max_sequence = max(op.sequence for op in operations)
        completed_ops = set()
        
        for sequence_num in range(1, max_sequence + 1):
            # Получаем операции текущей последовательности
            seq_operations = get_operations_by_sequence(station_id, sequence_num)
            
            # Проверяем, что все предшественники выполнены
            ready_ops = []
            for op in seq_operations:
                if all(pred in completed_ops for pred in op.predecessors):
                    ready_ops.append(op)
            
            if not ready_ops:
                continue
            
            # Выполняем операции последовательности параллельно
            # Запускаем процессы для всех операций одновременно
            if ready_ops:
                mechanics_resource = self.get_mechanics_resource(station_id)
                
                # Создаем процессы для всех операций
                processes = []
                for op in ready_ops:
                    processing_time = op.processing_time_by_type.get(aircraft.aircraft_type, 0.0)
                    if processing_time > 0:
                        # Запускаем процесс выполнения операции
                        process = self.env.process(
                            self._execute_operation(
                                op, aircraft, processing_time, mechanics_resource, station_id
                            )
                        )
                        processes.append(process)
                
                # Ждем завершения всех операций последовательности
                for process in processes:
                    yield process
                
                # Отмечаем операции как выполненные
                for op in ready_ops:
                    if op.name not in completed_ops:
                        completed_ops.add(op.name)
                        aircraft.operations_completed.append(op.name)
    
    def _execute_operation(
        self, op, aircraft: Aircraft, processing_time: float,
        mechanics_resource: simpy.Resource, station_id: int
    ):
        """Выполняет одну операцию с запросом необходимого количества механиков."""
        mechanics_needed = min(op.mechanics_required, mechanics_resource.capacity)
        
        # Запрашиваем необходимое количество механиков
        requests = []
        for _ in range(mechanics_needed):
            req = mechanics_resource.request()
            requests.append(req)
        
        # Ждем получения всех запросов
        for req in requests:
            yield req
        
        try:
            # Выполняем операцию
            yield self.env.timeout(processing_time)
        finally:
            # Освобождаем механиков (в SimPy используется resource.release(request))
            for req in requests:
                mechanics_resource.release(req)
    
    def run_batch(
        self,
        batch_size: int,
        aircraft_distribution: Dict[str, float],
        cycle_time: float,
        use_cycle_time: bool = True
    ):
        """
        Запускает партию самолетов.
        
        Параметры:
        - batch_size: количество самолетов в партии
        - aircraft_distribution: распределение типов самолетов (например, {"SA101": 0.6, "SA102": 0.4})
        - cycle_time: время цикла (в часах)
        - use_cycle_time: если True, самолеты запускаются с интервалом cycle_time
        """
        # Генерируем последовательность типов самолетов
        aircraft_types_list = []
        for _ in range(batch_size):
            rand = random.random()
            cumsum = 0.0
            for atype, prob in aircraft_distribution.items():
                cumsum += prob
                if rand <= cumsum:
                    aircraft_types_list.append(atype)
                    break
        
        # Запускаем самолеты
        for i, aircraft_type in enumerate(aircraft_types_list):
            aircraft = Aircraft(id=i + 1, aircraft_type=aircraft_type)
            aircraft.entry_time = self.env.now
            self.aircraft_list.append(aircraft)
            
            # Запускаем процесс обработки самолета
            self.env.process(self.aircraft_process(aircraft))
            
            # Если используется время цикла, ждем перед запуском следующего
            if use_cycle_time and i < batch_size - 1:
                yield self.env.timeout(cycle_time)
        
        # Ждем завершения всех самолетов
        # Проверяем, что все самолеты завершили обработку
        max_wait_time = 365 * 24 * 10  # Максимальное время ожидания: 10 лет
        wait_start = self.env.now
        while any(a.exit_time is None for a in self.aircraft_list):
            if self.env.now - wait_start > max_wait_time:
                # Таймаут - не все самолеты завершились
                incomplete = [a.id for a in self.aircraft_list if a.exit_time is None]
                print(f"Предупреждение: Не все самолеты завершились. Незавершенные: {incomplete}")
                break
            yield self.env.timeout(1.0)  # Проверяем каждую единицу времени
        
        # Финализируем статистику
        self.finalize_stats()
    
    def finalize_stats(self):
        """Финализирует статистику после завершения моделирования."""
        # Время завершения партии
        if self.aircraft_list:
            self.stats.batch_completion_time = max(
                a.exit_time for a in self.aircraft_list if a.exit_time is not None
            )
        
        # Загрузка станций
        total_time = self.stats.batch_completion_time
        for station_id in range(1, NUM_STATIONS + 1):
            stats = self.stats.station_stats[station_id]
            if total_time > 0:
                busy_time = stats.total_processing_time + stats.blocking_time
                stats.utilization = busy_time / (total_time * self.station_capacity)
            else:
                stats.utilization = 0.0

        # Средний уровень НЗП
        if self.stats.wip_levels:
            total_wip_time = sum(
                (self.stats.wip_levels[i+1][0] - self.stats.wip_levels[i][0]) * self.stats.wip_levels[i][1]
                for i in range(len(self.stats.wip_levels) - 1)
            )
            if total_time > 0:
                self.stats.avg_wip = total_wip_time / total_time
            else:
                self.stats.avg_wip = 0.0
        else:
            self.stats.avg_wip = 0.0
    
    def run(self, until: Optional[float] = None):
        """Запускает моделирование."""
        if until is None:
            until = 365 * 24 * 100
        self.env.run(until=until)

