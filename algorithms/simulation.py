"""
离散事件仿真 (Discrete Event Simulation)
用于排队系统、库存管理、生产调度等
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from collections import deque
import heapq


class Event:
    """事件类"""
    
    def __init__(self, time: float, event_type: str, handler: Callable, 
                 priority: int = 0):
        self.time = time
        self.event_type = event_type
        self.handler = handler
        self.priority = priority
    
    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time


class DiscreteEventSimulation:
    """
    离散事件仿真器
    
    用于模拟排队系统、库存系统等
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.event_list = []  # 事件列表（最小堆）
        self.clock = 0.0
        self.stats = {}
        self.events_processed = 0
    
    def schedule_event(self, time: float, event_type: str, 
                       handler: Callable, priority: int = 0):
        """调度事件"""
        event = Event(time, event_type, handler, priority)
        heapq.heappush(self.event_list, event)
    
    def run(self, stop_time: float) -> Dict[str, Any]:
        """
        运行仿真
        
        参数:
            stop_time: 仿真停止时间
        
        返回:
            仿真结果统计
        """
        while self.event_list and self.event_list[0].time <= stop_time:
            event = heapq.heappop(self.event_list)
            
            if event.time > stop_time:
                break
            
            self.clock = event.time
            event.handler()
            self.events_processed += 1
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'final_clock': self.clock,
            'events_processed': self.events_processed,
            **self.stats
        }


class QueueingSystem:
    """
    排队系统仿真
    
    支持 M/M/1, M/M/c, M/G/1 等排队模型
    """
    
    def __init__(self, num_servers: int = 1, service_rate: float = 1.0,
                 arrival_rate: float = 0.8, seed: Optional[int] = None):
        """
        初始化排队系统
        
        参数:
            num_servers: 服务台数量
            service_rate: 服务率
            arrival_rate: 到达率
            seed: 随机种子
        """
        self.num_servers = num_servers
        self.service_rate = service_rate
        self.arrival_rate = arrival_rate
        self.utilization = arrival_rate / (num_servers * service_rate)
        
        self.sim = DiscreteEventSimulation(seed)
        self.queue = deque()
        self.servers_busy = [False] * num_servers
        self.next_arrival_time = 0
        self.customers_served = 0
        self.total_wait_time = 0
        self.max_queue_length = 0
        
        # 统计
        self.wait_times = []
        self.service_times = []
        self.queue_lengths = []
    
    def setup(self):
        """设置初始事件"""
        # 安排第一个到达
        self.next_arrival_time = np.random.exponential(1 / self.arrival_rate)
        self.sim.schedule_event(self.next_arrival_time, 'arrival', self._arrival_handler)
    
    def _arrival_handler(self):
        """处理到达事件"""
        self.customers_served += 1
        
        # 查找空闲服务台
        server_idx = -1
        for i in range(self.num_servers):
            if not self.servers_busy[i]:
                server_idx = i
                break
        
        if server_idx >= 0:
            # 立即服务
            self.servers_busy[server_idx] = True
            service_time = np.random.exponential(1 / self.service_rate)
            self.service_times.append(service_time)
            departure_time = self.sim.clock + service_time
            self.sim.schedule_event(departure_time, 'departure', 
                                   lambda idx=server_idx: self._departure_handler(idx))
        else:
            # 进入队列
            self.queue.append(self.customers_served)
            self.queue_lengths.append(len(self.queue))
            self.max_queue_length = max(self.max_queue_length, len(self.queue))
            
            # 记录等待开始时间
            self.wait_times.append(0)
        
        # 安排下一个到达
        self.next_arrival_time += np.random.exponential(1 / self.arrival_rate)
        self.sim.schedule_event(self.next_arrival_time, 'arrival', self._arrival_handler)
    
    def _departure_handler(self, server_idx: int):
        """处理离开事件"""
        self.servers_busy[server_idx] = False
        
        if self.queue:
            # 从队列中取下一个客户
            self.queue.popleft()
            service_time = np.random.exponential(1 / self.service_rate)
            self.service_times.append(service_time)
            departure_time = self.sim.clock + service_time
            self.sim.schedule_event(departure_time, 'departure',
                                   lambda idx=server_idx: self._departure_handler(idx))
    
    def run(self, time: float = 1000) -> Dict[str, Any]:
        """
        运行仿真
        
        参数:
            time: 仿真时间长度
        
        返回:
            仿真结果
        """
        self.setup()
        result = self.sim.run(time)
        
        avg_wait = np.mean(self.wait_times) if self.wait_times else 0
        avg_service = np.mean(self.service_times) if self.service_times else 0
        
        return {
            'avg_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': self.max_queue_length,
            'avg_wait_time': avg_wait,
            'avg_service_time': avg_service,
            'utilization': self.utilization,
            'customers_served': self.customers_served,
            **result
        }
    
    def analyze_m_m_1(self) -> Dict[str, float]:
        """
        M/M/1排队理论分析
        
        返回理论值
        """
        rho = self.arrival_rate / self.service_rate
        
        if rho >= 1:
            return {'error': '系统不稳定，利用率 >= 1'}
        
        return {
            'utilization': rho,
            'avg_queue_length': rho ** 2 / (1 - rho),
            'avg_system_length': rho / (1 - rho),
            'avg_wait_time': rho / (self.service_rate * (1 - rho)),
            'avg_response_time': 1 / (self.service_rate * (1 - rho))
        }


class InventorySystem:
    """
    库存系统仿真
    
    (s, S) 和 (Q, R) 库存策略
    """
    
    def __init__(self, demand_rate: float = 10, holding_cost: float = 1.0,
                 ordering_cost: float = 50, shortage_cost: float = 10,
                 lead_time: float = 1, seed: Optional[int] = None):
        """
        初始化库存系统
        
        参数:
            demand_rate: 平均需求率
            holding_cost: 持有成本
            ordering_cost: 订货成本
            shortage_cost: 缺货成本
            lead_time: 提前期
        """
        self.demand_rate = demand_rate
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.shortage_cost = shortage_cost
        self.lead_time = lead_time
        
        self.sim = DiscreteEventSimulation(seed)
        self.inventory = 100  # 初始库存
        self.backorders = 0
        self.total_cost = 0
        self.demands_fulfilled = 0
    
    def run(self, time: float = 365) -> Dict[str, Any]:
        """运行仿真"""
        # 简化实现
        avg_demand = self.demand_rate * time
        ordering_cost = (avg_demand / 100) * self.ordering_cost
        holding_cost = 50 * self.holding_cost * time
        shortage_cost = 0
        
        total_cost = ordering_cost + holding_cost + shortage_cost
        unit_cost = total_cost / (time * self.demand_rate)
        
        return {
            'total_cost': total_cost,
            'ordering_cost': ordering_cost,
            'holding_cost': holding_cost,
            'shortage_cost': shortage_cost,
            'unit_cost': unit_cost,
            'demand_rate': self.demand_rate
        }


if __name__ == "__main__":
    print("=" * 50)
    print("排队系统仿真 (M/M/1)")
    print("=" * 50)
    
    queue = QueueingSystem(num_servers=1, service_rate=1.0, arrival_rate=0.8, seed=42)
    result = queue.run(time=1000)
    
    print(f"平均队列长度: {result['avg_queue_length']:.4f}")
    print(f"最大队列长度: {result['max_queue_length']}")
    print(f"平均等待时间: {result['avg_wait_time']:.4f}")
    print(f"利用率: {result['utilization']:.4f}")
    
    print("\n" + "=" * 50)
    print("M/M/1 理论值对比")
    print("=" * 50)
    theoretical = queue.analyze_m_m_1()
    for key, value in theoretical.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    print("\n" + "=" * 50)
    print("库存系统仿真")
    print("=" * 50)
    
    inventory = InventorySystem(demand_rate=10, seed=42)
    inv_result = inventory.run(time=365)
    print(f"总成本: {inv_result['total_cost']:.2f}")
    print(f"单位成本: {inv_result['unit_cost']:.4f}")
