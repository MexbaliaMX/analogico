"""
Performance Profiling Tools for RRAM Systems.

This module provides comprehensive tools for profiling and analyzing
the performance bottlenecks in RRAM-based systems including timing
analysis, memory usage, power consumption, and computational efficiency.
"""
import time
import numpy as np
import threading
import queue
import psutil
import os
import sys
import warnings
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMNeuromorphicNetwork, RRAMSpikingLayer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite
    from .advanced_materials_simulation import AdvancedMaterialsSimulator
    from .edge_computing_integration import EdgeRRAMOptimizer
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class ProfilerEventType(Enum):
    """Types of profiler events."""
    FUNCTION_CALL = "function_call"
    MEMORY_USAGE = "memory_usage"
    COMPUTATION = "computation"
    IO_OPERATION = "io_operation"
    HARDWARE_ACCESS = "hardware_access"
    ALGORITHM_STEP = "algorithm_step"


class PerformanceMetric(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    POWER_CONSUMPTION = "power_consumption"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    STABILITY = "stability"
    SCALABILITY = "scalability"


@dataclass
class ProfileEvent:
    """Data class for profiling events."""
    name: str
    start_time: float
    end_time: float
    event_type: ProfilerEventType
    metrics: Dict[str, Any]
    thread_id: int
    stack_trace: Optional[str] = None


class PerformanceProfiler:
    """
    Comprehensive performance profiler for RRAM systems.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 log_io_operations: bool = True,
                 log_memory_usage: bool = True,
                 log_computations: bool = True):
        """
        Initialize the performance profiler.
        
        Args:
            rram_interface: Optional RRAM interface for hardware profiling
            log_io_operations: Whether to log I/O operations
            log_memory_usage: Whether to log memory usage
            log_computations: Whether to log computations
        """
        self.rram_interface = rram_interface
        self.log_io_operations = log_io_operations
        self.log_memory_usage = log_memory_usage
        self.log_computations = log_computations
        
        self.events = []
        self.profiles = {}
        self.start_times = {}
        self.metrics = {
            'total_execution_time': 0.0,
            'peak_memory_usage': 0.0,
            'total_memory_allocated': 0.0,
            'average_cpu_utilization': 0.0,
            'function_calls': 0,
            'hardware_accesses': 0
        }
        self.active_profiling = False
        self.profiling_thread = None
    
    def start_profiling(self, name: str = "default_profile"):
        """
        Start profiling a specific operation or section.
        
        Args:
            name: Name of the profile session
        """
        self.active_profiling = True
        self.profiles[name] = []
        self.start_times[name] = time.time()
        
        # Start background profiling if needed
        if self.profiling_thread is None or not self.profiling_thread.is_alive():
            self.profiling_thread = threading.Thread(target=self._background_profiling)
            self.profiling_thread.daemon = True
            self.profiling_thread.start()
    
    def stop_profiling(self, name: str = "default_profile") -> Dict[str, Any]:
        """
        Stop profiling and return results.
        
        Args:
            name: Name of the profile session
            
        Returns:
            Profiling results dictionary
        """
        if name in self.start_times:
            total_time = time.time() - self.start_times[name]
            profile_events = self.profiles.get(name, [])
            
            # Calculate metrics
            avg_execution_time = np.mean([e.end_time - e.start_time for e in profile_events]) if profile_events else 0
            peak_memory = max([e.metrics.get('memory_usage', 0) for e in profile_events], default=0)
            
            results = {
                'session_name': name,
                'total_time': total_time,
                'events_count': len(profile_events),
                'average_event_time': avg_execution_time,
                'peak_memory_usage': peak_memory,
                'events': [e.__dict__ for e in profile_events],
                'metrics_summary': {
                    'total_execution_time': total_time,
                    'events_processed': len(profile_events),
                    'average_event_time': avg_execution_time,
                    'peak_memory': peak_memory
                }
            }
            
            # Update global metrics
            self.metrics['total_execution_time'] += total_time
            
            return results
        else:
            return {}
    
    def record_event(self, 
                    name: str, 
                    event_type: ProfilerEventType = ProfilerEventType.FUNCTION_CALL,
                    metrics: Optional[Dict[str, Any]] = None):
        """
        Record a profiling event.
        
        Args:
            name: Name of the event
            event_type: Type of event
            metrics: Additional metrics associated with the event
        """
        if not self.active_profiling:
            return
        
        if metrics is None:
            metrics = {}
        
        # Get current memory usage
        if self.log_memory_usage:
            process = psutil.Process(os.getpid())
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            metrics['memory_usage'] = current_memory
            metrics['memory_percent'] = process.memory_percent()
        
        # Get CPU usage
        metrics['cpu_percent'] = psutil.cpu_percent()
        
        event = ProfileEvent(
            name=name,
            start_time=time.time(),
            end_time=time.time(),
            event_type=event_type,
            metrics=metrics,
            thread_id=threading.get_ident()
        )
        
        # Add to active profiles
        for profile_name in self.start_times.keys():
            if profile_name in self.profiles:
                self.profiles[profile_name].append(event)
        
        # Add to global events
        self.events.append(event)
        
        # Update metrics
        if event_type == ProfilerEventType.FUNCTION_CALL:
            self.metrics['function_calls'] += 1
        elif event_type == ProfilerEventType.HARDWARE_ACCESS and self.rram_interface:
            self.metrics['hardware_accesses'] += 1
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Profile the execution of a function.
        
        Args:
            func: Function to profile
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Tuple of (function result, profiling metrics)
        """
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        self.record_event(
            name=f"{func_name}_start",
            event_type=ProfilerEventType.FUNCTION_CALL,
            metrics={'arguments_count': len(args), 'keyword_args_count': len(kwargs)}
        )
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = None
            kwargs['error'] = str(e)
        
        end_time = time.time()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            'execution_time': end_time - start_time,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_delta': memory_after - memory_before,
            'return_value_type': type(result).__name__ if result is not None else 'None'
        }
        
        self.record_event(
            name=f"{func_name}_end",
            event_type=ProfilerEventType.FUNCTION_CALL,
            metrics=metrics
        )
        
        return result, metrics
    
    def profile_algorithm(self, 
                         algorithm_func: Callable, 
                         *args, 
                         algorithm_name: str = "unknown") -> Dict[str, Any]:
        """
        Profile the execution of an algorithm.
        
        Args:
            algorithm_func: Algorithm function to profile
            *args: Arguments to the algorithm
            algorithm_name: Name of the algorithm
            
        Returns:
            Profiling results dictionary
        """
        self.start_profiling(f"algorithm_{algorithm_name}")
        
        # Record algorithm start
        self.record_event(
            name=f"{algorithm_name}_start",
            event_type=ProfilerEventType.ALGORITHM_STEP,
            metrics={'parameters_count': len(args)}
        )
        
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = algorithm_func(*args)
        except Exception as e:
            result = None
            print(f"Algorithm {algorithm_name} failed: {e}")
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Record algorithm end
        algorithm_metrics = {
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'peak_memory': max(start_memory, end_memory),
            'result_type': type(result).__name__ if result is not None else 'None'
        }
        
        self.record_event(
            name=f"{algorithm_name}_end",
            event_type=ProfilerEventType.ALGORITHM_STEP,
            metrics=algorithm_metrics
        )
        
        # Stop profiling and get results
        results = self.stop_profiling(f"algorithm_{algorithm_name}")
        
        return results
    
    def _background_profiling(self):
        """
        Background thread for continuous profiling.
        """
        while self.active_profiling:
            if self.log_memory_usage:
                process = psutil.Process(os.getpid())
                memory_metrics = {
                    'memory_rss_mb': process.memory_info().rss / 1024 / 1024,
                    'memory_vms_mb': process.memory_info().vms / 1024 / 1024,
                    'memory_percent': process.memory_percent(),
                    'cpu_percent': process.cpu_percent()
                }
                
                self.record_event(
                    name="system_monitor",
                    event_type=ProfilerEventType.COMPUTATION,
                    metrics=memory_metrics
                )
            
            time.sleep(0.1)  # Sample every 100ms

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'profiler_metrics': self.metrics,
            'event_counts_by_type': {
                event_type.value: len([e for e in self.events if e.event_type == event_type])
                for event_type in ProfilerEventType
            },
            'total_events': len(self.events),
            'active_profiles': list(self.start_times.keys()),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_mb': psutil.virtual_memory().total / 1024 / 1024,
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'platform': sys.platform,
                'python_version': sys.version
            }
        }
        
        # Calculate derived metrics
        if self.events:
            execution_times = [(e.end_time - e.start_time) for e in self.events]
            report['derived_metrics'] = {
                'average_execution_time': np.mean(execution_times),
                'max_execution_time': np.max(execution_times),
                'min_execution_time': np.min(execution_times),
                'std_execution_time': np.std(execution_times)
            }
        
        return report
    
    def save_profile_data(self, filename: str = "rram_profile.json"):
        """
        Save profiling data to file.
        
        Args:
            filename: Name of file to save profiling data
        """
        data = {
            'events': [e.__dict__ for e in self.events],
            'profiles': self.profiles,
            'metrics': self.metrics,
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_profile_data(self, filename: str = "rram_profile.json"):
        """
        Load profiling data from file.
        
        Args:
            filename: Name of file to load profiling data from
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.events = [ProfileEvent(**event) for event in data['events']]
        self.profiles = data['profiles']
        self.metrics = data['metrics']
    
    def visualize_profile(self, 
                         report: Dict[str, Any], 
                         plot_type: str = "execution_time",
                         save_path: Optional[str] = None):
        """
        Create visualizations of profiling data.
        
        Args:
            report: Profiling report from get_performance_report()
            plot_type: Type of plot ('execution_time', 'memory_usage', 'event_counts')
            save_path: Optional file path to save the plot
        """
        if plot_type == "execution_time":
            # Extract execution times
            exec_times = []
            labels = []
            
            for profile_name, events in self.profiles.items():
                for event in events:
                    exec_time = event.end_time - event.start_time
                    exec_times.append(exec_time * 1000)  # Convert to milliseconds
                    labels.append(f"{profile_name}:{event.name}")
            
            if exec_times:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(exec_times)), exec_times)
                plt.xlabel('Events')
                plt.ylabel('Execution Time (ms)')
                plt.title('Execution Time Profile')
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                else:
                    plt.show()
        
        elif plot_type == "memory_usage":
            # Extract memory usage data
            memory_values = []
            labels = []
            
            for event in self.events:
                if 'memory_usage' in event.metrics:
                    memory_values.append(event.metrics['memory_usage'])
                    labels.append(event.name)
            
            if memory_values:
                plt.figure(figsize=(10, 6))
                plt.plot(memory_values, marker='o')
                plt.xlabel('Event Index')
                plt.ylabel('Memory Usage (MB)')
                plt.title('Memory Usage Over Time')
                plt.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path)
                else:
                    plt.show()
        
        elif plot_type == "event_counts":
            # Count events by type
            event_counts = report['event_counts_by_type']
            types = list(event_counts.keys())
            counts = list(event_counts.values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(types, counts)
            plt.xlabel('Event Type')
            plt.ylabel('Count')
            plt.title('Event Count by Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()


class BottleneckAnalyzer:
    """
    Analyze performance bottlenecks in RRAM systems.
    """
    
    def __init__(self, profiler: PerformanceProfiler):
        """
        Initialize the bottleneck analyzer.
        
        Args:
            profiler: Performance profiler instance
        """
        self.profiler = profiler
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze system for performance bottlenecks.
        
        Returns:
            Analysis results with bottlenecks and recommendations
        """
        # Get events with execution times
        events_with_times = [
            e for e in self.profiler.events 
            if e.end_time > e.start_time
        ]
        
        if not events_with_times:
            return {"message": "No profiling data available"}
        
        # Calculate execution times
        execution_times = [
            (e, e.end_time - e.start_time) 
            for e in events_with_times
        ]
        
        # Find longest running events
        execution_times.sort(key=lambda x: x[1], reverse=True)
        top_events = execution_times[:10]  # Top 10 slowest events
        
        # Analyze by event type
        type_analysis = {}
        for event_type in ProfilerEventType:
            type_events = [e for e in events_with_times if e.event_type == event_type]
            if type_events:
                times = [e.end_time - e.start_time for e in type_events]
                type_analysis[event_type.value] = {
                    'count': len(type_events),
                    'total_time': sum(times),
                    'average_time': np.mean(times),
                    'max_time': np.max(times),
                    'min_time': np.min(times)
                }
        
        # Memory usage analysis
        memory_events = [
            e for e in events_with_times 
            if 'memory_delta' in e.metrics or 'memory_usage' in e.metrics
        ]
        
        if memory_events:
            memory_changes = [
                e.metrics.get('memory_delta', 0) 
                for e in memory_events
            ]
            max_memory_increase = max(memory_changes, default=0)
            
            # Find memory-intensive events
            memory_intensive = [
                (e, e.metrics.get('memory_delta', 0))
                for e in memory_events
                if e.metrics.get('memory_delta', 0) > 0
            ]
            memory_intensive.sort(key=lambda x: x[1], reverse=True)
            top_memory_events = memory_intensive[:5]
        else:
            max_memory_increase = 0
            top_memory_events = []
        
        # Generate analysis results
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'bottleneck_analysis': {
                'slowest_operations': [
                    {
                        'name': event.name,
                        'type': event.event_type.value,
                        'execution_time': exec_time,
                        'thread_id': event.thread_id
                    }
                    for event, exec_time in top_events
                ],
                'type_analysis': type_analysis,
                'memory_analysis': {
                    'max_memory_increase_mb': max_memory_increase,
                    'top_memory_events': [
                        {
                            'name': event.name,
                            'memory_increase_mb': mem_change
                        }
                        for event, mem_change in top_memory_events
                    ]
                },
                'recommendations': self._generate_recommendations(top_events, top_memory_events)
            }
        }
        
        return analysis
    
    def _generate_recommendations(self, 
                                 slow_events: List[Tuple[ProfileEvent, float]], 
                                 memory_events: List[Tuple[ProfileEvent, float]]) -> List[str]:
        """
        Generate recommendations based on bottleneck analysis.
        
        Args:
            slow_events: Top slowest events
            memory_events: Top memory-consuming events
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for slow functions
        slow_functions = [event for event, _ in slow_events if 'function' in event.name.lower()]
        if slow_functions:
            recommendations.append(
                f"Identified {len(slow_functions)} slow functions (>10ms). Consider optimization."
            )
        
        # Check for memory issues
        if memory_events and any(mem > 10 for _, mem in memory_events):  # More than 10MB
            recommendations.append(
                "High memory allocations detected (>10MB). Consider memory pooling or reuse."
            )
        
        # Check for I/O operations
        io_events = [event for event, _ in slow_events if event.event_type == ProfilerEventType.IO_OPERATION]
        if io_events:
            recommendations.append(
                f"Detected {len(io_events)} I/O operations that may be blocking. Consider async I/O."
            )
        
        # Check for hardware access
        hw_events = [event for event, _ in slow_events if event.event_type == ProfilerEventType.HARDWARE_ACCESS]
        if hw_events:
            recommendations.append(
                "Hardware access appears to be a bottleneck. Consider batch operations or caching."
            )
        
        if not recommendations:
            recommendations.append("No significant bottlenecks detected in current profiling data.")
        
        return recommendations
    
    def generate_optimization_report(self) -> str:
        """
        Generate a textual optimization report.
        
        Returns:
            Optimization report as string
        """
        analysis = self.analyze_bottlenecks()
        
        report = "RRAM System Performance Optimization Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Analysis Timestamp: {analysis['timestamp']}\n\n"
        
        # Bottleneck analysis
        bottle_analysis = analysis['bottleneck_analysis']
        
        report += "Top Slowest Operations:\n"
        report += "-" * 20 + "\n"
        for i, op in enumerate(bottle_analysis['slowest_operations'][:5]):
            report += f"  {i+1}. {op['name']} ({op['type']}): {op['execution_time']*1000:.2f}ms\n"
        
        report += "\nResource Usage by Type:\n"
        report += "-" * 20 + "\n"
        for type_name, stats in bottle_analysis['type_analysis'].items():
            report += f"  {type_name}: {stats['count']} operations, {stats['total_time']*1000:.2f}ms total\n"
        
        report += "\nMemory Analysis:\n"
        report += "-" * 15 + "\n"
        report += f"  Max memory increase: {bottle_analysis['memory_analysis']['max_memory_increase_mb']:.2f}MB\n"
        
        if bottle_analysis['memory_analysis']['top_memory_events']:
            report += "  Top memory consumers:\n"
            for i, event in enumerate(bottle_analysis['memory_analysis']['top_memory_events'][:3]):
                report += f"    {i+1}. {event['name']}: {event['memory_increase_mb']:.2f}MB\n"
        
        report += "\nRecommendations:\n"
        report += "-" * 15 + "\n"
        for rec in bottle_analysis['recommendations']:
            report += f"  • {rec}\n"
        
        return report


class HardwarePerformanceProfiler:
    """
    Profiler specifically for RRAM hardware operations.
    """
    
    def __init__(self, rram_interface: object):
        """
        Initialize the hardware profiler.
        
        Args:
            rram_interface: RRAM interface instance
        """
        self.rram_interface = rram_interface
        self.hardware_events = []
    
    def profile_write_matrix(self, 
                            matrix: np.ndarray, 
                            **kwargs) -> Dict[str, Any]:
        """
        Profile the matrix write operation to RRAM.
        
        Args:
            matrix: Matrix to write to RRAM
            **kwargs: Additional parameters
            
        Returns:
            Profiling results
        """
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        success = self.rram_interface.write_matrix(matrix, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        metrics = {
            'matrix_size': matrix.shape,
            'write_successful': success,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'data_volume_mb': matrix.size * 8 / 1024 / 1024  # 8 bytes per double
        }
        
        event = {
            'operation': 'write_matrix',
            'timestamp': start_time,
            'metrics': metrics
        }
        
        self.hardware_events.append(event)
        
        return metrics
    
    def profile_read_matrix(self, **kwargs) -> Dict[str, Any]:
        """
        Profile the matrix read operation from RRAM.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            Profiling results
        """
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        matrix = self.rram_interface.read_matrix(**kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        metrics = {
            'matrix_size': matrix.shape if matrix is not None else None,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'data_volume_mb': matrix.size * 8 / 1024 / 1024 if matrix is not None else 0
        }
        
        event = {
            'operation': 'read_matrix',
            'timestamp': start_time,
            'metrics': metrics
        }
        
        self.hardware_events.append(event)
        
        return metrics
    
    def profile_matrix_operation(self, 
                                operation: str, 
                                *args, 
                                **kwargs) -> Dict[str, Any]:
        """
        Profile a matrix operation on RRAM.
        
        Args:
            operation: Name of the operation (mvm, inversion, etc.)
            *args: Arguments for the operation
            **kwargs: Additional parameters
            
        Returns:
            Profiling results
        """
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        try:
            if operation == 'mvm':
                result = self.rram_interface.matrix_vector_multiply(*args, **kwargs)
            elif operation == 'invert_matrix':
                result = self.rram_interface.invert_matrix(*args, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            success = result is not None
        except Exception as e:
            result = None
            success = False
            print(f"Hardware operation failed: {e}")
        
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        metrics = {
            'operation': operation,
            'success': success,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'result_type': type(result).__name__ if result is not None else 'None'
        }
        
        event = {
            'operation': f'hardware_{operation}',
            'timestamp': start_time,
            'metrics': metrics
        }
        
        self.hardware_events.append(event)
        
        return metrics
    
    def get_hardware_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of hardware performance metrics.
        
        Returns:
            Summary of hardware performance
        """
        if not self.hardware_events:
            return {"message": "No hardware operations profiled yet"}
        
        # Calculate metrics
        write_ops = [e for e in self.hardware_events if e['operation'] == 'write_matrix']
        read_ops = [e for e in self.hardware_events if e['operation'] == 'read_matrix']
        other_ops = [e for e in self.hardware_events if e['operation'] not in ['write_matrix', 'read_matrix']]
        
        summary = {
            'total_operations': len(self.hardware_events),
            'write_operations': len(write_ops),
            'read_operations': len(read_ops),
            'other_operations': len(other_ops),
            'write_performance': self._calculate_performance_stats(write_ops),
            'read_performance': self._calculate_performance_stats(read_ops),
            'other_performance': self._calculate_performance_stats(other_ops)
        }
        
        return summary
    
    def _calculate_performance_stats(self, events: List[Dict]) -> Dict[str, float]:
        """
        Calculate performance statistics for a set of events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Performance statistics
        """
        if not events:
            return {}
        
        execution_times = [e['metrics']['execution_time'] for e in events if 'execution_time' in e['metrics']]
        
        if not execution_times:
            return {}
        
        return {
            'count': len(execution_times),
            'average_time_ms': np.mean(execution_times) * 1000,
            'min_time_ms': np.min(execution_times) * 1000,
            'max_time_ms': np.max(execution_times) * 1000,
            'std_time_ms': np.std(execution_times) * 1000
        }


def create_comprehensive_profiler( 
    rram_interface: Optional[object] = None
) -> Tuple[PerformanceProfiler, BottleneckAnalyzer, HardwarePerformanceProfiler]:
    """
    Create a comprehensive profiling system.
    
    Args:
        rram_interface: Optional RRAM interface
        
    Returns:
        Tuple of (general_profiler, bottleneck_analyzer, hardware_profiler)
    """
    general_profiler = PerformanceProfiler(rram_interface=rram_interface)
    bottleneck_analyzer = BottleneckAnalyzer(general_profiler)
    
    if rram_interface:
        hardware_profiler = HardwarePerformanceProfiler(rram_interface)
    else:
        hardware_profiler = None
    
    return general_profiler, bottleneck_analyzer, hardware_profiler


def demonstrate_profiling_tools():
    """
    Demonstrate the performance profiling tools.
    """
    print("Performance Profiling Tools Demonstration")
    print("="*50)
    
    # Create profiler components
    profiler, analyzer, hw_profiler = create_comprehensive_profiler()
    
    # Start profiling
    profiler.start_profiling("demo_profile")
    
    # Profile some computational operations
    print("Profiling computational operations...")
    
    # Profile a matrix operation
    G = np.random.rand(10, 10) * 1e-4
    G = G + 0.5 * np.eye(10)  # Well-conditioned
    b = np.random.rand(10)
    
    # Profile the HP-INV algorithm
    result, metrics = profiler.profile_algorithm(
        lambda x, y: hp_inv(x, y, max_iter=10, bits=4),
        G, b,
        algorithm_name="hp_inv"
    )
    
    print(f"HP-INV algorithm profiled. Execution time: {metrics['execution_time']:.4f}s")
    
    # Profile some other operations
    for i in range(5):
        # Record a mock event
        profiler.record_event(
            name=f"mock_operation_{i}",
            event_type=ProfilerEventType.COMPUTATION,
            metrics={'iteration': i, 'data_processed': np.random.randint(100, 1000)}
        )
    
    # Stop profiling and get results
    results = profiler.stop_profiling("demo_profile")
    print(f"Profile session completed. Events captured: {results['events_count']}")
    
    # Perform bottleneck analysis
    print("\nAnalyzing bottlenecks...")
    bottleneck_analysis = analyzer.analyze_bottlenecks()
    
    slow_operations = bottleneck_analysis['bottleneck_analysis']['slowest_operations'][:3]
    print(f"Top 3 slowest operations:")
    for i, op in enumerate(slow_operations):
        print(f"  {i+1}. {op['name']}: {op['execution_time']*1000:.2f}ms")
    
    # Generate optimization report
    print("\nGenerating optimization report...")
    opt_report = analyzer.generate_optimization_report()
    print(opt_report)
    
    # Get overall performance report
    print("Generating overall performance report...")
    perf_report = profiler.get_performance_report()
    print(f"Total events: {perf_report['total_events']}")
    print(f"Active profiles: {perf_report['active_profiles']}")
    
    # If we have a hardware profiler, demonstrate it too
    if hw_profiler:
        print("\nDemonstrating hardware profiler (simulated)...")
        # Note: This would actually interface with hardware in a real implementation
        hw_summary = hw_profiler.get_hardware_performance_summary()
        print(f"Hardware operations summary: {hw_summary.get('total_operations', 0)} operations")
    
    print("\nPerformance profiling demonstration completed!")


if __name__ == "__main__":
    demonstrate_profiling_tools()