"""
Visualization Tools for RRAM-based Computing Systems.

This module provides comprehensive visualization tools for monitoring,
analyzing, and understanding RRAM-based computations and their behavior.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
from pathlib import Path
import time
from datetime import datetime

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel, RRAMMaterialType
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer, SpikingNeuron
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite, BenchmarkPlotter
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class RRAMVisualizer:
    """
    Comprehensive visualization tools for RRAM-based computing systems.
    """
    
    def __init__(self, 
                 style: str = 'seaborn',
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the RRAM visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.setup_style()
    
    def setup_style(self):
        """Setup visualization style and parameters."""
        plt.style.use(self.style)
        sns.set_palette("husl")
    
    def visualize_rram_conductance_matrix(self, 
                                        conductance_matrix: np.ndarray,
                                        title: str = "RRAM Conductance Matrix",
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize an RRAM conductance matrix as a heatmap.
        
        Args:
            conductance_matrix: Conductance matrix to visualize
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(conductance_matrix, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Conductance (S)')
        
        # Add value annotations if matrix is small enough
        if conductance_matrix.shape[0] <= 10 and conductance_matrix.shape[1] <= 10:
            for i in range(conductance_matrix.shape[0]):
                for j in range(conductance_matrix.shape[1]):
                    text = ax.text(j, i, f'{conductance_matrix[i, j]:.2e}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_hp_inv_convergence(self,
                                   residuals: List[float],
                                   title: str = "HP-INV Convergence",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the convergence of HP-INV algorithm.
        
        Args:
            residuals: List of residual values over iterations
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        iterations = range(len(residuals))
        
        # Linear scale plot
        ax1.plot(iterations, residuals, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual Norm')
        ax1.set_title(f'{title} (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        
        # Log scale plot
        ax2.semilogy(iterations, residuals, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Residual Norm (log scale)')
        ax2.set_title(f'{title} (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_rram_model_parameters(self,
                                     model: Union[AdvancedRRAMModel, MaterialSpecificRRAMModel],
                                     title: str = "RRAM Model Parameters",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the parameters of an RRAM model.
        
        Args:
            model: RRAM model instance
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title)
        
        # Plot 1: Conductance history
        if hasattr(model, 'conductance_history') and model.conductance_history:
            axes[0, 0].plot(model.conductance_history, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Conductance (S)')
            axes[0, 0].set_title('Conductance Evolution')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No conductance history', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Conductance Evolution')
        
        # Plot 2: Resistance distribution (if we have multiple values)
        if hasattr(model, 'conductance_history') and len(model.conductance_history) > 1:
            resistances = [1/g if g != 0 else float('inf') for g in model.conductance_history]
            axes[0, 1].hist(resistances, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Resistance (Ω)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Resistance Distribution')
            axes[0, 1].set_yscale('log')
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data for distribution', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Resistance Distribution')
        
        # Plot 3: Material-specific parameters
        if hasattr(model, 'params'):
            params_dict = {
                'Low Resistance': model.params.low_resistance,
                'High Resistance': model.params.high_resistance,
                'Variability': model.params.alpha,
                'Drift Coefficient': model.params.drift_coefficient
            }
            
            param_names = list(params_dict.keys())
            param_values = list(params_dict.values())
            
            axes[1, 0].bar(param_names, param_values, color='green', alpha=0.7)
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Model Parameters')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No parameters found', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Model Parameters')
        
        # Plot 4: Temperature effects
        if hasattr(model, 'temperature'):
            temp_effects = []
            conductance_range = np.linspace(
                model.params.low_resistance, 
                model.params.high_resistance, 
                100
            )
            
            for g in conductance_range:
                temp_adj = model.get_temperature_dependent_conductance(1.0/g)
                temp_effects.append(temp_adj)
            
            axes[1, 1].plot(conductance_range, temp_effects, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Base Conductance (S)')
            axes[1, 1].set_ylabel('Temperature-Adjusted Conductance (S)')
            axes[1, 1].set_title(f'Temperature Effects (T={model.temperature}K)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No temperature data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Temperature Effects')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_spiking_activity(self,
                                 layer: RRAMSpikingLayer,
                                 duration: float = 1.0,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize spiking activity in a spiking layer.
        
        Args:
            layer: RRAMSpikingLayer instance
            duration: Duration to simulate
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]))
        
        # Simulate some spiking activity
        dt = 0.1  # time step in ms
        time_points = np.arange(0, duration, dt)
        n_inputs = layer.n_inputs
        n_neurons = layer.n_neurons
        
        # Store spike times
        input_spikes = []
        output_spikes = []
        
        for t in time_points:
            # Create random input spikes for demonstration
            input_activity = (np.random.random(n_inputs) > 0.8).astype(float)
            
            # Process through the layer
            output_activity = layer.forward(input_activity, dt)
            
            input_spikes.append(input_activity.copy())
            output_spikes.append(output_activity.copy())
        
        # Convert to arrays
        input_spikes = np.array(input_spikes).T  # Shape: (n_inputs, n_time_steps)
        output_spikes = np.array(output_spikes).T  # Shape: (n_neurons, n_time_steps)
        
        # Plot input spikes
        spike_times_input, neuron_indices_input = np.where(input_spikes > 0.5)
        if len(spike_times_input) > 0:
            axes[0].scatter(spike_times_input * dt, neuron_indices_input, 
                          c='blue', s=20, alpha=0.7, label='Input Spikes')
        axes[0].set_ylabel('Neuron Index')
        axes[0].set_title('Input Spike Raster')
        axes[0].set_xlim(0, duration)
        axes[0].grid(True, alpha=0.3)
        
        # Plot output spikes
        spike_times_output, neuron_indices_output = np.where(output_spikes > 0.5)
        if len(spike_times_output) > 0:
            axes[1].scatter(spike_times_output * dt, neuron_indices_output, 
                          c='red', s=20, alpha=0.7, label='Output Spikes')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Neuron Index')
        axes[1].set_title('Output Spike Raster')
        axes[1].set_xlim(0, duration)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_neural_activity_3d(self,
                                   network: Union[RRAMSpikingLayer, RRAMReservoirComputer],
                                   inputs: List[np.ndarray],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create a 3D visualization of neural activity in a network.
        
        Args:
            network: RRAM network layer
            inputs: List of input patterns
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        if not CORE_MODULES_AVAILABLE:
            warnings.warn("Core modules not available for 3D visualization")
            return go.Figure()
        
        # Process the inputs through the network
        states = []
        for input_data in inputs:
            if hasattr(network, 'update_state'):
                # Reservoir computer
                state = network.update_state(input_data)
            elif hasattr(network, 'forward'):
                # Spiking layer
                state = network.forward(input_data)
            else:
                state = input_data
            
            states.append(state)
        
        if not states:
            return go.Figure()
        
        states = np.array(states)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add trajectory of states through the network
        fig.add_trace(go.Scatter3d(
            x=states[:, 0] if states.shape[1] > 0 else [0],
            y=states[:, 1] if states.shape[1] > 1 else [0],
            z=states[:, 2] if states.shape[1] > 2 else [0],
            mode='lines+markers',
            line=dict(width=5),
            marker=dict(size=6),
            name='Network Activity Trajectory'
        ))
        
        fig.update_layout(
            title='3D Neural Activity Visualization',
            scene=dict(
                xaxis_title='Neuron State Dim 1',
                yaxis_title='Neuron State Dim 2',
                zaxis_title='Neuron State Dim 3'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_optimization_progress(self,
                                      optimizer: RRAMAwareOptimizer,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the progress of an optimization algorithm.
        
        Args:
            optimizer: RRAMAwareOptimizer instance
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Optimization Progress Visualization')
        
        # Plot convergence history if available
        if hasattr(optimizer.ml_optimizer, 'performance_history') and optimizer.ml_optimizer.performance_history:
            perf_history = optimizer.ml_optimizer.performance_history
            iterations = range(len(perf_history))
            residuals = [p.get('residual_norm', 0) for p in perf_history]
            iterations_taken = [p.get('iterations', 0) for p in perf_history]
            
            # Plot 1: Residual over time
            axes[0, 0].plot(iterations, residuals, 'b-o', markersize=4)
            axes[0, 0].set_xlabel('Optimization Iteration')
            axes[0, 0].set_ylabel('Residual')
            axes[0, 0].set_title('Residual Improvement')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Iterations taken over time
            axes[0, 1].plot(iterations, iterations_taken, 'r-s', markersize=4)
            axes[0, 1].set_xlabel('Optimization Iteration')
            axes[0, 1].set_ylabel('Iterations to Converge')
            axes[0, 1].set_title('Computational Efficiency')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No optimization history data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Residual Improvement')
            
            axes[0, 1].text(0.5, 0.5, 'No optimization history data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Computational Efficiency')
        
        # Plot parameter evolution if available
        if hasattr(optimizer.ml_optimizer, 'parameter_weights'):
            param_names = list(optimizer.ml_optimizer.parameter_weights.keys())
            param_values = [optimizer.ml_optimizer.parameter_weights[name] for name in param_names]
            
            axes[1, 0].bar(param_names, param_values, color='green', alpha=0.7)
            axes[1, 0].set_ylabel('Parameter Weight')
            axes[1, 0].set_title('Learned Parameter Weights')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No parameter weights data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learned Parameter Weights')
        
        # Plot efficiency score if available
        if hasattr(optimizer.ml_optimizer, 'performance_history') and optimizer.ml_optimizer.performance_history:
            efficiency_scores = []
            for perf in optimizer.ml_optimizer.performance_history:
                # Calculate efficiency as iterations / (residual + small constant)
                residual = perf.get('residual_norm', 1e-6)
                iterations = perf.get('iterations', 10)
                efficiency = iterations / (residual + 1e-10)
                efficiency_scores.append(efficiency)
            
            axes[1, 1].plot(range(len(efficiency_scores)), efficiency_scores, 'purple', linewidth=2)
            axes[1, 1].set_xlabel('Optimization Iteration')
            axes[1, 1].set_ylabel('Efficiency Score')
            axes[1, 1].set_title('Optimization Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No efficiency data', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Optimization Efficiency')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_real_time_monitor(self,
                               rram_interface: Optional[object] = None,
                               update_interval: float = 1.0) -> None:
        """
        Create a real-time monitoring interface for RRAM operation.
        
        Args:
            rram_interface: Optional RRAM interface for real-time data
            update_interval: Update interval in seconds
        """
        print("Starting RRAM Real-Time Monitor...")
        print(f"Update interval: {update_interval}s")
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlabel('Time')
        ax.set_ylabel('Conductance (S)')
        ax.set_title('Real-Time RRAM Conductance Monitoring')
        ax.grid(True, alpha=0.3)
        
        # Initialize plot data
        times = []
        conductances = []
        
        try:
            while True:
                current_time = time.time()
                
                # Get current conductance from interface if available
                if rram_interface is not None:
                    try:
                        # Read the current matrix or conductance values
                        if hasattr(rram_interface, 'read_matrix'):
                            matrix = rram_interface.read_matrix()
                            # Use average conductance as a summary metric
                            avg_conductance = np.mean(matrix) if matrix.size > 0 else 0
                        else:
                            avg_conductance = np.random.random() * 1e-4  # Simulated value
                    except (AttributeError, IOError, RuntimeError, ValueError) as e:
                        warnings.warn(f"Failed to read RRAM matrix: {e}")
                        avg_conductance = np.random.random() * 1e-4  # Default to random if error
                else:
                    # Simulate conductance changes for demonstration
                    avg_conductance = 1e-4 * (0.5 + 0.5 * np.sin(current_time / 5))
                
                # Update data
                times.append(current_time)
                conductances.append(avg_conductance)
                
                # Keep only recent data points (last 50 points)
                if len(times) > 50:
                    times.pop(0)
                    conductances.pop(0)
                
                # Update the plot
                ax.clear()
                ax.plot(times, conductances, 'b-', linewidth=2, marker='o')
                ax.set_xlabel('Time')
                ax.set_ylabel('Conductance (S)')
                ax.set_title('Real-Time RRAM Conductance Monitoring')
                ax.grid(True, alpha=0.3)
                
                # Add annotations
                ax.text(0.02, 0.98, f'Current: {avg_conductance:.2e}S', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.pause(update_interval)
                
        except KeyboardInterrupt:
            print("\nReal-time monitoring stopped.")
            plt.close(fig)


class RRAMDashboard:
    """
    Interactive dashboard for monitoring RRAM-based systems.
    """
    
    def __init__(self):
        """Initialize the RRAM dashboard."""
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Conductance Matrix', 'Convergence', 'Activity Heatmap', 'Parameter Evolution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
    
    def update_dashboard(self, 
                        conductance_matrix: Optional[np.ndarray] = None,
                        convergence_data: Optional[Dict[str, Any]] = None,
                        activity_data: Optional[np.ndarray] = None,
                        parameter_data: Optional[Dict[str, List[float]]] = None) -> None:
        """
        Update the dashboard with new data.
        
        Args:
            conductance_matrix: Current conductance matrix
            convergence_data: Convergence information
            activity_data: Activity matrix (e.g., neuron firing rates)
            parameter_data: Parameter evolution data
        """
        # Clear the figure
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Conductance Matrix', 'Convergence', 'Activity Heatmap', 'Parameter Evolution'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Add conductance matrix heatmap
        if conductance_matrix is not None:
            self.fig.add_trace(
                go.Heatmap(z=conductance_matrix, colorscale='Viridis'),
                row=1, col=1
            )
        
        # Add convergence plot
        if convergence_data is not None and 'residuals' in convergence_data:
            self.fig.add_trace(
                go.Scatter(x=list(range(len(convergence_data['residuals']))), 
                          y=convergence_data['residuals'],
                          mode='lines+markers',
                          name='Residuals'),
                row=1, col=2
            )
        
        # Add activity heatmap
        if activity_data is not None:
            self.fig.add_trace(
                go.Heatmap(z=activity_data, colorscale='Plasma'),
                row=2, col=1
            )
        
        # Add parameter evolution plot
        if parameter_data is not None:
            for param_name, values in parameter_data.items():
                self.fig.add_trace(
                    go.Scatter(x=list(range(len(values))), 
                              y=values,
                              mode='lines+markers',
                              name=param_name),
                    row=2, col=2
                )
        
        self.fig.update_layout(height=800, showlegend=True, title_text="RRAM Real-Time Dashboard")
    
    def show(self):
        """Display the dashboard."""
        self.fig.show()
    
    def save_dashboard(self, filename: str = "rram_dashboard.html"):
        """
        Save the dashboard to an HTML file.
        
        Args:
            filename: Name of file to save the dashboard
        """
        if self.fig:
            self.fig.write_html(filename)


class RRAMAnalytics:
    """
    Analytics tools for RRAM systems including statistical analysis and insights.
    """
    
    def __init__(self):
        """Initialize analytics tools."""
        self.analyses_history = []
    
    def analyze_conductance_distribution(self, conductance_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the distribution of conductance values in a matrix.
        
        Args:
            conductance_matrix: Matrix of conductance values
            
        Returns:
            Dictionary with statistical analysis
        """
        # Flatten the matrix to analyze conductance distribution
        flat_conductances = conductance_matrix.flatten()
        
        analysis = {
            'mean_conductance': float(np.mean(flat_conductances)),
            'std_conductance': float(np.std(flat_conductances)),
            'min_conductance': float(np.min(flat_conductances)),
            'max_conductance': float(np.max(flat_conductances)),
            'median_conductance': float(np.median(flat_conductances)),
            'histogram_bins': 20,
            'histogram_values': [],
            'conductance_range': float(np.max(flat_conductances) - np.min(flat_conductances))
        }
        
        # Calculate histogram
        hist, bins = np.histogram(flat_conductances, bins=analysis['histogram_bins'])
        analysis['histogram_values'] = hist.tolist()
        analysis['histogram_bins'] = bins.tolist()
        
        # Conductance uniformity metrics
        analysis['conductance_uniformity'] = 1.0 - (analysis['std_conductance'] / (analysis['mean_conductance'] + 1e-12))
        
        # Add to history
        analysis['timestamp'] = time.time()
        analysis['analysis_type'] = 'conductance_distribution'
        self.analyses_history.append(analysis)
        
        return analysis
    
    def analyze_stability(self, 
                         conductance_matrices: List[np.ndarray],
                         time_intervals: List[float]) -> Dict[str, Any]:
        """
        Analyze the stability of conductance values over time.
        
        Args:
            conductance_matrices: List of conductance matrices over time
            time_intervals: Time intervals between measurements
            
        Returns:
            Dictionary with stability analysis
        """
        if len(conductance_matrices) < 2:
            return {'error': 'Need at least 2 matrices for stability analysis'}
        
        # Calculate changes in conductance over time
        relative_changes = []
        absolute_changes = []
        
        for i in range(1, len(conductance_matrices)):
            prev_matrix = conductance_matrices[i-1]
            curr_matrix = conductance_matrices[i]
            
            # Calculate relative change
            relative_change = np.abs((curr_matrix - prev_matrix) / (prev_matrix + 1e-12))
            relative_changes.append(np.mean(relative_change))
            
            # Calculate absolute change
            absolute_change = np.abs(curr_matrix - prev_matrix)
            absolute_changes.append(np.mean(absolute_change))
        
        analysis = {
            'avg_relative_stability': float(np.mean(relative_changes)),
            'std_relative_stability': float(np.std(relative_changes)),
            'avg_absolute_stability': float(np.mean(absolute_changes)),
            'std_absolute_stability': float(np.std(absolute_changes)),
            'time_based_changes': [
                {'interval': time_intervals[i], 'relative_change': float(relative_changes[i]), 
                 'absolute_change': float(absolute_changes[i])}
                for i in range(len(relative_changes))
            ],
            'drift_rate': float(np.mean(absolute_changes) / np.mean(time_intervals)) if time_intervals else 0
        }
        
        # Add to history
        analysis['timestamp'] = time.time()
        analysis['analysis_type'] = 'stability'
        self.analyses_history.append(analysis)
        
        return analysis
    
    def generate_insights_report(self) -> str:
        """
        Generate an insights report from all analyses.
        
        Returns:
            Insights report as a string
        """
        if not self.analyses_history:
            return "No analyses performed yet."
        
        report = f"RRAM Analytics Insights Report\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "="*50 + "\n\n"
        
        # Group analyses by type
        conductance_analyses = [a for a in self.analyses_history if a.get('analysis_type') == 'conductance_distribution']
        stability_analyses = [a for a in self.analyses_history if a.get('analysis_type') == 'stability']
        
        if conductance_analyses:
            report += "Conductance Distribution Insights:\n"
            report += "-"*30 + "\n"
            
            # Average metrics
            avg_mean = np.mean([a['mean_conductance'] for a in conductance_analyses])
            avg_std = np.mean([a['std_conductance'] for a in conductance_analyses])
            avg_uniformity = np.mean([a['conductance_uniformity'] for a in conductance_analyses])
            
            report += f"  - Average conductance: {avg_mean:.2e} S\n"
            report += f"  - Average std dev: {avg_std:.2e} S\n"
            report += f"  - Average uniformity: {avg_uniformity:.3f}\n"
            
            if avg_uniformity > 0.95:
                report += f"  - Conductance distribution is very uniform\n"
            elif avg_uniformity > 0.8:
                report += f"  - Conductance distribution is reasonably uniform\n"
            else:
                report += f"  - Conductance distribution shows significant variation\n"
        
        if stability_analyses:
            report += "\nStability Insights:\n"
            report += "-"*15 + "\n"
            
            avg_stability = np.mean([a['avg_relative_stability'] for a in stability_analyses])
            drift_rate = np.mean([a['drift_rate'] for a in stability_analyses])
            
            report += f"  - Average relative stability: {avg_stability:.2e}\n"
            report += f"  - Average drift rate: {drift_rate:.2e} S/s\n"
            
            if avg_stability < 1e-3:
                report += f"  - Excellent stability observed\n"
            elif avg_stability < 1e-2:
                report += f"  - Good stability with minor variations\n"
            else:
                report += f"  - Stability could be improved - consider refresh cycles\n"
        
        return report


def create_rram_visualization_suite():
    """
    Create a comprehensive visualization suite for RRAM systems.
    
    Returns:
        Tuple of (visualizer, dashboard, analytics)
    """
    visualizer = RRAMVisualizer()
    dashboard = RRAMDashboard()
    analytics = RRAMAnalytics()
    
    return visualizer, dashboard, analytics


# Example usage and test functions
def example_visualization_workflow():
    """
    Example workflow showing how to use visualization tools.
    """
    print("Creating RRAM Visualization Example...")
    
    # Create a visualizer
    viz = RRAMVisualizer()
    
    # Create a sample conductance matrix
    matrix = np.random.rand(8, 8) * 1e-4  # Microsiemens
    matrix = matrix + 0.3 * np.eye(8)  # Add diagonal dominance
    
    # Visualize the matrix
    fig1 = viz.visualize_rram_conductance_matrix(matrix, "Example Conductance Matrix")
    plt.show()
    
    # Create residuals for convergence demo
    residuals = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    fig2 = viz.visualize_hp_inv_convergence(residuals, "Example Convergence")
    plt.show()
    
    # Create spiking layer visualization
    if CORE_MODULES_AVAILABLE:
        try:
            spiking_layer = RRAMSpikingLayer(5, 10)  # 5 inputs, 10 neurons
            fig3 = viz.visualize_spiking_activity(spiking_layer, duration=1.0)
            plt.show()
        except Exception as e:
            print(f"Could not create spiking visualization: {e}")


if __name__ == "__main__":
    example_visualization_workflow()