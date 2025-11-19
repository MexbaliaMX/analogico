"""
Enhanced Visualization for RRAM Systems.

This module provides advanced visualization tools for RRAM systems,
including real-time monitoring, 3D visualization, data analysis,
and interactive visualizations for understanding RRAM behavior.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import warnings
from dataclasses import dataclass
import time
from datetime import datetime
import threading
import pandas as pd
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Import from existing modules
try:
    from .hp_inv import hp_inv, block_hp_inv, adaptive_hp_inv
    from .advanced_rram_models import AdvancedRRAMModel, MaterialSpecificRRAMModel
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
    from .neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer
    from .optimization_algorithms import RRAMAwareOptimizer
    from .benchmarking_suite import RRAMBenchmarkSuite
    from .advanced_materials_simulation import AdvancedMaterialsSimulator
    from .edge_computing_integration import EdgeRRAMOptimizer
    from .performance_profiling import PerformanceProfiler
    from .gpu_simulation_acceleration import GPUSimulationAccelerator
    from .real_time_adaptive_systems import RealTimeAdaptiveSystem
    CORE_MODULES_AVAILABLE = True
except ImportError:
    CORE_MODULES_AVAILABLE = False
    warnings.warn("Core modules not available")


class RRAMDataVisualizer:
    """
    Advanced data visualizer for RRAM systems.
    """
    
    def __init__(self, 
                 style: str = 'seaborn',
                 figsize: Tuple[int, int] = (14, 10),
                 show_warnings: bool = True):
        """
        Initialize the RRAM data visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
            show_warnings: Whether to show warning messages
        """
        self.style = style
        self.figsize = figsize
        self.show_warnings = show_warnings
        self.setup_visualization_environment()
    
    def setup_visualization_environment(self):
        """Setup visualization style and parameters."""
        plt.style.use(self.style)
        sns.set_palette("husl")
        
        # Set up plotly template
        import plotly.io as pio
        pio.templates.default = "plotly_white"
    
    def visualize_conductance_matrix(self, 
                                   conductance_matrix: np.ndarray,
                                   title: str = "RRAM Conductance Matrix",
                                   save_path: Optional[str] = None,
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Visualize conductance matrix as interactive heatmap.
        
        Args:
            conductance_matrix: Conductance matrix to visualize
            title: Plot title
            save_path: Optional path to save the plot
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        fig_size = figsize or self.figsize
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        
        # Create heatmap
        im = ax.imshow(conductance_matrix, cmap='viridis', aspect='auto', origin='upper')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Conductance (S)', fontsize=12)
        
        # Add value annotations if matrix is small enough
        if conductance_matrix.size <= 100:  # Only annotate small matrices
            for i in range(conductance_matrix.shape[0]):
                for j in range(conductance_matrix.shape[1]):
                    text = ax.text(j, i, f'{conductance_matrix[i, j]:.2e}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_conductance_matrix_interactive(self, conductance_matrix: np.ndarray) -> go.Figure:
        """
        Create an interactive heatmap of conductance matrix using Plotly.
        
        Args:
            conductance_matrix: Conductance matrix to visualize
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=conductance_matrix,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Conductance: %{z:.2e} S<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive RRAM Conductance Matrix',
            xaxis_title='Column',
            yaxis_title='Row',
            width=800,
            height=700
        )
        
        return fig
    
    def visualize_hp_inv_convergence(self,
                                   residuals: List[float],
                                   title: str = "HP-INV Convergence",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the convergence of HP-INV algorithm with detailed metrics.
        
        Args:
            residuals: List of residual values over iterations
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
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
        
        # Convergence rate
        if len(residuals) > 1:
            convergence_rates = []
            for i in range(1, len(residuals)):
                rate = residuals[i] / residuals[i-1] if residuals[i-1] != 0 else 1.0
                convergence_rates.append(rate)
            
            ax3.plot(iterations[1:], convergence_rates, 'g-s', linewidth=2, markersize=6)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Convergence Rate')
            ax3.set_title('Convergence Rate')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No improvement threshold')
            ax3.legend()
        
        # Cumulative improvement
        initial_residual = residuals[0] if residuals else 1.0
        improvements = [initial_residual - res for res in residuals]
        ax4.plot(iterations, improvements, 'm-^', linewidth=2, markersize=6)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cumulative Improvement')
        ax4.set_title('Cumulative Improvement')
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_material_characteristics(self, 
                                          materials_data: Dict[str, np.ndarray],
                                          property_name: str = "Conductance",
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize characteristics of different RRAM materials.
        
        Args:
            materials_data: Dictionary mapping material names to characteristic data
            property_name: Name of property being plotted
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'RRAM Material Characteristics: {property_name}', fontsize=16)
        
        materials = list(materials_data.keys())
        data_arrays = list(materials_data.values())
        
        # Plot 1: Box plots comparing distributions
        box_data = [arr.flatten() for arr in data_arrays]
        axes[0, 0].boxplot(box_data, labels=materials)
        axes[0, 0].set_title(f'Distribution of {property_name}')
        axes[0, 0].set_ylabel(property_name)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Histograms
        axes[0, 1].clear()  # Clear to avoid previous content issues
        for i, (material, data) in enumerate(materials_data.items()):
            flat_data = data.flatten()
            axes[0, 1].hist(flat_data, bins=20, alpha=0.6, label=material, density=True)
        axes[0, 1].set_title(f'{property_name} Distribution by Material')
        axes[0, 1].set_xlabel(property_name)
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Plot 3: Bar plot of means and stds
        means = [np.mean(arr) for arr in data_arrays]
        stds = [np.std(arr) for arr in data_arrays]
        
        x_pos = range(len(materials))
        width = 0.35
        
        axes[1, 0].bar([x - width/2 for x in x_pos], means, width, label='Mean', alpha=0.8)
        axes[1, 0].bar([x + width/2 for x in x_pos], stds, width, label='Std Dev', alpha=0.8)
        axes[1, 0].set_xlabel('Material')
        axes[1, 0].set_ylabel(property_name)
        axes[1, 0].set_title(f'Mean and Std Dev by Material')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(materials, rotation=45)
        axes[1, 0].legend()
        
        # Plot 4: Violin plots for detailed distribution
        violin_parts = axes[1, 1].violinplot([arr.flatten() for arr in data_arrays], 
                                            positions=range(len(materials)), 
                                            showmeans=True)
        axes[1, 1].set_title(f'{property_name} Distribution (Violin Plot)')
        axes[1, 1].set_ylabel(property_name)
        axes[1, 1].set_xticks(range(len(materials)))
        axes[1, 1].set_xticklabels(materials, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_spiking_activity(self,
                                 spike_times_per_neuron: Dict[int, List[float]],
                                 duration: float = 1.0,
                                 title: str = "Spiking Activity Raster",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize spiking activity as raster plot.
        
        Args:
            spike_times_per_neuron: Dictionary mapping neuron IDs to spike time lists
            duration: Duration of simulation
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data for raster plot
        all_times = []
        all_neurons = []
        
        for neuron_id, spike_times in spike_times_per_neuron.items():
            for spike_time in spike_times:
                if spike_time <= duration:  # Only include spikes within duration
                    all_times.append(spike_time)
                    all_neurons.append(neuron_id)
        
        if all_times:
            ax.scatter(all_times, all_neurons, s=10, c='black', alpha=0.7)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title(title)
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
        
        # Add spike count information
        total_spikes = len(all_times)
        neurons_active = len(set(all_neurons)) if all_neurons else 0
        ax.text(0.02, 0.98, f'Total Spikes: {total_spikes}\nNeurons Active: {neurons_active}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_conductance_surface(self, 
                                        X: np.ndarray, 
                                        Y: np.ndarray, 
                                        Z: np.ndarray,
                                        title: str = "3D Conductance Surface",
                                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create 3D surface visualization of conductance data.
        
        Args:
            X: X coordinates
            Y: Y coordinates  
            Z: Conductance values
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Surface(
            x=X, 
            y=Y, 
            z=Z,
            colorscale='Viridis',
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2e}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Parameter 1',
                yaxis_title='Parameter 2', 
                zaxis_title='Conductance (S)'
            ),
            width=800,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_real_time_monitoring(self,
                                     data_callback: Callable[[], Tuple[List, List, Dict]],
                                     duration: float = 10.0,
                                     update_interval: float = 0.1,
                                     title: str = "Real-Time RRAM Monitoring") -> plt.Figure:
        """
        Create real-time visualization that updates continuously.
        
        Args:
            data_callback: Function that returns (x_data, y_data, metadata) 
            duration: Duration to monitor (in seconds)
            update_interval: Update interval (in seconds)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, (ax_main, ax_hist) = plt.subplots(2, 1, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Initialize empty plots
        line, = ax_main.plot([], [], 'b-', linewidth=2, label='Real-time Data')
        ax_main.set_xlabel('Time')
        ax_main.set_ylabel('Value')
        ax_main.set_title('Live Data Stream')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend()
        
        # Histogram subplot
        ax_hist.set_xlabel('Value Distribution')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of Values')
        ax_hist.grid(True, alpha=0.3)
        
        # Store data for real-time updates
        times_data = []
        values_data = []
        
        def update_plot(frame):
            try:
                x_data, y_data, metadata = data_callback()
                
                # Extend data
                times_data.extend(x_data)
                values_data.extend(y_data)
                
                # Keep only recent data to prevent memory issues
                if len(times_data) > 1000:
                    times_data[:] = times_data[-500:]
                    values_data[:] = values_data[-500:]
                
                # Update main plot
                if times_data and values_data:
                    line.set_data(times_data, values_data)
                    
                    # Adjust axis limits to show recent data
                    ax_main.set_xlim(
                        max(0, max(times_data) - 10),  # Show last 10 time units
                        max(times_data) + 1 if times_data else 10
                    )
                    ax_main.set_ylim(
                        min(min(values_data), ax_main.get_ylim()[0]) if values_data else -1,
                        max(max(values_data), ax_main.get_ylim()[1]) if values_data else 1
                    )
                
                # Update histogram
                if values_data:
                    ax_hist.clear()
                    ax_hist.hist(values_data, bins=50, density=True, alpha=0.7)
                    ax_hist.set_xlabel('Value Distribution')
                    ax_hist.set_ylabel('Density')
                    ax_hist.set_title('Histogram of Values')
                    ax_hist.grid(True, alpha=0.3)
                
                return line, ax_hist
            except Exception as e:
                print(f"Error in real-time update: {e}")
                return line, ax_hist
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, 
            update_plot, 
            frames=int(duration / update_interval),
            interval=update_interval*1000,  # Convert to milliseconds
            blit=False
        )
        
        return fig


class InteractiveRRAMDashboard:
    """
    Interactive dashboard for monitoring and controlling RRAM systems.
    """
    
    def __init__(self, 
                 rram_interface: Optional[object] = None,
                 update_function: Optional[Callable] = None):
        """
        Initialize the interactive dashboard.
        
        Args:
            rram_interface: Optional RRAM interface for live data
            update_function: Custom update function for dashboard data
        """
        self.rram_interface = rram_interface
        self.update_function = update_function
        self.fig = None
        self.axs = None
        self.data = {'timestamps': [], 'values': []}
        self.lock = threading.Lock()
        
        # Control elements
        self.sliders = {}
        self.buttons = {}
        self.textboxes = {}
        
        self.setup_dashboard()
    
    def setup_dashboard(self):
        """Set up the dashboard layout and controls."""
        self.fig, self.axs = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('RRAM System Dashboard', fontsize=16)
        
        # Subplot titles
        self.axs[0, 0].set_title('Conductance Matrix Heatmap')
        self.axs[0, 1].set_title('Convergence Plot')
        self.axs[1, 0].set_title('Performance Metrics')
        self.axs[1, 1].set_title('Control Panel')
        
        # Add interactive controls to control panel
        self.setup_controls()
        
        # Add update button
        ax_button = plt.axes([0.4, 0.02, 0.1, 0.04])  # [left, bottom, width, height]
        button = Button(ax_button, 'Update Data')
        button.on_clicked(self.update_all_plots)
        self.buttons['update'] = button
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def setup_controls(self):
        """Set up interactive controls."""
        # Add sliders for parameter adjustment
        ax_precision = plt.axes([0.02, 0.7, 0.15, 0.03])
        slider_precision = Slider(ax_precision, 'Precision', 2, 8, valinit=4)
        self.sliders['precision'] = slider_precision
        
        ax_relax = plt.axes([0.02, 0.65, 0.15, 0.03])
        slider_relax = Slider(ax_relax, 'Relax Factor', 0.1, 2.0, valinit=0.9)
        self.sliders['relax_factor'] = slider_relax
        
        ax_noise = plt.axes([0.02, 0.6, 0.15, 0.03])
        slider_noise = Slider(ax_noise, 'Noise Level', 0.001, 0.1, valinit=0.01, valfmt="%.3f")
        self.sliders['noise_level'] = slider_noise
    
    def update_all_plots(self, event):
        """Update all plots with current data."""
        # Get current parameter values from sliders
        precision = int(self.sliders['precision'].val)
        relax_factor = self.sliders['relax_factor'].val
        noise_level = self.sliders['noise_level'].val
        
        # Update each plot
        self.update_matrix_heatmap()
        self.update_convergence_plot()
        self.update_performance_plot()
        
        # Print current parameter values
        print(f"Parameters updated: bits={precision}, relax_factor={relax_factor:.3f}, noise_std={noise_level:.4f}")
        
        # Redraw
        self.fig.canvas.draw()
    
    def update_matrix_heatmap(self):
        """Update the conductance matrix heatmap."""
        # Clear the subplot
        self.axs[0, 0].cla()
        self.axs[0, 0].set_title('Conductance Matrix Heatmap')
        
        # Generate sample matrix with current parameters
        # In real usage, this would come from the RRAM interface
        size = 8
        matrix = np.random.rand(size, size) * 1e-4  # Microsiemens
        matrix = matrix + 0.5 * np.eye(size)  # Diagonally dominant
        
        im = self.axs[0, 0].imshow(matrix, cmap='viridis', aspect='auto')
        self.axs[0, 0].set_xlabel('Column')
        self.axs[0, 0].set_ylabel('Row')
        
        # Add colorbar
        self.fig.colorbar(im, ax=self.axs[0, 0])
        
        # Add value annotations for small matrices
        if matrix.size <= 64:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = self.axs[0, 0].text(j, i, f'{matrix[i, j]:.2e}',
                                         ha="center", va="center", color="white", fontsize=8)
    
    def update_convergence_plot(self):
        """Update the convergence plot."""
        self.axs[0, 1].cla()
        self.axs[0, 1].set_title('Convergence Plot')
        
        # Generate sample convergence data
        iterations = range(10)
        residuals = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        self.axs[0, 1].semilogy(iterations, residuals, 'b-o', linewidth=2)
        self.axs[0, 1].set_xlabel('Iteration')
        self.axs[0, 1].set_ylabel('Residual Norm (log scale)')
        self.axs[0, 1].grid(True, alpha=0.3)
    
    def update_performance_plot(self):
        """Update the performance metrics plot."""
        self.axs[1, 0].cla()
        self.axs[1, 0].set_title('Performance Metrics')
        
        # Sample performance metrics
        metrics = ['Accuracy', 'Speed', 'Stability', 'Efficiency']
        values = [0.95, 0.87, 0.92, 0.89]
        
        bars = self.axs[1, 0].bar(metrics, values)
        self.axs[1, 0].set_ylabel('Score')
        self.axs[1, 0].set_ylim(0, 1)
        
        # Color bars based on performance
        colors = ['red' if v < 0.8 else 'orange' if v < 0.9 else 'green' for v in values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    def show_dashboard(self):
        """Display the interactive dashboard."""
        plt.show()
    
    def save_dashboard(self, filename: str = "rram_dashboard.png"):
        """Save the dashboard to file."""
        plt.savefig(filename, dpi=300, bbox_inches='tight')


class RRAMAnalyticsVisualizer:
    """
    Analytics visualizer for RRAM system analysis.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (16, 12),
                 style: str = 'seaborn'):
        """
        Initialize the analytics visualizer.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        self.setup_visualization_environment()
    
    def setup_visualization_environment(self):
        """Setup visualization environment."""
        plt.style.use(self.style)
        sns.set_palette("husl")
    
    def visualize_performance_comparison(self,
                                       performance_data: Dict[str, List[float]],
                                       metric_name: str = "Execution Time",
                                       title: str = "Performance Comparison",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize performance comparison between different approaches.
        
        Args:
            performance_data: Dictionary mapping approach names to performance values
            metric_name: Name of the performance metric
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        approaches = list(performance_data.keys())
        data_lists = list(performance_data.values())
        
        # Plot 1: Box plots comparison
        box_parts = ax1.boxplot(data_lists, labels=approaches)
        ax1.set_title(f'{metric_name} Distribution')
        ax1.set_ylabel(metric_name)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Bar plot of means with error bars
        means = [np.mean(data) for data in data_lists]
        stds = [np.std(data) for data in data_lists]
        
        bars = ax2.bar(approaches, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_title(f'{metric_name} by Approach (Mean ± Std)')
        ax2.set_ylabel(metric_name)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean_val:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Violin plot for detailed distribution
        violin_parts = ax3.violinplot(data_lists, positions=range(len(approaches)), 
                                    showmeans=True, showmedians=True)
        ax3.set_title(f'{metric_name} Distribution (Violin Plot)')
        ax3.set_ylabel(metric_name)
        ax3.set_xticks(range(len(approaches)))
        ax3.set_xticklabels(approaches, rotation=45)
        
        # Plot 4: Cumulative distribution
        for i, (approach, data) in enumerate(zip(approaches, data_lists)):
            sorted_data = np.sort(data)
            cumulative_probs = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax4.plot(sorted_data, cumulative_probs, label=approach, linewidth=2)
        
        ax4.set_title('Cumulative Distribution')
        ax4.set_xlabel(metric_name)
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_correlation_matrix(self,
                                   correlation_data: np.ndarray,
                                   variable_names: List[str],
                                   title: str = "Correlation Matrix",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize correlation matrix of different variables in RRAM system.
        
        Args:
            correlation_data: Correlation matrix
            variable_names: Names of variables
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(correlation_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(variable_names)))
        ax.set_yticks(range(len(variable_names)))
        ax.set_xticklabels(variable_names, rotation=45, ha='right')
        ax.set_yticklabels(variable_names)
        
        # Loop over data dimensions and create text annotations
        for i in range(len(variable_names)):
            for j in range(len(variable_names)):
                text = ax.text(j, i, f'{correlation_data[i, j]:.2f}',
                             ha="center", va="center", color="white" if abs(correlation_data[i, j]) > 0.5 else "black")
        
        ax.set_title(title)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_trend_analysis(self,
                               time_series_data: Dict[str, Union[List, np.ndarray]],
                               time_axis: Union[List, np.ndarray],
                               title: str = "Trend Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize trends in time series data for RRAM systems.
        
        Args:
            time_series_data: Dictionary mapping variable names to time series
            time_axis: Time values for x-axis
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(len(time_series_data), 1, figsize=self.figsize, sharex=True)
        if len(time_series_data) == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16)
        
        for i, (var_name, data) in enumerate(time_series_data.items()):
            ax = axes[i]
            
            # Plot the data
            ax.plot(time_axis, data, linewidth=2, label=var_name)
            
            # Add trend line
            if len(data) > 2:
                coeffs = np.polyfit(range(len(data)), data, 1)
                trend_line = np.polyval(coeffs, range(len(data)))
                ax.plot(time_axis, trend_line, '--', color='red', 
                       label=f'Trend (slope: {coeffs[0]:.3f})')
            
            ax.set_ylabel(var_name)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistical info
            mean_val = np.mean(data)
            std_val = np.std(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.3e}\nStd: {std_val:.3e}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[-1].set_xlabel('Time')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_enhanced_visualizer(style: str = 'seaborn') -> Tuple[RRAMDataVisualizer, InteractiveRRAMDashboard, RRAMAnalyticsVisualizer]:
    """
    Create enhanced visualizer components.
    
    Args:
        style: Visualization style to use
        
    Returns:
        Tuple of (data_visualizer, dashboard, analytics_visualizer)
    """
    data_visualizer = RRAMDataVisualizer(style=style)
    dashboard = InteractiveRRAMDashboard()
    analytics_visualizer = RRAMAnalyticsVisualizer(style=style)
    
    return data_visualizer, dashboard, analytics_visualizer


def demonstrate_enhanced_visualization():
    """
    Demonstrate the enhanced visualization capabilities.
    """
    print("Enhanced Visualization Demonstration")
    print("="*50)
    
    # Create visualizers
    data_viz, dashboard, analytics_viz = create_enhanced_visualizer()
    
    print("Created enhanced visualizers")
    
    # Create sample data for visualization
    print("\nCreating sample data...")
    
    # 1. Conductance matrix
    conductance_matrix = np.random.rand(8, 8) * 1e-4 + 0.3 * np.eye(8)
    print(f"Created conductance matrix: {conductance_matrix.shape}")
    
    # 2. Convergence data
    residuals = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    print(f"Created convergence data: {len(residuals)} points")
    
    # 3. Spiking data
    spike_times = {
        0: [0.1, 0.3, 0.7, 0.9],
        1: [0.2, 0.4, 0.8],
        2: [0.15, 0.5, 0.6, 0.85],
        3: [0.25, 0.75]
    }
    print(f"Created spiking data: {len(spike_times)} neurons")
    
    # 4. Material data
    materials_data = {
        'HfO2': np.random.exponential(1e-5, (50, 50)) + 1e-6,
        'TaOx': np.random.exponential(1.5e-5, (50, 50)) + 1e-6,
        'TiO2': np.random.exponential(1.2e-5, (50, 50)) + 1e-6
    }
    print(f"Created material data for: {list(materials_data.keys())}")
    
    # 5. Performance data
    performance_data = {
        'CPU Solver': np.random.exponential(0.01, 100).tolist(),
        'GPU Solver': np.random.exponential(0.005, 100).tolist(),
        'RRAM Hardware': np.random.exponential(0.002, 100).tolist()
    }
    print(f"Created performance data for: {list(performance_data.keys())}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Conductance matrix visualization
    fig1 = data_viz.visualize_conductance_matrix(
        conductance_matrix, 
        title="Sample RRAM Conductance Matrix"
    )
    print("  ✓ Conductance matrix visualization created")
    
    # 2. Convergence visualization
    fig2 = data_viz.visualize_hp_inv_convergence(
        residuals,
        title="HP-INV Convergence Analysis"
    )
    print("  ✓ Convergence visualization created")
    
    # 3. Material characteristics visualization
    fig3 = data_viz.visualize_material_characteristics(
        materials_data,
        property_name="Conductance"
    )
    print("  ✓ Material characteristics visualization created")
    
    # 4. Spiking activity visualization
    fig4 = data_viz.visualize_spiking_activity(
        spike_times,
        duration=1.0,
        title="Neural Spiking Activity"
    )
    print("  ✓ Spiking activity visualization created")
    
    # 5. Performance comparison visualization
    fig5 = analytics_viz.visualize_performance_comparison(
        performance_data,
        metric_name="Execution Time (s)",
        title="Performance Comparison: CPU vs GPU vs RRAM"
    )
    print("  ✓ Performance comparison visualization created")
    
    # Show some plots
    plt.show()
    
    print("\nEnhanced visualization demonstration completed!")
    print("Visualizations include:")
    print("- Conductance matrix heatmap")
    print("- HP-INV convergence analysis") 
    print("- Material property comparisons")
    print("- Spiking neural activity")
    print("- Performance benchmarking")


if __name__ == "__main__":
    demonstrate_enhanced_visualization()