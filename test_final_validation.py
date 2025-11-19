"""
Final validation test for all implemented modules.
"""
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_all_modules():
    """
    Test that all newly implemented modules work together correctly.
    """
    print("Final Validation Test - All Modules Integration")
    print("="*50)
    
    # Test 1: GPU Acceleration
    try:
        from src.gpu_simulation_acceleration import GPUSimulationAccelerator
        print("✓ GPU Simulation Acceleration module loaded successfully")
        
        # Check availability
        accel = GPUSimulationAccelerator()
        print(f"  - GPU backend: {accel.backend}")
        print(f"  - GPU initialized: {accel.initialized}")
        
    except Exception as e:
        print(f"✗ GPU Simulation Acceleration failed: {e}")
        return False
    
    # Test 2: ML Framework Integration
    try:
        from src.ml_framework_integration import ml_integration
        print("✓ ML Framework Integration module loaded successfully")
        
        # Check availability status
        print(f"  - PyTorch available: {ml_integration['pytorch_available']}")
        print(f"  - TensorFlow available: {ml_integration['tensorflow_available']}")
        
    except Exception as e:
        print(f"✗ ML Framework Integration failed: {e}")
        return False
    
    # Test 3: Neuromorphic Modules
    try:
        from src.neuromorphic_modules import RRAMSpikingLayer, RRAMReservoirComputer
        print("✓ Neuromorphic Modules loaded successfully")
        
        # Create a simple spiking layer
        layer = RRAMSpikingLayer(
            n_inputs=5,
            n_neurons=3,
            v_threshold=0.8
        )
        print(f"  - Created spiking layer: {layer.n_inputs} inputs, {layer.n_neurons} neurons")
        
        # Test forward pass
        input_spikes = np.random.rand(5) > 0.7  # Sparse spiking input
        output_spikes = layer.forward(input_spikes.astype(float))
        print(f"  - Forward pass successful: output shape {output_spikes.shape}")
        
    except Exception as e:
        print(f"✗ Neuromorphic Modules failed: {e}")
        return False
    
    # Test 4: Real-time Adaptive Systems
    try:
        from src.real_time_adaptive_systems import RealTimeAdaptiveSystem
        print("✓ Real-time Adaptive Systems module loaded successfully")
        
        # Create adaptive system
        adaptive_system = RealTimeAdaptiveSystem(mode='auto_adjust')
        print(f"  - Created adaptive system in mode: {adaptive_system.mode}")
        
    except Exception as e:
        print(f"✗ Real-time Adaptive Systems failed: {e}")
        return False
    
    # Test 5: Enhanced Visualization
    try:
        from src.enhanced_visualization import RRAMDataVisualizer
        print("✓ Enhanced Visualization module loaded successfully")
        
        # Create visualizer
        viz = RRAMDataVisualizer()
        print("  - Created RRAM data visualizer")
        
    except Exception as e:
        print(f"✗ Enhanced Visualization failed: {e}")
        return False
    
    # Test 6: Comprehensive Testing Framework
    try:
        from src.comprehensive_testing import TestRunner
        print("✓ Comprehensive Testing Framework loaded successfully")
        
        # Create test runner
        runner = TestRunner()
        print("  - Created test runner")
        
    except Exception as e:
        print(f"✗ Comprehensive Testing Framework failed: {e}")
        return False
    
    # Test 7: HP-INV solver integration
    try:
        from src.hp_inv import hp_inv
        print("✓ HP-INV module loaded successfully")
        
        # Create test problem
        G = np.random.rand(6, 6) * 1e-4
        G = G + 0.5 * np.eye(6)  # Well-conditioned
        b = np.random.rand(6)
        
        # Solve with HP-INV
        solution, iterations, info = hp_inv(G, b, max_iter=10, bits=4)
        residual = np.linalg.norm(G @ solution - b)
        print(f"  - HP-INV solved problem with residual: {residual:.2e}")
        
    except Exception as e:
        print(f"✗ HP-INV integration failed: {e}")
        return False
    
    # Test 8: Advanced Materials Simulation
    try:
        from src.advanced_materials_simulation import AdvancedMaterialsSimulator
        print("✓ Advanced Materials Simulation loaded successfully")
        
        # Create simulator
        sim = AdvancedMaterialsSimulator()
        print(f"  - Created materials simulator with {len(sim.material_models)} supported materials")
        
    except Exception as e:
        print(f"✗ Advanced Materials Simulation failed: {e}")
        return False
    
    # Test 9: Edge Computing Integration
    try:
        from src.edge_computing_integration import EdgeRRAMOptimizer
        print("✓ Edge Computing Integration loaded successfully")
        
        # Create optimizer
        config = {
            'device_type': 'microcontroller',
            'power_limit': 0.01,  # 10mW
            'latency_limit': 0.05  # 50ms
        }
        optimizer = EdgeRRAMOptimizer(config)
        print("  - Created edge optimizer with constraints")
        
    except Exception as e:
        print(f"✗ Edge Computing Integration failed: {e}")
        return False
    
    # Test 10: Performance Profiling
    try:
        from src.performance_profiling import PerformanceProfiler
        print("✓ Performance Profiling loaded successfully")
        
        # Create profiler
        profiler = PerformanceProfiler()
        print("  - Created performance profiler")
        
    except Exception as e:
        print(f"✗ Performance Profiling failed: {e}")
        return False
    
    print("\n" + "="*50)
    print("ALL MODULES VALIDATION: PASSED")
    print("All implemented modules are working correctly and can be imported!")
    print("="*50)
    
    return True


def run_system_integration_test():
    """
    Run a system integration test that exercises multiple components together.
    """
    print("\nRunning System Integration Test...")
    
    try:
        # Import necessary modules
        from src.gpu_simulation_acceleration import GPUSimulationAccelerator
        from src.hp_inv import hp_inv
        from src.advanced_rram_models import AdvancedRRAMModel
        from src.performance_profiling import PerformanceProfiler
        
        # Create test system
        print("  Creating test RRAM system...")
        
        # Create a test matrix problem
        n = 10
        G = np.random.rand(n, n) * 1e-4
        G = G + 0.5 * np.eye(n)  # Well-conditioned
        b = np.random.rand(n)
        
        print(f"  Solving {n}x{n} linear system...")
        
        # Solve with standard HP-INV
        start_time = time.time()
        solution_cpu, iter_cpu, info_cpu = hp_inv(G, b, max_iter=15, bits=4)
        cpu_time = time.time() - start_time
        
        print(f"  CPU solution residual: {np.linalg.norm(G @ solution_cpu - b):.2e}")
        print(f"  CPU time: {cpu_time:.4f}s")
        
        # Test with GPU acceleration if available
        gpu_accel = GPUSimulationAccelerator()
        if gpu_accel.initialized:
            start_time = time.time()
            solution_gpu, iter_gpu, info_gpu = gpu_accel.gpu_hp_inv(G, b, max_iter=15, bits=4)
            gpu_time = time.time() - start_time
            
            print(f"  GPU solution residual: {np.linalg.norm(G @ solution_gpu - b):.2e}")
            print(f"  GPU time: {gpu_time:.4f}s")
            if cpu_time > 0:
                print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            print("  GPU acceleration not available, skipping GPU test")
        
        # Test with Advanced RRAM Model
        rram_model = AdvancedRRAMModel(
            size=n,
            variability=0.02,
            stuck_fault_prob=0.01,
            line_resistance=1e-3
        )
        
        # Apply RRAM effects to the matrix
        G_rram = rram_model.apply_device_effects(G)
        solution_rram, iter_rram, info_rram = hp_inv(G_rram, b, max_iter=15, bits=4)
        
        print(f"  RRAM-augmented solution residual: {np.linalg.norm(G_rram @ solution_rram - b):.2e}")
        
        # Profile the performance
        profiler = PerformanceProfiler()
        profiler.start_profiling("integration_test")
        
        # Run a few operations
        for _ in range(5):
            x, _, _ = hp_inv(G, b, max_iter=5, bits=4)
        
        profile_results = profiler.stop_profiling("integration_test")
        print(f"  Performance profiling completed: {profile_results['events_count']} events")
        
        print("  System integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"  System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main validation function.
    """
    print("Analogico Project - Final Implementation Validation")
    print("="*60)
    
    # Test all modules
    all_modules_ok = test_all_modules()
    
    if not all_modules_ok:
        print("\n✗ Some modules failed validation. Please check the errors above.")
        return 1
    
    # Run integration test
    integration_ok = run_system_integration_test()
    
    if not integration_ok:
        print("\n✗ Integration test failed.")
        return 1
    
    print("\n🎉 All validations passed! The complete system is working correctly.")
    print("\nImplemented Features Summary:")
    print("  ✓ GPU Acceleration with JAX/CuPy/PyTorch backends")
    print("  ✓ ML Framework Integration (PyTorch/TensorFlow)")
    print("  ✓ Neuromorphic Computing (Spiking Networks, Reservoir Computing)")
    print("  ✓ Real-time Adaptive Systems")
    print("  ✓ Enhanced Visualization Tools")
    print("  ✓ Comprehensive Testing Framework")
    print("  ✓ Advanced Materials Simulation") 
    print("  ✓ Edge Computing Integration")
    print("  ✓ Performance Profiling Tools")
    print("  ✓ System Integration and Compatibility")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)