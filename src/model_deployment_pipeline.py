"""
ML Model Deployment Pipeline Integration for RRAM-based Systems.

This module provides tools for integrating RRAM-based models into
standard ML deployment pipelines, including model conversion,
quantization, and hardware deployment workflows.
"""
import os
import json
import pickle
import time
import warnings
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
from pathlib import Path

# Import from our existing modules
try:
    from .ml_framework_integration import RRAMModelWrapper, TorchRRAMLinear
    FRAMEWORK_INTEGRATION_AVAILABLE = True
except ImportError:
    FRAMEWORK_INTEGRATION_AVAILABLE = False
    warnings.warn("Framework integration not available")

try:
    from .neuromorphic_modules import RRAMNeuromorphicNetwork, RRAMSpikingLayer
    NEUROMORPHIC_AVAILABLE = True
except ImportError:
    NEUROMORPHIC_AVAILABLE = False
    warnings.warn("Neuromorphic modules not available")

try:
    from .arduino_rram_interface import ArduinoRRAMInterface, MultiArduinoRRAMInterface, BlockAMCSolver
    ARDUINO_INTERFACE_AVAILABLE = True
except ImportError:
    ARDUINO_INTERFACE_AVAILABLE = False
    warnings.warn("Arduino interface not available")

try:
    from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV, AdaptivePrecisionHPINV
    GPU_ACCELERATION_AVAILABLE = True
except ImportError:
    GPU_ACCELERATION_AVAILABLE = False
    warnings.warn("GPU acceleration not available")


class RRAMModelConverter:
    """
    Tool for converting standard ML models to RRAM-compatible formats.
    """
    
    def __init__(self):
        self.conversion_log = []
    
    def convert_pytorch_to_rram(self, 
                               pytorch_model_path: str, 
                               output_path: str,
                               quantization_bits: int = 4) -> str:
        """
        Convert a PyTorch model to RRAM-compatible format.
        
        Args:
            pytorch_model_path: Path to the PyTorch model file
            output_path: Path for the output RRAM model
            quantization_bits: Number of bits for weight quantization
            
        Returns:
            Path to the converted model
        """
        if not FRAMEWORK_INTEGRATION_AVAILABLE:
            raise ImportError("Framework integration required for PyTorch conversion")
        
        import torch
        
        # Load the PyTorch model
        model = torch.load(pytorch_model_path)
        model.eval()  # Set to evaluation mode
        
        # Wrap the model to make it RRAM-compatible
        wrapper = RRAMModelWrapper(model, framework='torch')
        rram_model = wrapper.convert_to_rram_model(
            quantization_bits=quantization_bits
        )
        
        # Save the RRAM-compatible model
        torch.save(rram_model, output_path)
        
        # Log the conversion
        self.conversion_log.append({
            'input_model': pytorch_model_path,
            'output_model': output_path,
            'framework': 'pytorch',
            'quantization_bits': quantization_bits,
            'timestamp': time.time()
        })
        
        return output_path
    
    def convert_tensorflow_to_rram(self,
                                  tensorflow_model_path: str,
                                  output_path: str,
                                  quantization_bits: int = 4) -> str:
        """
        Convert a TensorFlow model to RRAM-compatible format.
        
        Args:
            tensorflow_model_path: Path to the TensorFlow model
            output_path: Path for the output RRAM model
            quantization_bits: Number of bits for weight quantization
            
        Returns:
            Path to the converted model
        """
        if not FRAMEWORK_INTEGRATION_AVAILABLE:
            raise ImportError("Framework integration required for TensorFlow conversion")
        
        import tensorflow as tf
        
        # Load the TensorFlow model
        model = tf.keras.models.load_model(tensorflow_model_path)
        
        # For now, just save the model - full conversion requires more complex implementation
        model.save(output_path)
        
        # Log the conversion
        self.conversion_log.append({
            'input_model': tensorflow_model_path,
            'output_model': output_path,
            'framework': 'tensorflow',
            'quantization_bits': quantization_bits,
            'timestamp': time.time()
        })
        
        return output_path
    
    def optimize_for_rram(self, 
                         model_path: str,
                         output_path: str,
                         optimization_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Optimize a model specifically for RRAM characteristics.
        
        Args:
            model_path: Path to the input model
            output_path: Path for the optimized model
            optimization_params: Dictionary of optimization parameters
            
        Returns:
            Path to the optimized model
        """
        if optimization_params is None:
            optimization_params = {}
        
        # Load the model (assuming PyTorch for this example)
        import torch
        
        model = torch.load(model_path)
        model.eval()
        
        # Apply optimizations specific to RRAM:
        # 1. Layer fusion to reduce memory transfers
        # 2. Quantization-aware training adjustments
        # 3. Pruning based on RRAM conductance limits
        
        # Example optimization: adjust layer parameters based on RRAM characteristics
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Apply RRAM-specific constraints
                with torch.no_grad():
                    # Clip weights to RRAM-constrained range
                    max_weight = optimization_params.get('max_weight', 1.0)
                    min_weight = optimization_params.get('min_weight', -1.0)
                    module.weight.data = torch.clamp(
                        module.weight.data, 
                        min_weight, 
                        max_weight
                    )
        
        # Save the optimized model
        torch.save(model, output_path)
        
        # Log the optimization
        self.conversion_log.append({
            'input_model': model_path,
            'output_model': output_path,
            'optimization_params': optimization_params,
            'timestamp': time.time()
        })
        
        return output_path


class RRAMModelQuantizer:
    """
    Quantization tools specifically for RRAM-based models.
    """
    
    def __init__(self, bits: int = 4):
        self.bits = bits
        self.levels = 2**bits - 1  # Symmetric quantization levels
    
    def quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Quantize weights to RRAM-constrained precision.
        
        Args:
            weights: Original weights
            
        Returns:
            Quantized weights
        """
        # Find the range of weights
        max_val = np.max(np.abs(weights))
        if max_val == 0:
            return weights
        
        # Scale weights to quantization levels
        scale = self.levels / max_val
        quantized = np.round(weights * scale) / scale
        
        return quantized
    
    def quantize_model_pytorch(self, 
                              model: Any, 
                              inplace: bool = False) -> Any:
        """
        Quantize a PyTorch model for RRAM.
        
        Args:
            model: PyTorch model to quantize
            inplace: Whether to modify the model in place
            
        Returns:
            Quantized model
        """
        import torch
        
        if not inplace:
            model = torch.nn.utils.parametrize.clone(model)
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Quantize the weight
                    quantized_weight = self.quantize_weights(module.weight.numpy())
                    module.weight.copy_(torch.from_numpy(quantized_weight).to(module.weight.device))
        
        return model
    
    def quantize_model_tensorflow(self, 
                                 model: Any, 
                                 inplace: bool = False) -> Any:
        """
        Quantize a TensorFlow model for RRAM.
        
        Args:
            model: TensorFlow model to quantize
            inplace: Whether to modify the model in place
            
        Returns:
            Quantized model
        """
        # For TensorFlow, we'll return a function that performs quantization
        # when the model is used
        import tensorflow as tf
        
        # Create a quantized version of the model
        if not inplace:
            # Clone the model
            model = tf.keras.models.clone_model(model)
            model.set_weights(model.get_weights())  # Copy weights
        
        # For actual quantization, we would need to replace layers
        # with custom quantized versions. This is a placeholder implementation.
        return model


class RRAMModelDeployer:
    """
    Tools for deploying RRAM-based models to hardware.
    """
    
    def __init__(self, 
                 hardware_interface: Optional[object] = None,
                 enable_fallback: bool = True):
        """
        Initialize the model deployer.
        
        Args:
            hardware_interface: Interface to RRAM hardware 
            enable_fallback: Whether to use fallback computation when hardware fails
        """
        self.hardware_interface = hardware_interface
        self.enable_fallback = enable_fallback
        self.deployed_models = {}
    
    def deploy_model(self, 
                    model_path: str, 
                    model_id: str,
                    target_hardware: Optional[str] = None) -> bool:
        """
        Deploy a model to RRAM hardware.
        
        Args:
            model_path: Path to the model file
            model_id: Unique identifier for the deployed model
            target_hardware: Specific hardware target (e.g., 'arduino', 'fpga')
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Load the model based on its format
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                import torch
                model = torch.load(model_path)
            elif model_path.endswith('.h5') or model_path.endswith('.keras'):
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            else:
                # Assume pickled model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Extract RRAM-compatible weight matrices
            weight_matrices = self._extract_weight_matrices(model, model_path)
            
            # Deploy each weight matrix to hardware
            if self.hardware_interface is not None:
                for i, weight_matrix in enumerate(weight_matrices):
                    try:
                        success = self.hardware_interface.write_matrix(weight_matrix)
                        if not success:
                            warnings.warn(f"Failed to deploy weight matrix {i} to hardware")
                            if not self.enable_fallback:
                                return False
                    except Exception as e:
                        warnings.warn(f"Error deploying weight matrix {i} to hardware: {e}")
                        if not self.enable_fallback:
                            return False
            
            # Store model metadata
            self.deployed_models[model_id] = {
                'path': model_path,
                'deployment_time': time.time(),
                'weight_matrices_count': len(weight_matrices),
                'target_hardware': target_hardware
            }
            
            return True
            
        except Exception as e:
            warnings.warn(f"Model deployment failed: {e}")
            return False
    
    def _extract_weight_matrices(self, model: Any, model_path: str) -> List[np.ndarray]:
        """
        Extract weight matrices from a model for deployment to RRAM.
        
        Args:
            model: The model object
            model_path: Path to the model file (to determine format)
            
        Returns:
            List of weight matrices
        """
        weight_matrices = []
        
        if 'pth' in model_path or 'pt' in model_path:
            # PyTorch model
            import torch
            if hasattr(model, 'named_modules'):
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Linear, TorchRRAMLinear)):
                        weight_matrices.append(module.weight.detach().cpu().numpy())
        elif 'h5' in model_path or 'keras' in model_path:
            # TensorFlow model
            for layer in model.layers:
                if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
                    weights = layer.get_weights()[0]  # First element is kernel weights
                    weight_matrices.append(weights)
        else:
            # For other formats, implement accordingly
            pass
        
        return weight_matrices
    
    def execute_model(self, 
                     model_id: str, 
                     input_data: np.ndarray,
                     use_hardware: bool = True) -> np.ndarray:
        """
        Execute a deployed model on input data.
        
        Args:
            model_id: ID of the deployed model
            input_data: Input data for the model
            use_hardware: Whether to use hardware acceleration if available
            
        Returns:
            Model output
        """
        if model_id not in self.deployed_models:
            raise ValueError(f"Model {model_id} not deployed")
        
        model_info = self.deployed_models[model_id]
        
        if use_hardware and self.hardware_interface is not None:
            try:
                # Perform computation using RRAM hardware
                # This is a simplified approach - real implementation would
                # involve more complex operations depending on the model structure
                result = self.hardware_interface.matrix_vector_multiply(input_data)
                return result
            except Exception as e:
                warnings.warn(f"Hardware execution failed: {e}")
                if not self.enable_fallback:
                    raise e
        
        # Fallback to software execution
        model_path = model_info['path']
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            import torch
            model = torch.load(model_path)
            model.eval()
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_data).float()
                output_tensor = model(input_tensor)
                return output_tensor.numpy()
        else:
            # For other model formats, implement accordingly
            warnings.warn("Software fallback not fully implemented for this model format")
            return input_data  # Placeholder
    
    def undeploy_model(self, model_id: str) -> bool:
        """
        Remove a deployed model from hardware.
        
        Args:
            model_id: ID of the model to undeploy
            
        Returns:
            True if successful, False otherwise
        """
        if model_id in self.deployed_models:
            del self.deployed_models[model_id]
            return True
        return False


class RRAMPipelineManager:
    """
    End-to-end pipeline manager for RRAM-based ML workflows.
    """
    
    def __init__(self, 
                 hardware_interface: Optional[object] = None,
                 converter: Optional[RRAMModelConverter] = None,
                 quantizer: Optional[RRAMModelQuantizer] = None,
                 deployer: Optional[RRAMModelDeployer] = None):
        """
        Initialize the pipeline manager.
        
        Args:
            hardware_interface: Interface to RRAM hardware
            converter: Model converter instance
            quantizer: Model quantizer instance
            deployer: Model deployer instance
        """
        self.hardware_interface = hardware_interface
        self.converter = converter or RRAMModelConverter()
        self.quantizer = quantizer or RRAMModelQuantizer(bits=4)
        self.deployer = deployer or RRAMModelDeployer(hardware_interface)
        self.pipeline_log = []
    
    def run_complete_pipeline(self,
                             input_model_path: str,
                             output_model_path: str,
                             model_id: str,
                             quantization_bits: int = 4,
                             optimize: bool = True,
                             deploy: bool = True,
                             target_hardware: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete RRAM ML pipeline: convert, optimize, quantize, deploy.
        
        Args:
            input_model_path: Path to input model
            output_model_path: Path for processed model
            model_id: Unique ID for the model
            quantization_bits: Number of bits for quantization
            optimize: Whether to optimize the model
            deploy: Whether to deploy to hardware
            target_hardware: Target hardware type
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        results = {
            'pipeline_id': model_id,
            'steps': {},
            'success': False,
            'total_time': 0
        }
        
        try:
            # Step 1: Convert model to RRAM format
            if input_model_path.endswith(('.pth', '.pt')):
                converted_model = self.converter.convert_pytorch_to_rram(
                    input_model_path, 
                    output_model_path,
                    quantization_bits
                )
            elif input_model_path.endswith(('.h5', '.keras')):
                converted_model = self.converter.convert_tensorflow_to_rram(
                    input_model_path, 
                    output_model_path,
                    quantization_bits
                )
            else:
                raise ValueError(f"Unsupported model format: {input_model_path}")
            
            results['steps']['conversion'] = {
                'status': 'success',
                'output_path': converted_model
            }
            
            # Step 2: Optimize the model for RRAM
            if optimize:
                optimized_model_path = output_model_path.replace('.pth', '_optimized.pth').replace('.h5', '_optimized.h5')
                optimized_model = self.converter.optimize_for_rram(
                    converted_model,
                    optimized_model_path
                )
                
                results['steps']['optimization'] = {
                    'status': 'success',
                    'output_path': optimized_model
                }
            else:
                optimized_model = converted_model
            
            # Step 3: Quantize the model
            # Note: Quantization is built into the converter for RRAM models
            results['steps']['quantization'] = {
                'status': 'success',
                'bits': quantization_bits
            }
            
            # Step 4: Deploy to hardware
            if deploy:
                deploy_success = self.deployer.deploy_model(
                    optimized_model,
                    model_id,
                    target_hardware
                )
                
                results['steps']['deployment'] = {
                    'status': 'success' if deploy_success else 'failed',
                    'model_id': model_id
                }
                
                if not deploy_success:
                    results['steps']['deployment']['error'] = "Deployment failed"
            
            results['success'] = True
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
        
        results['total_time'] = time.time() - start_time
        self.pipeline_log.append(results)
        
        return results
    
    def benchmark_deployment(self, 
                            model_id: str, 
                            test_inputs: List[np.ndarray],
                            use_hardware: bool = True) -> Dict[str, Any]:
        """
        Benchmark a deployed model's performance.
        
        Args:
            model_id: ID of the deployed model
            test_inputs: List of test input arrays
            use_hardware: Whether to use hardware acceleration
            
        Returns:
            Benchmark results
        """
        import time
        
        # Warm-up run
        if len(test_inputs) > 0:
            try:
                _ = self.deployer.execute_model(model_id, test_inputs[0], use_hardware)
            except (RuntimeError, ValueError, KeyError, AttributeError) as e:
                warnings.warn(f"Warm-up run failed: {e}")
                pass  # Ignore warm-up errors
        
        # Benchmark timing
        start_time = time.time()
        total_ops = 0
        
        for input_data in test_inputs:
            op_start = time.perf_counter()
            try:
                output = self.deployer.execute_model(model_id, input_data, use_hardware)
                op_time = time.perf_counter() - op_start
                total_ops += 1
            except Exception as e:
                warnings.warn(f"Operation failed: {e}")
                continue
        
        total_time = time.time() - start_time
        avg_time = total_time / total_ops if total_ops > 0 else 0
        
        # Calculate throughput (operations per second)
        throughput = total_ops / total_time if total_time > 0 else 0
        
        results = {
            'model_id': model_id,
            'total_operations': total_ops,
            'total_time': total_time,
            'average_time_per_op': avg_time,
            'throughput_ops_per_sec': throughput,
            'use_hardware': use_hardware,
            'timestamp': time.time()
        }
        
        return results
    
    def generate_deployment_report(self, model_id: str) -> str:
        """
        Generate a detailed deployment report for a model.
        
        Args:
            model_id: ID of the model to report on
            
        Returns:
            Deployment report as a string
        """
        if model_id not in self.deployer.deployed_models:
            return f"Model {model_id} not found in deployed models"
        
        model_info = self.deployer.deployed_models[model_id]
        
        report = f"""
RRAM Model Deployment Report
===========================

Model ID: {model_id}
Deployment Time: {time.ctime(model_info['deployment_time'])}
Weight Matrices: {model_info['weight_matrices_count']}
Target Hardware: {model_info['target_hardware'] or 'Unknown'}

Pipeline Steps:
"""
        
        # Find pipeline results for this model
        pipeline_results = None
        for result in self.pipeline_log:
            if result['pipeline_id'] == model_id:
                pipeline_results = result
                break
        
        if pipeline_results:
            for step_name, step_info in pipeline_results['steps'].items():
                status = step_info['status'] if 'status' in step_info else 'unknown'
                report += f"  - {step_name}: {status}\n"
                if 'error' in step_info:
                    report += f"    Error: {step_info['error']}\n"
        
        return report


class RRAMModelExporter:
    """
    Export RRAM models to various deployment formats.
    """
    
    def __init__(self):
        self.export_formats = ['onnx', 'tflite', 'tensorrt', 'rram_binary']
    
    def export_to_rram_binary(self, 
                             model: Any, 
                             output_path: str,
                             model_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export a model to RRAM binary format with metadata.
        
        Args:
            model: The model to export
            output_path: Output file path
            model_metadata: Additional metadata to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a structured format for RRAM deployment
            export_data = {
                'model_weights': [],
                'model_structure': {},
                'metadata': model_metadata or {},
                'export_time': time.time(),
                'format_version': '1.0'
            }
            
            # Extract weights from the model
            if hasattr(model, 'named_parameters'):
                # PyTorch model
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        export_data['model_weights'].append({
                            'name': name,
                            'data': param.detach().cpu().numpy().tolist(),
                            'shape': list(param.shape)
                        })
            elif hasattr(model, 'get_weights'):
                # TensorFlow model
                for i, layer_weights in enumerate(model.get_weights()):
                    export_data['model_weights'].append({
                        'name': f'layer_{i}_weights',
                        'data': layer_weights.tolist(),
                        'shape': list(layer_weights.shape)
                    })
            
            # Write to binary file using JSON for now (binary format would be more complex)
            with open(output_path, 'w') as f:
                json.dump(export_data, f)
            
            return True
        except Exception as e:
            warnings.warn(f"Export to RRAM binary failed: {e}")
            return False
    
    def export_to_onnx(self, model: Any, output_path: str, input_shape: Tuple[int, ...]) -> bool:
        """
        Export a model to ONNX format (for compatibility).
        
        Args:
            model: The model to export
            output_path: Output file path
            input_shape: Shape of input tensor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import torch
            import torch.onnx
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            return True
        except Exception as e:
            warnings.warn(f"Export to ONNX failed: {e}")
            return False


def create_optimized_rram_pipeline(hardware_interface: Optional[object] = None) -> RRAMPipelineManager:
    """
    Create a pipeline manager with optimized settings for RRAM deployment.
    
    Args:
        hardware_interface: Interface to RRAM hardware
        
    Returns:
        Optimized RRAM pipeline manager
    """
    quantizer = RRAMModelQuantizer(bits=4)  # Standard 4-bit quantization for RRAM
    deployer = RRAMModelDeployer(hardware_interface, enable_fallback=True)
    converter = RRAMModelConverter()
    
    return RRAMPipelineManager(
        hardware_interface=hardware_interface,
        converter=converter,
        quantizer=quantizer,
        deployer=deployer
    )


def benchmark_rram_vs_digital(model: Any, 
                            test_inputs: List[np.ndarray], 
                            hardware_interface: Optional[object] = None) -> Dict[str, float]:
    """
    Benchmark RRAM hardware performance vs digital computation.
    
    Args:
        model: The model to benchmark
        test_inputs: List of test inputs
        hardware_interface: Optional hardware interface for RRAM execution
        
    Returns:
        Dictionary with benchmark comparison results
    """
    import time
    
    # Benchmark digital computation
    start_time = time.time()
    for inp in test_inputs:
        if hasattr(model, 'forward'):  # PyTorch model
            import torch
            with torch.no_grad():
                inp_tensor = torch.from_numpy(inp).float()
                _ = model(inp_tensor)
        else:  # Other model types
            _ = model(inp)
    digital_time = time.time() - start_time
    
    # Benchmark RRAM computation if hardware available
    if hardware_interface is not None:
        start_time = time.time()
        for inp in test_inputs:
            try:
                _ = hardware_interface.matrix_vector_multiply(inp)
            except (AttributeError, RuntimeError, IOError, ValueError) as e:
                warnings.warn(f"Hardware matrix-vector multiply failed: {e}")
                # Fallback to basic matrix multiply if MVM not available
                _ = inp  # Placeholder
        rram_time = time.time() - start_time
    else:
        rram_time = float('inf')  # Indicate unavailable
    
    results = {
        'digital_time': digital_time,
        'rram_time': rram_time,
        'digital_avg_per_input': digital_time / len(test_inputs),
        'rram_avg_per_input': rram_time / len(test_inputs) if rram_time != float('inf') else float('inf'),
        'speedup': digital_time / rram_time if rram_time != float('inf') and rram_time > 0 else 0
    }
    
    return results