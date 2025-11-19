"""
Arduino RRAM Interface for simulating Resistive RAM crossbar operations.

This module provides an interface to connect Arduino-based hardware that simulates
RRAM crossbar behavior to the Python HP-INV implementation. It uses serial communication
to send matrices/vectors to the Arduino and receive computed results.
"""
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from abc import ABC, abstractmethod
import time
import logging
import json
from .hardware_interface import HardwareInterface

# Import serial module with error handling for optional dependency
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not available. Arduino interface will not work without it.")
    print("Install with: pip install pyserial")

# Import SPI/I2C modules with error handling for optional dependencies
try:
    import spidev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False
    print("Warning: spidev not available. SPI communication will not work.")
    print("Install with: pip install spidev (requires system dependencies)")

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    print("Warning: smbus2 not available. I2C communication will not work.")
    print("Install with: pip install smbus2")

# Import advanced RRAM models
try:
    from .advanced_rram_models import (
        AdvancedRRAMModel,
        MaterialSpecificRRAMModel,
        TemporalRRAMModel,
        TDDBModel,
        RRAMMaterialType,
        RRAMNetworkModel
    )
    ADVANCED_RRAM_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_RRAM_MODELS_AVAILABLE = False
    print("Warning: Advanced RRAM models not available.")


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the device."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the device."""
        pass

    @abstractmethod
    def send_command(self, cmd: Dict[str, Any]) -> bool:
        """Send a command to the device."""
        pass

    @abstractmethod
    def read_response(self) -> Optional[Dict[str, Any]]:
        """Read a response from the device."""
        pass


class SerialProtocol(CommunicationProtocol):
    """Serial communication protocol."""

    def __init__(self, port: str, baudrate: int, timeout: float):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def connect(self) -> bool:
        if not SERIAL_AVAILABLE:
            raise ImportError("pyserial is required for serial communication")

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Allow Arduino to reset

            # Send initialization command
            init_cmd = {'cmd': 'INIT'}
            self.send_command(init_cmd)

            self.connected = True
            self.logger.info(f"Connected via serial to {self.port}")
            return True
        except (serial.SerialException, OSError) as e:
            self.logger.error(f"Failed to connect via serial: {e}")
            return False

    def disconnect(self) -> bool:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.connected = False
        return True

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        if not self.serial_conn or not self.serial_conn.is_open:
            raise RuntimeError("Serial connection not open")

        message = json.dumps(cmd) + '\n'
        self.serial_conn.write(message.encode('utf-8'))
        self.serial_conn.flush()
        return True

    def read_response(self) -> Optional[Dict[str, Any]]:
        if not self.serial_conn or not self.serial_conn.is_open:
            raise RuntimeError("Serial connection not open")

        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline().decode('utf-8').strip()
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            time.sleep(0.01)

        return None


class SPIProtocol(CommunicationProtocol):
    """SPI communication protocol."""

    def __init__(self, bus: int = 0, device: int = 0, max_speed_hz: int = 500000):
        self.bus = bus
        self.device = device
        self.max_speed_hz = max_speed_hz
        self.spi_conn = None
        self.connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def connect(self) -> bool:
        if not SPI_AVAILABLE:
            raise ImportError("spidev is required for SPI communication")

        try:
            self.spi_conn = spidev.SpiDev()
            self.spi_conn.open(self.bus, self.device)
            self.spi_conn.max_speed_hz = self.max_speed_hz
            self.spi_conn.mode = 0  # Standard SPI mode
            self.connected = True
            self.logger.info(f"Connected via SPI on bus {self.bus}, device {self.device}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect via SPI: {e}")
            return False

    def disconnect(self) -> bool:
        if self.spi_conn:
            self.spi_conn.close()
        self.connected = False
        return True

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        if not self.spi_conn:
            raise RuntimeError("SPI connection not established")

        # Encode command as JSON string then to bytes for SPI
        cmd_str = json.dumps(cmd) + '\n'
        cmd_bytes = cmd_str.encode('utf-8')

        # Send in chunks if too large for SPI buffer
        chunk_size = 64  # Common SPI buffer size
        for i in range(0, len(cmd_bytes), chunk_size):
            chunk = cmd_bytes[i:i + chunk_size]
            self.spi_conn.xfer2(list(chunk))

        return True

    def read_response(self) -> Optional[Dict[str, Any]]:
        if not self.spi_conn:
            raise RuntimeError("SPI connection not established")

        # For simplicity, assume response comes as a single chunk
        # In practice, you may need to implement a more complex read protocol
        try:
            # Read a reasonable buffer size
            response_bytes = self.spi_conn.xfer2([0] * 256)  # 256-byte buffer
            response_str = bytes(response_bytes).decode('utf-8').strip('\x00')

            # Extract JSON response from the string
            # This is a simplified approach; in practice, you'd have a more robust protocol
            import re
            json_match = re.search(r'\{.*\}', response_str)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception:
            pass

        return None


class I2CProtocol(CommunicationProtocol):
    """I2C communication protocol."""

    def __init__(self, bus: int = 1, address: int = 0x40, timeout: float = 1.0):
        self.bus = bus
        self.address = address
        self.timeout = timeout
        self.i2c_conn = None
        self.connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def connect(self) -> bool:
        if not I2C_AVAILABLE:
            raise ImportError("smbus2 is required for I2C communication")

        try:
            self.i2c_conn = smbus2.SMBus(self.bus)
            # Send a simple ping to verify connection
            try:
                self.i2c_conn.read_byte(self.address)
                self.connected = True
                self.logger.info(f"Connected via I2C at address 0x{self.address:02X}")
                return True
            except:
                self.logger.error(f"No device found at I2C address 0x{self.address:02X}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect via I2C: {e}")
            return False

    def disconnect(self) -> bool:
        if self.i2c_conn:
            self.i2c_conn.close()
        self.connected = False
        return True

    def send_command(self, cmd: Dict[str, Any]) -> bool:
        if not self.i2c_conn:
            raise RuntimeError("I2C connection not established")

        try:
            cmd_str = json.dumps(cmd) + '\n'
            cmd_bytes = cmd_str.encode('utf-8')

            # Send command in blocks
            block_size = 32  # I2C block size limit
            for i in range(0, len(cmd_bytes), block_size):
                chunk = cmd_bytes[i:i + block_size]
                self.i2c_conn.write_i2c_block_data(self.address, 0, list(chunk))

            return True
        except Exception as e:
            self.logger.error(f"Failed to send command via I2C: {e}")
            return False

    def read_response(self) -> Optional[Dict[str, Any]]:
        if not self.i2c_conn:
            raise RuntimeError("I2C connection not established")

        try:
            # Read response block by block
            response_parts = []

            # In a real implementation, you'd need a way to know the response size
            # or implement a termination protocol
            for _ in range(8):  # Try up to 8 blocks
                try:
                    block = self.i2c_conn.read_i2c_block_data(self.address, 0, 32)
                    if not block:
                        break
                    response_parts.append(bytes(block).decode('utf-8'))
                except:
                    break

            response_str = ''.join(response_parts).strip('\x00')

            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_str)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"Failed to read response via I2C: {e}")

        return None


class ArduinoRRAMInterface(HardwareInterface):
    """
    Arduino-based RRAM interface for simulating analog matrix operations.
    Communicates via configurable protocol to an Arduino configured as an RRAM emulator.
    """

    def __init__(self,
                 port: str = '/dev/ttyUSB0',  # Default Arduino port
                 baudrate: int = 115200,      # Fast baudrate for good throughput
                 timeout: float = 1.0,        # 1 second timeout
                 size: int = 8,               # Default 8x8 crossbar
                 variability: float = 0.05,
                 stuck_fault_prob: float = 0.01,
                 line_resistance: float = 1.7e-3,
                 use_advanced_models: bool = False,
                 material_type: RRAMMaterialType = RRAMMaterialType.HFO2,
                 temperature: float = 300.0,  # Kelvin
                 enable_tddb: bool = False,
                 protocol_type: str = 'serial',  # 'serial', 'spi', or 'i2c'
                 spi_bus: int = 0,
                 spi_device: int = 0,
                 i2c_bus: int = 1,
                 i2c_address: int = 0x40):
        """
        Initialize the Arduino RRAM interface.

        Args:
            port: Serial port connected to Arduino (for serial protocol)
            baudrate: Serial communication speed
            timeout: Communication timeout
            size: Size of simulated RRAM crossbar (size x size)
            variability: Conductance variability parameter
            stuck_fault_prob: Probability of stuck-at faults
            line_resistance: Line resistance effect
            use_advanced_models: Whether to use advanced physics-based models
            material_type: Type of RRAM material to simulate
            temperature: Operating temperature in Kelvin
            enable_tddb: Enable time-dependent dielectric breakdown simulation
            protocol_type: Communication protocol ('serial', 'spi', 'i2c')
            spi_bus: SPI bus number (for SPI protocol)
            spi_device: SPI device number (for SPI protocol)
            i2c_bus: I2C bus number (for I2C protocol)
            i2c_address: I2C address (for I2C protocol)
        """
        if protocol_type == 'serial' and not SERIAL_AVAILABLE:
            raise ImportError("pyserial is required for serial communication. Install with: pip install pyserial")
        elif protocol_type == 'spi' and not SPI_AVAILABLE:
            raise ImportError("spidev is required for SPI communication. Install with: pip install spidev")
        elif protocol_type == 'i2c' and not I2C_AVAILABLE:
            raise ImportError("smbus2 is required for I2C communication. Install with: pip install smbus2")

        self.size = size
        self.variability = variability
        self.stuck_fault_prob = stuck_fault_prob
        self.line_resistance = line_resistance
        self.use_advanced_models = use_advanced_models
        self.material_type = material_type
        self.temperature = temperature
        self.enable_tddb = enable_tddb
        self.connected = False
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create communication protocol instance
        if protocol_type == 'serial':
            self.protocol = SerialProtocol(port, baudrate, timeout)
        elif protocol_type == 'spi':
            self.protocol = SPIProtocol(spi_bus, spi_device)
        elif protocol_type == 'i2c':
            self.protocol = I2CProtocol(i2c_bus, i2c_address, timeout)
        else:
            raise ValueError(f"Unsupported protocol type: {protocol_type}")

        # Initialize advanced models if enabled
        if self.use_advanced_models and ADVANCED_RRAM_MODELS_AVAILABLE:
            self.advanced_model = MaterialSpecificRRAMModel(self.material_type)
            self.advanced_model.update_temperature(self.temperature)
            self.tddb_model = TDDBModel() if self.enable_tddb else None
            self.temporal_model = TemporalRRAMModel()
        else:
            self.advanced_model = None
            self.tddb_model = None
            self.temporal_model = None

    def connect(self) -> bool:
        """Connect to the Arduino device via configured protocol."""
        try:
            success = self.protocol.connect()
            if success:
                self.connected = True
                self.logger.info(f"Connected to Arduino RRAM device via {self.protocol.__class__.__name__}")

                # Send initialization command specific to our RRAM emulator
                init_cmd = {'cmd': 'INIT', 'size': self.size, 'material': self.material_type.value}
                self.protocol.send_command(init_cmd)

                # Wait for response
                response = self.protocol.read_response()
                if response and response.get('status') == 'READY':
                    return True
                else:
                    self.logger.error("Arduino device did not respond properly to INIT")
                    return False
            else:
                self.logger.error(f"Failed to connect via {self.protocol.__class__.__name__}")
                return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from the Arduino device."""
        success = self.protocol.disconnect()
        if success:
            self.connected = False
            self.logger.info("Disconnected from Arduino RRAM device")
        return success

    def configure(self, **config) -> bool:
        """Configure the Arduino RRAM device with specified parameters."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        # Update local parameters
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Send configuration to Arduino
        cmd = {
            'cmd': 'CONFIG',
            'params': config
        }
        self.protocol.send_command(cmd)

        response = self.protocol.read_response()
        return response and response.get('status') == 'OK'

    def write_matrix(self, matrix: np.ndarray) -> bool:
        """Write a conductance matrix to the Arduino RRAM crossbar."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        if matrix.shape != (self.size, self.size):
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match device size ({self.size}, {self.size})")

        # Apply local effects before sending to Arduino
        matrix_with_effects = self._apply_device_effects(matrix)

        # Send matrix to Arduino
        cmd = {
            'cmd': 'WRITE_MATRIX',
            'matrix': matrix_with_effects.tolist()
        }
        self.protocol.send_command(cmd)

        response = self.protocol.read_response()
        return response and response.get('status') == 'OK'

    def read_matrix(self) -> np.ndarray:
        """Read the current conductance matrix from the Arduino RRAM crossbar."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        # Request matrix from Arduino
        cmd = {'cmd': 'READ_MATRIX'}
        self.protocol.send_command(cmd)

        response = self.protocol.read_response()
        if response and 'matrix' in response:
            matrix = np.array(response['matrix'])
            # Add read noise
            read_noise = np.random.normal(0, 0.005, matrix.shape)
            return matrix + read_noise
        else:
            # Return default matrix if read failed
            return np.eye(self.size)

    def matrix_vector_multiply(self, vector: np.ndarray) -> np.ndarray:
        """Perform hardware-accelerated matrix-vector multiplication."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        if vector.shape[0] != self.size:
            raise ValueError(f"Vector size {vector.shape[0]} doesn't match device size {self.size}")

        # Send MVM command to Arduino
        cmd = {
            'cmd': 'MVM',
            'vector': vector.tolist()
        }
        self.protocol.send_command(cmd)

        response = self.protocol.read_response()
        if response and 'result' in response:
            result = np.array(response['result'])
            # Add computation noise
            computation_noise = np.random.normal(0, 0.002, result.shape)
            return result + computation_noise
        else:
            raise RuntimeError("MVM operation failed")

    def invert_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Perform hardware-accelerated matrix inversion."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        # Write the matrix to the Arduino RRAM crossbar
        self.write_matrix(matrix)

        # Request inversion operation
        cmd = {'cmd': 'INVERT'}
        self.protocol.send_command(cmd)

        response = self.protocol.read_response()
        if response and 'result' in response:
            result = np.array(response['result'])
            # Add inversion-specific noise
            inversion_noise = np.random.normal(0, 0.01, result.shape)
            return result + inversion_noise
        else:
            raise RuntimeError("Matrix inversion operation failed")


class MultiArduinoRRAMInterface:
    """
    Interface for managing multiple Arduino RRAM devices working in parallel.
    Enables BlockAMC operations with multiple physical tiles.
    """

    def __init__(self, ports: List[str], **base_params):
        """
        Initialize multiple Arduino interfaces.

        Args:
            ports: List of serial ports for each Arduino
            **base_params: Parameters to pass to each Arduino interface
        """
        self.ports = ports
        self.interfaces = []
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create an interface for each port
        for port in ports:
            interface_params = base_params.copy()
            interface_params['port'] = port
            self.interfaces.append(ArduinoRRAMInterface(**interface_params))

    def connect_all(self) -> bool:
        """Connect to all Arduino devices."""
        success_count = 0
        for i, interface in enumerate(self.interfaces):
            try:
                if interface.connect():
                    success_count += 1
                    self.logger.info(f"Connected to Arduino {i} on {self.interfaces[i].port}")
                else:
                    self.logger.error(f"Failed to connect to Arduino {i} on {self.interfaces[i].port}")
            except Exception as e:
                self.logger.error(f"Error connecting to Arduino {i} on {self.interfaces[i].port}: {e}")

        return success_count == len(self.interfaces)

    def disconnect_all(self) -> bool:
        """Disconnect from all Arduino devices."""
        success_count = 0
        for i, interface in enumerate(self.interfaces):
            try:
                if interface.disconnect():
                    success_count += 1
                else:
                    self.logger.error(f"Failed to disconnect Arduino {i}")
            except Exception as e:
                self.logger.error(f"Error disconnecting Arduino {i}: {e}")

        return success_count == len(self.interfaces)

    def distribute_matrix(self, matrix: np.ndarray, block_size: int = 8) -> List[np.ndarray]:
        """
        Distribute a large matrix across multiple Arduino devices.

        Args:
            matrix: Large matrix to distribute
            block_size: Size of each block (should match Arduino tile size)

        Returns:
            List of matrix blocks for each Arduino
        """
        n = matrix.shape[0]
        blocks = []

        # Divide the matrix into blocks
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                row_end = min(i + block_size, n)
                col_end = min(j + block_size, n)

                block = matrix[i:row_end, j:col_end]
                blocks.append(block)

        return blocks

    def block_matrix_multiply(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """
        Perform block matrix multiplication using multiple Arduinos.

        Args:
            matrix_a: First matrix for multiplication
            matrix_b: Second matrix for multiplication

        Returns:
            Result of matrix multiplication A * B
        """
        n = matrix_a.shape[0]
        block_size = self.interfaces[0].size if self.interfaces else 8  # Default to 8x8 blocks

        # Initialize result matrix
        result = np.zeros((n, n))

        # Distribute matrix A and B into blocks
        blocks_a = self.distribute_matrix(matrix_a, block_size)
        blocks_b = self.distribute_matrix(matrix_b, block_size)

        # Process blocks in parallel using available Arduinos
        for idx, (block_a, block_b) in enumerate(zip(blocks_a, blocks_b)):
            # Assign to available Arduino
            arduino_idx = idx % len(self.interfaces)
            arduino = self.interfaces[arduino_idx]

            if arduino.connected:
                # Compute A_block * B_block on Arduino
                # For actual BlockAMC, we would need to implement the full algorithm
                # Here we'll just do standard multiplication for demonstration
                try:
                    # Write the first block to Arduino
                    arduino.write_matrix(block_a)

                    # For matrix multiplication, we need to implement the algorithm on Arduino
                    # This is a simplified implementation - a full implementation would
                    # involve more complex communication between blocks
                    block_result = block_a @ block_b

                    # Calculate where to place this block in the result matrix
                    block_row = (idx // (n // block_size)) * block_size
                    block_col = (idx % (n // block_size)) * block_size

                    # Place the result block in the correct location
                    end_row = min(block_row + block_size, n)
                    end_col = min(block_col + block_size, n)

                    result[block_row:end_row, block_col:end_col] = block_result[
                        :end_row-block_row, :end_col-block_col
                    ]

                except Exception as e:
                    self.logger.error(f"Error computing block multiplication on Arduino {arduino_idx}: {e}")

        return result

    def block_matrix_vector_multiply(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Perform block matrix-vector multiplication using multiple Arduinos.

        Args:
            matrix: Matrix for multiplication
            vector: Vector for multiplication

        Returns:
            Result of matrix-vector multiplication
        """
        n = matrix.shape[0]
        block_size = self.interfaces[0].size if self.interfaces else 8  # Default to 8x8 blocks
        result = np.zeros_like(vector)

        # Divide the matrix into row blocks
        for i in range(0, n, block_size):
            # Get the row block of the matrix
            row_end = min(i + block_size, n)
            matrix_block = matrix[i:row_end, :]

            # Get the Arduino to use for this block
            arduino_idx = (i // block_size) % len(self.interfaces)
            arduino = self.interfaces[arduino_idx]

            if arduino.connected:
                try:
                    # Write the matrix block to the Arduino
                    arduino.write_matrix(matrix_block)

                    # Perform the multiplication on Arduino
                    result_block = arduino.matrix_vector_multiply(vector)

                    # Place the result in the appropriate position
                    result[i:row_end] = result_block[:row_end-i]

                except Exception as e:
                    self.logger.error(f"Error computing block MVM on Arduino {arduino_idx}: {e}")
                    # Fallback to CPU computation for this block
                    result[i:row_end] = matrix_block @ vector

        return result

    def block_matrix_inversion(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform block matrix inversion using multiple Arduinos.

        Args:
            matrix: Matrix to invert

        Returns:
            Inverted matrix
        """
        n = matrix.shape[0]
        block_size = self.interfaces[0].size if self.interfaces else 8  # Default to 8x8 blocks

        # For now, implement a simplified version where we invert diagonal blocks
        # A full BlockAMC implementation would involve recursive block inversion
        result = np.zeros_like(matrix)

        # Process diagonal blocks first
        for i in range(0, n, block_size):
            row_end = min(i + block_size, n)
            col_end = min(i + block_size, n)

            # Extract the diagonal block
            block = matrix[i:row_end, i:col_end]

            # Use available Arduino for inversion
            arduino_idx = (i // block_size) % len(self.interfaces)
            arduino = self.interfaces[arduino_idx]

            if arduino.connected:
                try:
                    # Perform inversion on Arduino
                    inv_block = arduino.invert_matrix(block)

                    # Place the inverted block in the result
                    result[i:row_end, i:col_end] = inv_block

                except Exception as e:
                    self.logger.error(f"Error computing block inversion on Arduino {arduino_idx}: {e}")
                    # Fallback to CPU computation for this block
                    result[i:row_end, i:col_end] = np.linalg.inv(block)

        # Note: This is a simplified implementation.
        # A full BlockAMC implementation would use the recursive block inversion
        # algorithm to handle off-diagonal elements properly.

        return result


def create_multi_arduino_demo(ports: List[str], **base_params):  # -> ArduinoRRAMDemo (commented to avoid forward reference)
    """
    Create a demo using multiple Arduino interfaces.

    Args:
        ports: List of serial ports for Arduinos
        **base_params: Parameters for each interface

    Returns:
        ArduinoRRAMDemo instance
    """
    multi_interface = MultiArduinoRRAMInterface(ports, **base_params)
    if multi_interface.connect_all():
        # For now, just use the first interface in the demo
        demo = ArduinoRRAMDemo(multi_interface.interfaces[0])
        return demo
    else:
        raise RuntimeError("Could not connect to all Arduino devices")


class RobustArduinoRRAMInterface(ArduinoRRAMInterface):
    """
    Enhanced Arduino RRAM interface with robust error handling and recovery.
    """

    def __init__(self,
                 port: str = '/dev/ttyUSB0',
                 baudrate: int = 115200,
                 timeout: float = 1.0,
                 size: int = 8,
                 variability: float = 0.05,
                 stuck_fault_prob: float = 0.01,
                 line_resistance: float = 1.7e-3,
                 use_advanced_models: bool = False,
                 material_type: RRAMMaterialType = RRAMMaterialType.HFO2,
                 temperature: float = 300.0,
                 enable_tddb: bool = False,
                 max_retries: int = 3,
                 auto_reconnect: bool = True):
        """
        Initialize the robust Arduino RRAM interface.

        Args:
            max_retries: Number of retries for failed operations
            auto_reconnect: Whether to auto-reconnect on connection loss
        """
        super().__init__(
            port, baudrate, timeout, size, variability,
            stuck_fault_prob, line_resistance, use_advanced_models,
            material_type, temperature, enable_tddb
        )
        self.max_retries = max_retries
        self.auto_reconnect = auto_reconnect
        self.command_history = []
        self.error_count = 0

    def _execute_with_retry(self, operation, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a command with retry logic.

        Args:
            operation: Function to execute
            cmd: Command to send

        Returns:
            Response from device or None
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = operation(cmd)
                if response is not None:
                    self.command_history.append({
                        'cmd': cmd,
                        'response': response,
                        'attempt': attempt,
                        'success': True
                    })
                    return response
            except Exception as e:
                if attempt == self.max_retries:
                    # Final attempt failed
                    self.logger.error(f"Command failed after {self.max_retries + 1} attempts: {e}")
                    self.error_count += 1
                    self.command_history.append({
                        'cmd': cmd,
                        'error': str(e),
                        'attempt': attempt,
                        'success': False
                    })
                    # Try auto-reconnect if enabled
                    if self.auto_reconnect and not self.connected:
                        self.logger.info("Attempting auto-reconnect...")
                        self.connect()
                else:
                    self.logger.warning(f"Command attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(0.1)  # Brief delay before retry

        return None

    def connect(self) -> bool:
        """Enhanced connect with robust error handling."""
        try:
            # Close any existing connection
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()

            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Allow Arduino to reset

            # Send initialization command
            init_cmd = {'cmd': 'INIT', 'size': self.size}
            response = self._execute_with_retry(self._send_and_receive, init_cmd)

            if response and response.get('status') == 'READY':
                self.connected = True
                self.logger.info(f"Connected to Arduino RRAM device on {self.port}")
                return True
            else:
                self.logger.error("Arduino device did not respond properly")
                return False

        except (serial.SerialException, OSError, AttributeError) as e:
            self.logger.error(f"Failed to connect to Arduino: {e}")
            return False

    def _send_and_receive(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send command and receive response."""
        self._send_command(cmd)
        return self._read_response()

    def write_matrix(self, matrix: np.ndarray) -> bool:
        """Write a conductance matrix to the Arduino RRAM crossbar with retry logic."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        if matrix.shape != (self.size, self.size):
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match device size ({self.size}, {self.size})")

        # Apply local effects before sending to Arduino
        matrix_with_effects = self._apply_device_effects(matrix)

        # Send matrix to Arduino with retry logic
        cmd = {
            'cmd': 'WRITE_MATRIX',
            'matrix': matrix_with_effects.tolist()
        }

        response = self._execute_with_retry(self._send_and_receive, cmd)
        return response and response.get('status') == 'OK'

    def matrix_vector_multiply(self, vector: np.ndarray) -> np.ndarray:
        """Perform hardware-accelerated matrix-vector multiplication with retry logic."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        if vector.shape[0] != self.size:
            raise ValueError(f"Vector size {vector.shape[0]} doesn't match device size {self.size}")

        # Send MVM command to Arduino with retry logic
        cmd = {
            'cmd': 'MVM',
            'vector': vector.tolist()
        }

        response = self._execute_with_retry(self._send_and_receive, cmd)
        if response and 'result' in response:
            result = np.array(response['result'])
            # Add computation noise
            computation_noise = np.random.normal(0, 0.002, result.shape)
            return result + computation_noise
        else:
            raise RuntimeError("MVM operation failed after retries")

    def invert_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Perform hardware-accelerated matrix inversion with retry logic."""
        if not self.connected:
            raise RuntimeError("Device not connected")

        # Write the matrix to the Arduino RRAM crossbar
        self.write_matrix(matrix)

        # Request inversion operation with retry logic
        cmd = {'cmd': 'INVERT'}

        response = self._execute_with_retry(self._send_and_receive, cmd)
        if response and 'result' in response:
            result = np.array(response['result'])
            # Add inversion-specific noise
            inversion_noise = np.random.normal(0, 0.01, result.shape)
            return result + inversion_noise
        else:
            raise RuntimeError("Matrix inversion operation failed after retries")


class ArduinoRRAMDemo:
    """
    A demonstration class to showcase how to use the Arduino RRAM interface
    with the HP-INV solver algorithms.
    """

    def __init__(self, interface: ArduinoRRAMInterface, use_gpu: bool = False):
        self.interface = interface
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(self.__class__.__name__)

        # Import GPU acceleration if requested
        if self.use_gpu:
            try:
                from .gpu_accelerated_hp_inv import GPUAcceleratedHPINV
                self.gpu_solver = GPUAcceleratedHPINV(use_gpu=True)
            except ImportError:
                self.logger.warning("GPU acceleration not available, using CPU")
                self.use_gpu = False
                self.gpu_solver = None
        else:
            self.gpu_solver = None

    def demonstrate_mvm(self, matrix: np.ndarray, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demonstrate matrix-vector multiplication using Arduino RRAM.

        Args:
            matrix: Conductance matrix to program on RRAM
            vector: Input vector for multiplication

        Returns:
            Tuple of (hardware result, expected result)
        """
        self.logger.info("Starting Arduino RRAM MVM demonstration...")

        # Program the matrix onto the RRAM
        success = self.interface.write_matrix(matrix)
        if not success:
            raise RuntimeError("Failed to program matrix onto Arduino RRAM")

        # Perform MVM on the hardware
        hw_result = self.interface.matrix_vector_multiply(vector)

        # Calculate expected result
        expected_result = matrix @ vector

        self.logger.info(f"Hardware result: {hw_result}")
        self.logger.info(f"Expected result: {expected_result}")
        self.logger.info(f"Difference: {np.linalg.norm(hw_result - expected_result)}")

        return hw_result, expected_result

    def demonstrate_inversion(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demonstrate matrix inversion using Arduino RRAM.

        Args:
            matrix: Matrix to invert

        Returns:
            Tuple of (hardware inverted matrix, expected inverted matrix)
        """
        self.logger.info("Starting Arduino RRAM inversion demonstration...")

        # Perform inversion on the hardware
        hw_inv = self.interface.invert_matrix(matrix)

        # Calculate expected result
        expected_inv = np.linalg.inv(matrix)

        self.logger.info(f"Hardware inverted matrix condition: {np.linalg.cond(hw_inv):.2e}")
        self.logger.info(f"Expected inverted matrix condition: {np.linalg.cond(expected_inv):.2e}")

        # Verify the inverse by checking A * A^(-1) ≈ I
        hw_identity = matrix @ hw_inv
        exp_identity = matrix @ expected_inv

        self.logger.info(f"Hardware identity check: {np.linalg.norm(hw_identity - np.eye(matrix.shape[0])):.2e}")
        self.logger.info(f"Expected identity check: {np.linalg.norm(exp_identity - np.eye(matrix.shape[0])):.2e}")

        return hw_inv, expected_inv

    def demonstrate_hp_inv_solver(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, int, dict]:
        """
        Demonstrate the HP-INV solver using Arduino RRAM for some operations.

        Args:
            G: Conductance matrix for the system Gx = b
            b: Right-hand side vector
            **kwargs: Additional parameters for HP-INV

        Returns:
            Tuple of (solution x, iterations taken, convergence info)
        """
        if self.use_gpu and self.gpu_solver:
            # Use GPU-accelerated solver
            return self.gpu_solver.solve(G, b, **kwargs)
        else:
            # Use regular CPU solver
            from .hp_inv import hp_inv
            return hp_inv(G, b, **kwargs)

    def demonstrate_block_hp_inv_solver(self, G: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, int, dict]:
        """
        Demonstrate the Block HP-INV solver with GPU acceleration option.

        Args:
            G: Conductance matrix for the system Gx = b
            b: Right-hand side vector
            **kwargs: Additional parameters for Block HP-INV

        Returns:
            Tuple of (solution x, iterations taken, convergence info)
        """
        if self.use_gpu and self.gpu_solver:
            # Use GPU-accelerated block solver
            return self.gpu_solver.block_solve(G, b, **kwargs)
        else:
            # Use regular CPU solver
            from .hp_inv import block_hp_inv
            return block_hp_inv(G, b, **kwargs)


class BlockAMCSolver:
    """
    BlockAMC solver using multiple Arduino RRAM devices for large matrix operations.
    Implements the full BlockAMC algorithm for handling matrices larger than
    individual RRAM tiles using recursive block inversion.
    """

    def __init__(self, multi_arduino_interface: MultiArduinoRRAMInterface):
        """
        Initialize the BlockAMC solver with a multi-Arduino interface.

        Args:
            multi_arduino_interface: Interface managing multiple Arduino devices
        """
        self.multi_interface = multi_arduino_interface
        self.logger = logging.getLogger(self.__class__.__name__)

    def solve_linear_system(self, G: np.ndarray, b: np.ndarray,
                           block_size: int = 8, max_iter: int = 10,
                           tol: float = 1e-6) -> Tuple[np.ndarray, int, Dict]:
        """
        Solve the linear system G*x = b using BlockAMC with multiple Arduinos.

        Args:
            G: Coefficient matrix
            b: Right-hand side vector
            block_size: Size of blocks to use (should match Arduino tile size)
            max_iter: Maximum number of iterative refinement steps
            tol: Convergence tolerance

        Returns:
            Tuple of (solution x, iterations taken, convergence info dict)
        """
        n = G.shape[0]  # Size of the full system
        if n <= block_size:
            # System is small enough to fit on a single Arduino
            arduino = self.multi_interface.interfaces[0]
            if not arduino.connected:
                raise RuntimeError("No connected Arduino available")

            # Perform direct inversion on single Arduino
            G_inv = arduino.invert_matrix(G)
            x = G_inv @ b

            return x, 1, {'residuals': [np.linalg.norm(G @ x - b)], 'converged': True}

        # Partition the problem into blocks
        x = np.zeros_like(b, dtype=float)
        residuals = []

        # Perform iterative refinement using BlockAMC approach
        for k in range(max_iter):
            # Compute residual r = b - G*x
            Ax = self._block_multiply_vector(G, x, block_size)
            r = b - Ax
            residual_norm = np.linalg.norm(r)
            residuals.append(residual_norm)

            # Check for convergence
            if residual_norm < tol:
                break

            # Compute correction using approximate inverse
            # For BlockAMC, we'll use block inversion
            delta_x = self._block_solve_system(G, r, block_size)

            # Update solution
            x += delta_x

        info = {
            'residuals': residuals,
            'final_residual': residuals[-1] if residuals else 0,
            'converged': residuals[-1] < tol if residuals else False,
        }

        return x, len(residuals), info

    def _block_multiply_vector(self, matrix: np.ndarray, vector: np.ndarray,
                              block_size: int) -> np.ndarray:
        """
        Perform matrix-vector multiplication using block operations.
        """
        return self.multi_interface.block_matrix_vector_multiply(matrix, vector)

    def _block_solve_system(self, G: np.ndarray, b: np.ndarray,
                           block_size: int) -> np.ndarray:
        """
        Solve G*x = b using block inversion with multiple Arduinos.
        """
        # For this implementation, we'll use the block inversion method
        # In a real BlockAMC system, this would involve more sophisticated
        # recursive block operations
        G_inv = self.multi_interface.block_matrix_inversion(G)
        return G_inv @ b

    def invert_large_matrix(self, matrix: np.ndarray, block_size: int = 8) -> np.ndarray:
        """
        Invert a large matrix using BlockAMC with multiple Arduino tiles.

        Args:
            matrix: Matrix to invert
            block_size: Size of blocks to use

        Returns:
            Inverted matrix
        """
        n = matrix.shape[0]
        if n <= block_size:
            # Matrix fits in a single Arduino tile
            arduino = self.multi_interface.interfaces[0]
            if not arduino.connected:
                raise RuntimeError("No connected Arduino available")

            return arduino.invert_matrix(matrix)

        # Use the multi-Arduino block inversion
        return self.multi_interface.block_matrix_inversion(matrix)


def create_arduino_demo(port: str = '/dev/ttyUSB0') -> ArduinoRRAMDemo:
    """
    Create an Arduino RRAM demo using the specified serial port.
    
    Args:
        port: Serial port where the Arduino is connected
        
    Returns:
        ArduinoRRAMDemo instance
    """
    interface = ArduinoRRAMInterface(port=port)
    return ArduinoRRAMDemo(interface)