### 7. Material-Specific RRAM Models (`src.advanced_rram_models`)

#### `HfO2RRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

HfO₂-based RRAM model with specific switching characteristics based on oxygen vacancy migration (VCM mechanism).

**Parameters:**
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_state: str` - Initial state "high" or "low" (default: "high")

**Methods:**
- `apply_voltage(voltage, duration)` - Apply voltage with HfO₂-specific VCM switching
- `get_conductance()` - Get current conductance
- `get_device_state()` - Get comprehensive device state

**Example:**
```python
from src.advanced_rram_models import HfO2RRAMModel

# Create an HfO2 device model
device = HfO2RRAMModel(device_area=1e-12, temperature=320)

# Apply a SET voltage pulse (formation of oxygen vacancies)
result = device.apply_voltage(-1.8, 1e-8)  # -1.8V, 10ns pulse
print(f"Switched: {result['switched']}, New resistance: {result['resistance_after']:.2e}Ω")
```

#### `TaOxRRAMModel(device_area=1e-12, temperature=300.0, initial_state="high", mechanism="VCM")`

TaOₓ-based RRAM model with VCM or ECM switching mechanism options.

**Parameters:**
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_state: str` - Initial state "high" or "low" (default: "high")
- `mechanism: str` - Switching mechanism "VCM" or "ECM" (default: "VCM")

#### `TiO2RRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

TiO₂-based RRAM model with specific switching characteristics based on titanium interstitials and oxygen vacancies.

#### `NiOxRRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

NiOₓ-based RRAM model with ECM mechanism and metal filament formation.

#### `CoOxRRAMModel(device_area=1e-12, temperature=300.0, initial_state="high")`

CoOₓ-based RRAM model with mixed VCM/ECM behavior.

#### `AdvancedRRAMModel(material=RRAMMaterial.HFO2, device_area=1e-12, temperature=300.0, initial_resistance_state="high", ecm_vcm_ratio=0.5)`

Main class that provides access to material-specific models with unified interface.

**Parameters:**
- `material: RRAMMaterial` - Type of RRAM material (default: RRAMMaterial.HFO2)
- `device_area: float` - Physical area of the device in m² (default: 1e-12)
- `temperature: float` - Operating temperature in Kelvin (default: 300.0)
- `initial_resistance_state: str` - Initial state "high" or "low" (default: "high")
- `ecm_vcm_ratio: float` - Ratio of ECM to VCM for mixed-mechanism materials (default: 0.5)

**Example:**
```python
from src.advanced_rram_models import AdvancedRRAMModel, RRAMMaterial

# Create devices with different materials
materials = ["HfO2", "TaOx", "TiO2", "NiOx", "CoOx"]

for material in materials:
    device = AdvancedRRAMModel(material=getattr(RRAMMaterial, material.replace("Ox", "O2")))
    state = device.get_device_state()
    print(f"{material}: {state['resistance']:.2e}Ω, state: {state['state']}")
```