/*
 * Arduino RRAM Simulator
 * 
 * This sketch simulates Resistive RAM crossbar behavior for matrix operations.
 * It communicates with a Python host via serial protocol to receive commands
 * and send back computation results.
 */

#include <ArduinoJson.h>
#include <SPI.h>

// Configuration constants
#define MATRIX_SIZE 8  // Default 8x8 matrix
#define MAX_MATRIX_SIZE 16 // Maximum supported size

// Global variables for the RRAM simulation
float rram_matrix[MAX_MATRIX_SIZE][MAX_MATRIX_SIZE];
bool connected = false;
int matrix_size = MATRIX_SIZE;

// Simulated device characteristics
float variability = 0.05;
float stuck_fault_prob = 0.01;
float line_resistance = 1.7e-3;

// Serial communication buffer
char serial_buffer[2048];
int buffer_index = 0;

void setup() {
  Serial.begin(115200);
  delay(2000); // Wait for serial connection
  
  // Initialize RRAM matrix to identity
  for(int i = 0; i < MAX_MATRIX_SIZE; i++) {
    for(int j = 0; j < MAX_MATRIX_SIZE; j++) {
      rram_matrix[i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix
    }
  }
  
  connected = true;
  
  // Send ready message
  sendResponse("status", "READY");
}

void loop() {
  // Check for incoming serial data
  while(Serial.available()) {
    char c = Serial.read();
    
    // Process command when newline is received
    if(c == '\n') {
      serial_buffer[buffer_index] = '\0';
      processCommand(serial_buffer);
      buffer_index = 0;
    } else if(buffer_index < sizeof(serial_buffer) - 1) {
      serial_buffer[buffer_index++] = c;
    }
  }
}

void processCommand(char* cmd_str) {
  DynamicJsonDocument doc(2048);
  DeserializationError error = deserializeJson(doc, cmd_str);
  
  if(error) {
    sendResponse("error", "Invalid JSON");
    return;
  }
  
  const char* cmd = doc["cmd"];
  
  if(strcmp(cmd, "INIT") == 0) {
    int size = doc["size"];
    if(size > 0 && size <= MAX_MATRIX_SIZE) {
      matrix_size = size;
      sendResponse("status", "READY");
    } else {
      sendResponse("status", "ERROR");
    }
  }
  else if(strcmp(cmd, "CONFIG") == 0) {
    // Update configuration parameters
    if(doc.containsKey("params")) {
      JsonObject params = doc["params"];
      
      if(params.containsKey("variability")) variability = params["variability"];
      if(params.containsKey("stuck_fault_prob")) stuck_fault_prob = params["stuck_fault_prob"];
      if(params.containsKey("line_resistance")) line_resistance = params["line_resistance"];
    }
    sendResponse("status", "OK");
  }
  else if(strcmp(cmd, "WRITE_MATRIX") == 0) {
    if(doc.containsKey("matrix")) {
      JsonArray matrix = doc["matrix"];
      
      // Copy the matrix data to our RRAM simulation
      for(int i = 0; i < matrix_size; i++) {
        JsonArray row = matrix[i];
        for(int j = 0; j < matrix_size; j++) {
          rram_matrix[i][j] = row[j];
        }
      }
      
      // Apply simulated RRAM effects
      applyDeviceEffects();
      
      sendResponse("status", "OK");
    } else {
      sendResponse("status", "ERROR");
    }
  }
  else if(strcmp(cmd, "READ_MATRIX") == 0) {
    // Send the current RRAM matrix back
    DynamicJsonDocument resp(2048);
    resp["matrix"] = matrix_to_json_array(rram_matrix);
    sendJson(resp);
  }
  else if(strcmp(cmd, "MVM") == 0) {
    // Perform Matrix-Vector Multiplication
    if(doc.containsKey("vector")) {
      JsonArray vector = doc["vector"];
      float input_vec[MATRIX_SIZE];
      
      // Convert JSON vector to array
      for(int i = 0; i < matrix_size; i++) {
        input_vec[i] = vector[i];
      }
      
      // Perform multiplication
      float result[MATRIX_SIZE];
      matrix_vector_multiply(input_vec, result);
      
      // Send result back
      DynamicJsonDocument resp(1024);
      JsonArray result_arr = resp.createNestedArray("result");
      for(int i = 0; i < matrix_size; i++) {
        result_arr.add(result[i]);
      }
      sendJson(resp);
    } else {
      sendResponse("status", "ERROR");
    }
  }
  else if(strcmp(cmd, "INVERT") == 0) {
    // Perform matrix inversion (simplified for demonstration)
    float inv_matrix[MAX_MATRIX_SIZE][MAX_MATRIX_SIZE];
    
    // Copy current RRAM matrix to work array
    for(int i = 0; i < matrix_size; i++) {
      for(int j = 0; j < matrix_size; j++) {
        inv_matrix[i][j] = rram_matrix[i][j];
      }
    }
    
    // Perform inversion (this is a simplified implementation)
    bool success = invert_matrix(inv_matrix, matrix_size);
    
    if(success) {
      DynamicJsonDocument resp(2048);
      resp["result"] = matrix_to_json_array(inv_matrix);
      sendJson(resp);
    } else {
      sendResponse("status", "ERROR");
    }
  }
  else {
    sendResponse("error", "Unknown command");
  }
}

void applyDeviceEffects() {
  // Apply simulated RRAM effects: variability, stuck faults, line resistance
  for(int i = 0; i < matrix_size; i++) {
    for(int j = 0; j < matrix_size; j++) {
      // Apply variability
      float var_effect = 1.0 + (float)random(-100, 100) / 1000.0 * variability;
      rram_matrix[i][j] *= var_effect;
      
      // Apply stuck-at faults (simplified model)
      if((float)random(0, 10000) / 10000.0 < stuck_fault_prob) {
        rram_matrix[i][j] = 0.0;
      }
      
      // Apply line resistance effects
      float line_effect = (float)random(-100, 100) / 1000000.0 * line_resistance;
      rram_matrix[i][j] += line_effect;
    }
  }
}

void matrix_vector_multiply(float* vector, float* result) {
  // Perform matrix-vector multiplication
  for(int i = 0; i < matrix_size; i++) {
    result[i] = 0.0;
    for(int j = 0; j < matrix_size; j++) {
      result[i] += rram_matrix[i][j] * vector[j];
    }
  }
}

bool invert_matrix(float matrix[][MAX_MATRIX_SIZE], int n) {
  // Simple Gaussian elimination for matrix inversion
  // Create augmented matrix [A|I]
  float aug[MAX_MATRIX_SIZE][MAX_MATRIX_SIZE * 2];
  
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      aug[i][j] = matrix[i][j];
    }
  }
  
  // Create identity matrix next to the original
  for(int i = 0; i < n; i++) {
    for(int j = n; j < 2 * n; j++) {
      aug[i][j] = (i == j - n) ? 1.0 : 0.0;
    }
  }
  
  // Perform row operations
  for(int i = 0; i < n; i++) {
    // Find pivot
    if(aug[i][i] == 0) {
      // Look for a non-zero pivot in this column
      bool found = false;
      for(int k = i + 1; k < n; k++) {
        if(aug[k][i] != 0) {
          // Swap rows
          for(int j = 0; j < 2 * n; j++) {
            float temp = aug[i][j];
            aug[i][j] = aug[k][j];
            aug[k][j] = temp;
          }
          found = true;
          break;
        }
      }
      
      if(!found) {
        // Matrix is singular
        return false;
      }
    }
    
    // Scale the row
    float pivot = aug[i][i];
    for(int j = 0; j < 2 * n; j++) {
      aug[i][j] /= pivot;
    }
    
    // Eliminate column
    for(int k = 0; k < n; k++) {
      if(k != i) {
        float factor = aug[k][i];
        for(int j = 0; j < 2 * n; j++) {
          aug[k][j] -= factor * aug[i][j];
        }
      }
    }
  }
  
  // Copy the inverted matrix back
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      matrix[i][j] = aug[i][j + n];
    }
  }
  
  return true;
}

JsonArray matrix_to_json_array(float matrix[][MAX_MATRIX_SIZE]) {
  DynamicJsonDocument temp_doc(2048);
  JsonArray json_matrix = temp_doc.to<JsonArray>();
  
  for(int i = 0; i < matrix_size; i++) {
    JsonArray row = json_matrix.createNestedArray();
    for(int j = 0; j < matrix_size; j++) {
      row.add(matrix[i][j]);
    }
  }
  
  // Move the array to the global document
  return json_matrix;
}

void sendResponse(const char* key, const char* value) {
  DynamicJsonDocument resp(256);
  resp[key] = value;
  sendJson(resp);
}

void sendJson(JsonDocument& doc) {
  serializeJson(doc, Serial);
  Serial.println();
}