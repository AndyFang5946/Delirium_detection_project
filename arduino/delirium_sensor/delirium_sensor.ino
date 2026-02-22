/*
 * Delirium Detection Sensor Node
 * Reads data from MAX30102 (PPG) and LM35 (Temperature) sensors
 * Sends data over serial communication to Python application
 * 
 * Hardware connections:
 * MAX30102:
 *   - SDA to Arduino SDA (A4 on Uno, 20 on Mega)
 *   - SCL to Arduino SCL (A5 on Uno, 21 on Mega)
 *   - VCC to 3.3V
 *   - GND to GND
 * 
 * LM35:
 *   - Signal to A0
 *   - VCC to 5V
 *   - GND to GND
 */

#include <Wire.h>
#include "MAX30105.h"

// Sensor objects
MAX30105 particleSensor;
const int lm35Pin = A1;  // LM35 analog pin

// Configuration
const int SAMPLING_RATE = 100;  // Hz
const long SAMPLE_INTERVAL = 1000 / SAMPLING_RATE;  // milliseconds
const int BUFFER_SIZE = 100;

// Buffers for PPG data
uint32_t redBuffer[BUFFER_SIZE];
uint32_t irBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// Timing
unsigned long lastSampleTime = 0;
unsigned long lastTransmitTime = 0;
const long TRANSMIT_INTERVAL = 1000;  // Send data every 1 second

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  delay(100);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found. Check connections.");
    while (1);  // Halt if sensor not found
  }
  
  // Configure MAX30102
  byte ledBrightness = 60;  // 0=Off to 255=50mA
  byte sampleAverage = 8;      // 1, 2, 4, 8, 16, 32
  byte ledMode = 2;           // 1=Red only, 2=Red+IR
  int sampleRate = 100;        // 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411;        // 69, 118, 215, 411
  int adcRange = 16384;         // 2048, 4096, 8192, 16384
  
  particleSensor.setup();
  
  // Initialize LM35
  pinMode(lm35Pin, INPUT);
  
  Serial.println("SYSTEM_READY");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Read sensors at fixed interval
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    
    // Read PPG data
    redBuffer[bufferIndex] = particleSensor.getRed();
    irBuffer[bufferIndex] = particleSensor.getIR();
    
    // Update buffer index
    bufferIndex++;
    if (bufferIndex >= BUFFER_SIZE) {
      bufferIndex = 0;
    }
  }
  
  // Transmit data periodically
  if (currentTime - lastTransmitTime >= TRANSMIT_INTERVAL) {
    lastTransmitTime = currentTime;
    transmitData();
  }
  
  // Process incoming commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "GET_DATA") {
      transmitData();
    } else if (command == "GET_BUFFER") {
      transmitBuffer();
    } else if (command == "STATUS") {
      Serial.println("STATUS_OK");
    }
  }
}

void transmitData() {
  /*
   * Transmit the most recent sensor reading in JSON format
   */
  int lastIndex = (bufferIndex - 1 + BUFFER_SIZE) % BUFFER_SIZE;
  
  // Read temperature
  int tempRaw = analogRead(lm35Pin);
  float temperature = (tempRaw * 4.88) / 10;  // Convert to °C
  
  // Send JSON data with separate red and IR channels
  Serial.print("{\"type\":\"data\",\"red\":");
  Serial.print(redBuffer[lastIndex]);
  Serial.print(",\"ir\":");
  Serial.print(irBuffer[lastIndex]);
  Serial.print(",\"temp\":");
  Serial.print(temperature, 2);
  Serial.print(",\"timestamp\":");
  Serial.print(millis());
  Serial.println("}");
}

void transmitBuffer() {
  /*
   * Transmit the entire buffer of PPG data
   */
  Serial.println("{\"type\":\"buffer\",\"data\":[");
  
  for (int i = 0; i < BUFFER_SIZE; i++) {
    Serial.print("{\"red\":");
    Serial.print(redBuffer[i]);
    Serial.print(",\"ir\":");
    Serial.print(irBuffer[i]);
    Serial.print("}");
    
    if (i < BUFFER_SIZE - 1) {
      Serial.println(",");
    } else {
      Serial.println();
    }
  }
  
  Serial.println("]}");
}
