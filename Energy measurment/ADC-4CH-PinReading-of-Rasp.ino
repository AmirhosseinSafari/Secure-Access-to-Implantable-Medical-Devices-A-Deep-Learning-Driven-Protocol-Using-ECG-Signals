#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <SPI.h>
#include <SD.h>

// Initialize ADS1015 and variables
Adafruit_ADS1015 ads1015;
float V_Rassp = 0;
float I_Rassp = 0;
float POWER_Rassp = 0;

const int chipSelect = 4;   // SD card chip select pin
const int signalPin = 7;    // Pin connected to Raspberry Pi (through voltage divider)
bool recording = false;

void setup(void) {
    pinMode(signalPin, INPUT_PULLUP);  // Enable internal pull-up resistor on signal pin
    Serial.begin(9600);

    Serial.println("Hello!");
    Serial.println("Getting single-ended readings from AIN0..3");
    Serial.println("ADC Range: +/- 6.144V (1 bit = 3mV)");

    // Initialize ADS1015
    ads1015.begin();

    // Initialize SD card
    Serial.print("Initializing SD card...");
    if (!SD.begin(chipSelect)) {
        Serial.println("SD card initialization failed!");
        while (true);  // Stop the program if SD initialization fails
    }
    Serial.println("SD card initialized.");
}

void loop(void) {
    // Read signal pin state (inverted logic due to pull-up resistor)
    int signalState = digitalRead(signalPin);

    // Start or stop recording based on signal pin state
    if (signalState == LOW) {
        recording = true;  // Start recording when pin is pulled low
    } else {
        recording = false; // Stop recording when pin is high
    }

    // If recording is true, proceed with reading and writing data
    if (recording) {
        int16_t adc0, adc1;

        // Read ADC channels
        adc0 = ads1015.readADC_SingleEnded(0);
        V_Rassp = adc0 / 115.28;  // Voltage in Volts

        adc1 = ads1015.readADC_SingleEnded(1);
        I_Rassp = (adc1 / 340.4) - 0.01176;  // Current in Amps

        POWER_Rassp = V_Rassp * I_Rassp;  // Calculate power

        // Print readings to Serial Monitor
        Serial.print("V_Rassp: ");
        Serial.println(V_Rassp);
        Serial.print("I_Rassp: ");
        Serial.println(I_Rassp);
        Serial.print("POWER_Rass(W): ");
        Serial.println(POWER_Rassp);

        // Write data to SD card
        writeDataToSD(V_Rassp, I_Rassp, POWER_Rassp);
    }

    delay(1000); // Delay to prevent overloading the system
}

void writeDataToSD(float voltage, float current, float power) {
    File dataFile = SD.open("data.csv", FILE_WRITE); // Open file in append mode
    if (dataFile) {
        // Write data in CSV format: Voltage, Current, Power
        dataFile.print(voltage);  // Write Voltage
        dataFile.print(", ");     // Comma separator
        dataFile.print(current);  // Write Current
        dataFile.print(", ");     // Comma separator
        dataFile.println(power);  // Write Power with newline
        dataFile.close();         // Close the file
        Serial.println("Data written to SD card.");
    } else {
        Serial.println("Error opening file!");
    }
}
