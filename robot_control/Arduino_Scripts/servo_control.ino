// Servo controller for the airbrush robot.
//
// Reads digital signals from the STM32H723 Octopus board and drives
// two servos accordingly.  The Klipper firmware sets output pins
// HIGH/LOW; this Arduino translates those into servo positions.
//
// Wiring (Octopus -> Arduino -> Servo):
//   PB6  ->  A0 (digital input)  ->  Servo 1 (pin 11)  Pump refill enable
//   PB7  ->  A1 (digital input)  ->  Servo 2 (pin 10)  Airbrush needle retract
//
// Klipper commands:
//   SET_PIN PIN=servo_pump_refill VALUE=1       -> Servo 1 to 180 deg
//   SET_PIN PIN=servo_pump_refill VALUE=0       -> Servo 1 to 0 deg
//   SET_PIN PIN=servo_airbrush_needle VALUE=1   -> Servo 2 to 180 deg (needle retracted, spray ON)
//   SET_PIN PIN=servo_airbrush_needle VALUE=0   -> Servo 2 to 0 deg   (needle forward, spray OFF)

#include <Servo.h>

// Servo output pins (Arduino side)
const int servoPin1 = 11;  // pump refill servo
const int servoPin2 = 10;  // airbrush needle servo

// Digital input pins (from Octopus PB6 / PB7)
const int inputPin1 = A0;  // PB6 -> pump refill
const int inputPin2 = A1;  // PB7 -> airbrush needle

Servo Servo1;
Servo Servo2;

void setup() {
    Servo1.attach(servoPin1);
    Servo2.attach(servoPin2);

    pinMode(inputPin1, INPUT);
    pinMode(inputPin2, INPUT);

    // Start with both servos in the OFF position
    Servo1.write(0);
    Servo2.write(0);

    Serial.begin(115200);
}

void loop() {
    int state1 = digitalRead(inputPin1);
    int state2 = digitalRead(inputPin2);

    // Servo 1: pump refill enable
    Servo1.write(state1 == HIGH ? 180 : 0);

    // Servo 2: airbrush needle retract (enables spraying)
    Servo2.write(state2 == HIGH ? 180 : 0);
}
