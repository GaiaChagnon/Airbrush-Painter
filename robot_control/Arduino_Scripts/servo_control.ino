// Servo controller for the airbrush robot.
//
// Reads digital signals from the STM32H723 Octopus board and drives
// two servos accordingly.  The Klipper firmware sets output pins
// HIGH/LOW; this Arduino translates those into servo positions.
//
// Wiring (Octopus -> Arduino -> Servo):
//   PB6  ->  A0 (digital input)  ->  Servo 1 (pin 11)  Airbrush needle retract
//   PB7  ->  A1 (digital input)  ->  Servo 2 (pin 10)  Pump refill valve
//   GND  ->  GND (shared ground between Octopus and Arduino)
//
// Klipper commands:
//   SET_PIN PIN=servo_airbrush_needle VALUE=1   -> Servo 1 needle retracted, spray ON
//   SET_PIN PIN=servo_airbrush_needle VALUE=0   -> Servo 1 needle forward, spray OFF
//   SET_PIN PIN=servo_pump_refill VALUE=1       -> Servo 2 valve OPEN  (SERVO2_ON_US)
//   SET_PIN PIN=servo_pump_refill VALUE=0       -> Servo 2 valve CLOSED (SERVO2_OFF_US)
//
// Debug: open Serial Monitor at 115200 baud to see input state changes.

#include <Servo.h>

// --------------- pin assignments ---------------
const int SERVO_PIN_1 = 11;   // airbrush needle servo
const int SERVO_PIN_2 = 10;   // pump refill servo
const int INPUT_PIN_1 = A0;   // PB6 -> airbrush needle
const int INPUT_PIN_2 = A1;   // PB7 -> pump refill

// --------------- servo pulse range (must match servo_setup.ino) ---------------
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2500;

// --------------- calibrated positions (microseconds) ---------------
// Servo 1 -- airbrush needle
const int SERVO1_OFF_US = 2500;  // needle forward, spray OFF
const int SERVO1_ON_US  = 2100;  // needle retracted, spray ON

// Servo 2 -- pump refill valve  (adjust after calibration)
const int SERVO2_OFF_US = 2300;  // valve closed
const int SERVO2_ON_US  = 2300;  // valve open

// --------------- debounce ---------------
// 50 ms rejects EMI spikes without adding perceptible latency.
const unsigned long DEBOUNCE_MS = 50;

// --------------- timing ---------------
// Minimum interval between loop iterations (ms).
// Keeps CPU utilization low; 10 ms still gives <60 ms response time
// with debounce included.
const unsigned long LOOP_INTERVAL_MS = 10;

struct Debounce {
    int  stable;              // last accepted (debounced) reading
    int  previous;            // raw reading on the prior iteration
    unsigned long changed_at; // millis() when `previous` last flipped
};

Debounce db1 = {LOW, LOW, 0};
Debounce db2 = {LOW, LOW, 0};

Servo servo1;
Servo servo2;

int last_state1 = -1;  // force initial update (no valid pin state is -1)
int last_state2 = -1;

// Returns the debounced state of `pin`, updating `db` in place.
int debounced_read(int pin, Debounce &db) {
    int raw = digitalRead(pin);

    if (raw != db.previous) {
        db.changed_at = millis();
        db.previous   = raw;
    } else if ((millis() - db.changed_at) >= DEBOUNCE_MS) {
        db.stable = raw;
    }

    return db.stable;
}

void setup() {
    Serial.begin(115200);

    servo1.attach(SERVO_PIN_1, SERVO_MIN_US, SERVO_MAX_US);
    servo2.attach(SERVO_PIN_2, SERVO_MIN_US, SERVO_MAX_US);

    // INPUT_PULLDOWN gives a defined LOW when Octopus pins are not driven
    // (before Klipper init or if the cable is disconnected).  Octopus
    // push-pull outputs override the weak pull-down when active.
    pinMode(INPUT_PIN_1, INPUT_PULLDOWN);
    pinMode(INPUT_PIN_2, INPUT_PULLDOWN);

    servo1.writeMicroseconds(SERVO1_OFF_US);
    servo2.writeMicroseconds(SERVO2_OFF_US);

    Serial.println("Servo controller ready");
    Serial.print("  Servo 1 (needle): OFF=");
    Serial.print(SERVO1_OFF_US);
    Serial.print("us  ON=");
    Serial.print(SERVO1_ON_US);
    Serial.println("us");
    Serial.print("  Servo 2 (valve):  OFF=");
    Serial.print(SERVO2_OFF_US);
    Serial.print("us  ON=");
    Serial.print(SERVO2_ON_US);
    Serial.println("us");
}

void loop() {
    int state1 = debounced_read(INPUT_PIN_1, db1);
    int state2 = debounced_read(INPUT_PIN_2, db2);

    if (state1 != last_state1) {
        int us = (state1 == HIGH) ? SERVO1_ON_US : SERVO1_OFF_US;
        servo1.writeMicroseconds(us);
        last_state1 = state1;
        Serial.print("Needle: ");
        Serial.print(state1 == HIGH ? "RETRACTED" : "FORWARD");
        Serial.print("  (");
        Serial.print(us);
        Serial.println("us)");
    }

    if (state2 != last_state2) {
        int us = (state2 == HIGH) ? SERVO2_ON_US : SERVO2_OFF_US;
        servo2.writeMicroseconds(us);
        last_state2 = state2;
        Serial.print("Valve:  ");
        Serial.print(state2 == HIGH ? "OPEN" : "CLOSED");
        Serial.print("  (");
        Serial.print(us);
        Serial.println("us)");
    }

    delay(LOOP_INTERVAL_MS);
}
