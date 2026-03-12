// Servo controller for the airbrush robot.
//
// Reads digital signals from the STM32H723 Octopus board and drives
// two servos accordingly.  The Klipper firmware sets output pins
// HIGH/LOW; this Arduino translates those into servo positions.
//
// Wiring (Octopus -> Arduino -> Servo):
//   PB6  ->  A0 (digital input)  ->  Servo 1 (pin 11)  Pump refill valve
//   PB7  ->  A1 (digital input)  ->  Servo 2 (pin 10)  Airbrush needle retract
//
// Klipper commands:
//   SET_PIN PIN=servo_pump_refill VALUE=1       -> Servo 1 valve OPEN  (2000 us)
//   SET_PIN PIN=servo_pump_refill VALUE=0       -> Servo 1 valve CLOSED (2500 us)
//   SET_PIN PIN=servo_airbrush_needle VALUE=1   -> Servo 2 needle retracted, spray ON
//   SET_PIN PIN=servo_airbrush_needle VALUE=0   -> Servo 2 needle forward, spray OFF
//
// Anti-jitter: each input is debounced over DEBOUNCE_MS before the
// servo position is updated.  This prevents glitches from EMI or
// brief signal transitions on the Octopus output pins.

#include <Servo.h>

// --------------- pin assignments ---------------
const int SERVO_PIN_1 = 11;   // pump refill servo
const int SERVO_PIN_2 = 10;   // airbrush needle servo
const int INPUT_PIN_1 = A0;   // PB6 -> pump refill
const int INPUT_PIN_2 = A1;   // PB7 -> airbrush needle

// --------------- calibrated positions (microseconds) ---------------
// Servo 1 – pump refill valve
const int SERVO1_OFF_US = 2500;  // valve closed
const int SERVO1_ON_US  = 2000;  // valve open

// Servo 2 – airbrush needle  (adjust after calibration)
const int SERVO2_OFF_US = 500;   // needle forward, spray OFF
const int SERVO2_ON_US  = 2500;  // needle retracted, spray ON

// --------------- debounce ---------------
// Signal must remain stable for this duration before a state change
// is accepted.  50 ms rejects EMI spikes without adding perceptible
// latency to valve actuation.
const unsigned long DEBOUNCE_MS = 50;

struct Debounce {
    int  stable;              // last accepted (debounced) reading
    int  previous;            // raw reading on the prior loop iteration
    unsigned long changed_at; // millis() when `previous` last flipped
};

Debounce db1 = {LOW, LOW, 0};
Debounce db2 = {LOW, LOW, 0};

Servo servo1;
Servo servo2;

// Returns the debounced state of `pin`, updating `db` in place.
int debounced_read(int pin, Debounce &db) {
    int raw = digitalRead(pin);

    if (raw != db.previous) {
        // Input just changed — restart the stability timer
        db.changed_at = millis();
        db.previous   = raw;
    } else if ((millis() - db.changed_at) >= DEBOUNCE_MS) {
        // Input held steady long enough — accept it
        db.stable = raw;
    }

    return db.stable;
}

void setup() {
    Serial.begin(115200);

    servo1.attach(SERVO_PIN_1);
    servo2.attach(SERVO_PIN_2);

    pinMode(INPUT_PIN_1, INPUT);
    pinMode(INPUT_PIN_2, INPUT);

    // Both servos start in the OFF (safe) position
    servo1.writeMicroseconds(SERVO1_OFF_US);
    servo2.writeMicroseconds(SERVO2_OFF_US);
}

void loop() {
    int state1 = debounced_read(INPUT_PIN_1, db1);
    int state2 = debounced_read(INPUT_PIN_2, db2);

    servo1.writeMicroseconds(state1 == HIGH ? SERVO1_ON_US : SERVO1_OFF_US);
    servo2.writeMicroseconds(state2 == HIGH ? SERVO2_ON_US : SERVO2_OFF_US);
}
