#include <Servo.h>

const int servoPin1 = 11;
const int servoPin2 = 10;

Servo Servo1;
Servo Servo2;

void setup() {
  Servo1.attach(servoPin1, 500, 2500);  // adjust to your servo
  Servo2.attach(servoPin2, 500, 2500);

  Serial.begin(115200);
}

void loop() {
  Serial.println("upper");
  Servo1.writeMicroseconds(2500);
  delay(5000);

  Serial.println("lower");
  Servo1.writeMicroseconds(2100);
  delay(5000);
}