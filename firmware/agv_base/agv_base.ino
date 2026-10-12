// agv_base.ino - Arduino GIGA R1 low-level controller for the ball-following AGV.
//
// Hardware (same circuit as before):
//   BTS7960 drive motor : RPWM 5, LPWM 6, R_EN 7, L_EN 8
//   Steering servo      : signal 9 (driven with mbed PwmOut, 50 Hz)
//   Wheel encoder       : channel A on pin 2 (3.3 V supply!), optional channel B on pin 3
//
// The Pi talks to this board over USB serial. One command per line:
//   C <counts_per_s> <steer_deg>   closed-loop speed command (+ = forward), steering (+ = LEFT)
//   D <duty_pct> <steer_deg>       open-loop duty command (used for calibration)
//   G <kp> <ki> <kff> <u0> <accel> speed-loop gains: duty% per cps, duty% per cps*s,
//                                  feed-forward duty% per cps, static (deadband) duty%,
//                                  max acceleration in counts/s^2
//   T <center_deg> <dir> <max_deg> [gain]
//                                  servo trim: servo angle for straight wheels, +1/-1,
//                                  wheel steering limit (deg), servo deg per wheel deg
//   P <servo_deg>                  CALIBRATION: put the servo at this raw angle (0..180),
//                                  ignoring trim and limits. Any C, D or T command ends it.
//   R                              reset encoder count to 0
//   X                              clear a stall fault
//   ?                              print settings
// If no C or D command arrives for WATCHDOG_MS, the drive stops (steering holds).
//
// Telemetry every 20 ms:
//   E,<ms>,<count>,<cps>,<duty_pct>,<steer_deg>,<flags>
//   flags: bit0 = stall fault, bit1 = watchdog stop, bit2 = open-loop mode
// Every other line starts with '#'.

#include "mbed.h"
#include "USB/PluggableUSBSerial.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------- pins ----------------
const int PIN_RPWM  = 5;
const int PIN_LPWM  = 6;
const int PIN_REN   = 7;
const int PIN_LEN   = 8;
const int PIN_SERVO = 9;
const int PIN_ENC_A = 2;
const int PIN_ENC_B = 3;
const bool ENC_HAS_B = false;   // channel B (yellow) gives no pulses on this motor -> not used.
                                // Direction is taken from the drive command instead.

// ---------------- hard safety limits (the Pi cannot exceed these) ----------------
const float MAX_DUTY_PCT      = 70.0f;   // 4S battery: keep at 70
const float STEER_HARD_MAX    = 40.0f;   // absolute wheel steering limit in degrees
const float STEER_RATE_DPS    = 200.0f;  // servo slew limit
const float DUTY_SLEW_PCT_S   = 250.0f;  // output slew limit (protects the gearbox)
const unsigned long WATCHDOG_MS = 300;
const unsigned long PERIOD_US   = 20000; // 50 Hz control loop
const float STALL_DUTY_PCT    = 45.0f;   // a stall is: this much duty...
const float STALL_CPS         = 20.0f;   // ...but slower than this...
const unsigned long STALL_MS  = 1500;    // ...for this long

// ---------------- settings changeable over serial ----------------
float servoCenterDeg = 90.0f;
float servoDir       = 1.0f;
float servoGain      = 1.0f;    // servo degrees per wheel degree (steering linkage ratio)
bool  servoRaw       = false;   // calibration mode (P command)
float servoRawDeg    = 90.0f;
float steerMaxDeg    = 25.0f;
float kp    = 0.01f;    // duty% per (counts/s) of error
float ki    = 0.02f;    // duty% per (counts/s * s)
float kff   = 0.015f;   // duty% per (counts/s) of target
float u0    = 40.0f;    // duty% needed to overcome static friction
float accel = 3000.0f;  // counts/s^2

// ---------------- state ----------------
enum Mode { MODE_SPEED, MODE_DUTY };
Mode mode = MODE_SPEED;

volatile long encCount = 0;
volatile int  encDir   = 1;     // used when there is no channel B

float targetCps = 0, refCps = 0, measCps = 0, integ = 0;
float targetDuty = 0, outDuty = 0;
float targetSteer = 0, curSteer = 0;
long  lastCount = 0;
bool  stallFault = false, watchdogStop = true;
unsigned long lastCmdMs = 0, stallStartMs = 0, lastTickUs = 0;

mbed::PwmOut *servoPwm = nullptr;
char lineBuf[96];
int  lineLen = 0;

// ---------------- encoder ----------------
void encIsr() {
  if (ENC_HAS_B) encCount += digitalRead(PIN_ENC_B) ? -1 : 1;
  else           encCount += encDir;
}

// ---------------- outputs ----------------
void applyDrive(float duty) {
  int pwm = (int)(fabsf(duty) * 255.0f / 100.0f + 0.5f);
  if (pwm <= 0) {
    analogWrite(PIN_RPWM, 0);
    analogWrite(PIN_LPWM, 0);
    digitalWrite(PIN_REN, LOW);
    digitalWrite(PIN_LEN, LOW);
    return;
  }
  if (pwm > 255) pwm = 255;
  encDir = duty > 0 ? 1 : -1;    // single-channel encoder: direction follows the drive
  digitalWrite(PIN_REN, HIGH);
  digitalWrite(PIN_LEN, HIGH);
  if (duty > 0) { analogWrite(PIN_LPWM, 0); analogWrite(PIN_RPWM, pwm); }
  else          { analogWrite(PIN_RPWM, 0); analogWrite(PIN_LPWM, pwm); }
}

void writeServo(float steerDeg) {
  float a = servoRaw ? servoRawDeg : servoCenterDeg + servoDir * servoGain * steerDeg;
  if (a < 0) a = 0;
  if (a > 180) a = 180;
  int us = 544 + (int)lroundf(a * (2400.0f - 544.0f) / 180.0f);   // Servo-library scale
  if (servoPwm) servoPwm->pulsewidth_us(us);
}

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

float stepToward(float cur, float target, float maxStep) {
  float d = target - cur;
  if (d > maxStep) return cur + maxStep;
  if (d < -maxStep) return cur - maxStep;
  return target;
}

// ---------------- control loop ----------------
void controlStep(float dt, unsigned long nowMs) {
  // measured speed
  noInterrupts();
  long c = encCount;
  interrupts();
  float inst = (c - lastCount) / dt;
  lastCount = c;
  measCps = 0.6f * measCps + 0.4f * inst;

  // watchdog
  watchdogStop = (nowMs - lastCmdMs > WATCHDOG_MS);
  if (watchdogStop) { targetCps = 0; targetDuty = 0; }

  float duty = 0;
  if (mode == MODE_DUTY) {
    refCps = 0; integ = 0;
    duty = targetDuty;
  } else {
    refCps = stepToward(refCps, targetCps, accel * dt);
    if (fabsf(refCps) < 1.0f) {
      refCps = 0; integ = 0; duty = 0;
    } else {
      float err = refCps - measCps;
      float s = refCps > 0 ? 1.0f : -1.0f;
      float ff = s * u0 + kff * refCps;
      float newInteg = clampf(integ + ki * err * dt, -MAX_DUTY_PCT, MAX_DUTY_PCT);
      duty = ff + kp * err + newInteg;
      bool saturated = fabsf(duty) > MAX_DUTY_PCT;
      if (!saturated || (err * duty < 0)) integ = newInteg;   // anti-windup
      duty = ff + kp * err + integ;
      if (duty * s < 0) duty = 0;                              // never brake by reversing
    }
  }
  duty = clampf(duty, -MAX_DUTY_PCT, MAX_DUTY_PCT);

  // stall protection
  if (fabsf(outDuty) >= STALL_DUTY_PCT && fabsf(measCps) < STALL_CPS) {
    if (stallStartMs == 0) stallStartMs = nowMs;
    if (nowMs - stallStartMs > STALL_MS && !stallFault) {
      stallFault = true;
      txLine("# STALL: motor powered but wheel not turning. Drive disabled, send X to clear.");
    }
  } else {
    stallStartMs = 0;
  }
  if (stallFault) { duty = 0; integ = 0; refCps = 0; }

  outDuty = stepToward(outDuty, duty, DUTY_SLEW_PCT_S * dt);
  applyDrive(outDuty);

  curSteer = stepToward(curSteer, targetSteer, STEER_RATE_DPS * dt);
  writeServo(curSteer);
}

// NEVER blocks. Serial.print() on the GIGA waits forever when the Pi keeps the port open but
// stops reading - that froze the control loop (and the watchdog) with the motor in its last state.
// send_nb() drops the line instead when the USB buffer is busy.
void txLine(const String &s) {
  String line = s + "\n";
  // `Serial` is a UART wrapper on the GIGA; _SerialUSB is the USB CDC object underneath it.
  // A USB packet holds 64 bytes, so longer lines go out in pieces; give up after ~5 ms.
  if (!_SerialUSB.connected()) return;
  const uint8_t *p = (const uint8_t *)line.c_str();
  uint32_t left = line.length();
  for (int tries = 0; left > 0 && tries < 50; tries++) {
    uint32_t sent = 0;
    _SerialUSB.send_nb((uint8_t *)p, left, &sent, true);
    p += sent;
    left -= sent;
    if (left > 0) delayMicroseconds(100);
  }
}

void sendTelemetry(unsigned long nowMs) {
  int flags = (stallFault ? 1 : 0) | (watchdogStop ? 2 : 0) | (mode == MODE_DUTY ? 4 : 0);
  txLine(String("E,") + String(nowMs) + "," + String(lastCount) + "," + String(measCps, 1) + "," +
         String(outDuty, 1) + "," + String(curSteer, 1) + "," + String(flags));
}

// ---------------- commands ----------------
int parseFloats(char *s, float *out, int maxN) {
  int n = 0;
  for (char *tok = strtok(s, " ,\t"); tok && n < maxN; tok = strtok(NULL, " ,\t")) {
    out[n++] = atof(tok);
  }
  return n;
}

void printSettings() {
  txLine(String("# settings servo_center=") + String(servoCenterDeg, 1) +
         " dir=" + String((int)servoDir) + " steer_max=" + String(steerMaxDeg, 1) +
         " gain=" + String(servoGain, 3) + " kp=" + String(kp, 5) + " ki=" + String(ki, 5) +
         " kff=" + String(kff, 5) + " u0=" + String(u0, 1) + " accel=" + String(accel, 0) +
         " enc_b=" + String(ENC_HAS_B ? 1 : 0) + " stall=" + String(stallFault ? 1 : 0));
}

void handleLine(char *line) {
  char cmd = line[0];
  float v[5] = {0, 0, 0, 0, 0};
  int n = parseFloats(line + 1, v, 5);
  unsigned long now = millis();
  switch (cmd) {
    case 'P':
      if (n >= 1) {
        servoRaw = true;
        servoRawDeg = clampf(v[0], 0, 180);
        writeServo(0);
        txLine(String("# servo raw ") + String(servoRawDeg, 1));
      }
      break;
    case 'C':
      servoRaw = false;
      if (n >= 2) {
        if (mode != MODE_SPEED) { mode = MODE_SPEED; integ = 0; refCps = 0; }
        targetCps = v[0];
        targetSteer = clampf(v[1], -steerMaxDeg, steerMaxDeg);
        lastCmdMs = now;
      }
      break;
    case 'D':
      servoRaw = false;
      if (n >= 2) {
        mode = MODE_DUTY;
        targetDuty = clampf(v[0], -MAX_DUTY_PCT, MAX_DUTY_PCT);
        targetSteer = clampf(v[1], -steerMaxDeg, steerMaxDeg);
        lastCmdMs = now;
      }
      break;
    case 'G':
      if (n >= 5) { kp = v[0]; ki = v[1]; kff = v[2]; u0 = clampf(v[3], 0, MAX_DUTY_PCT); accel = fabsf(v[4]); integ = 0; }
      printSettings();
      break;
    case 'T':
      if (n >= 3) {
        servoCenterDeg = clampf(v[0], 45, 135);
        servoDir = v[1] < 0 ? -1.0f : 1.0f;
        steerMaxDeg = clampf(fabsf(v[2]), 0, STEER_HARD_MAX);
        if (n >= 4 && fabsf(v[3]) > 0.1f) servoGain = fabsf(v[3]);
        servoRaw = false;
      }
      printSettings();
      break;
    case 'R':
      noInterrupts(); encCount = 0; interrupts();
      lastCount = 0;
      break;
    case 'X':
      stallFault = false; stallStartMs = 0;
      txLine("# stall fault cleared");
      break;
    case '?':
      printSettings();
      break;
    default:
      break;
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) { lineBuf[lineLen] = '\0'; handleLine(lineBuf); lineLen = 0; }
    } else if (lineLen < (int)sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;   // too long: drop it
    }
  }
}

// ---------------- setup / loop ----------------
void setup() {
  pinMode(PIN_RPWM, OUTPUT);
  pinMode(PIN_LPWM, OUTPUT);
  pinMode(PIN_REN, OUTPUT);
  pinMode(PIN_LEN, OUTPUT);
  applyDrive(0);

  pinMode(PIN_ENC_A, INPUT_PULLUP);   // open-collector output, no external resistor
  if (ENC_HAS_B) pinMode(PIN_ENC_B, INPUT_PULLUP);
  // Both edges, like the original RPM.ino, so the measured 2340 counts per wheel turn holds.
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), encIsr, CHANGE);

  servoPwm = new mbed::PwmOut(digitalPinToPinName(PIN_SERVO));
  servoPwm->period_ms(20);
  writeServo(0);

  Serial.begin(115200);
  lastTickUs = micros();
  lastCmdMs = millis();
  txLine("# agv_base ready");
}

void loop() {
  readSerial();
  unsigned long nowUs = micros();
  if (nowUs - lastTickUs >= PERIOD_US) {
    float dt = (nowUs - lastTickUs) * 1e-6f;
    lastTickUs = nowUs;
    unsigned long nowMs = millis();
    controlStep(dt, nowMs);
    sendTelemetry(nowMs);
  }
}
