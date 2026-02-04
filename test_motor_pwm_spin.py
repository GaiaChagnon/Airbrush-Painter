#!/usr/bin/env python3
"""
Simple motor PWM spin test.

Uses hardware PWM on the step pin to spin the motor.
"""
import socket
import json
import time

SOCKET_PATH = "/tmp/klippy_uds"
ETX = b"\x03"


def write_config() -> None:
    """Write printer.cfg with PWM step control."""
    config = '''# Config for PWM step testing
[mcu]
serial: /dev/serial/by-id/usb-Klipper_stm32h723xx_130028001051313234353230-if00
restart_method: command

[printer]
kinematics: none
max_velocity: 300
max_accel: 3000

# Step pin as hardware PWM
# 1000 Hz = 5 RPS at 200 steps/rev
[output_pin step_pwm]
pin: !PF13
pwm: True
cycle_time: 0.001
value: 0

# Enable pin
[output_pin enable_test]
pin: PF14
value: 0

# Direction pin
[output_pin dir_test]
pin: !PF12
value: 0
'''
    with open('/home/gaia/printer.cfg', 'w') as f:
        f.write(config)


def restart_klipper() -> None:
    """Restart Klipper to load config."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(SOCKET_PATH)
        sock.settimeout(5.0)
        sock.sendall(json.dumps({
            "id": 1,
            "method": "gcode/script",
            "params": {"script": "RESTART"}
        }).encode() + ETX)
        sock.close()
    except Exception:
        pass
    time.sleep(5)


def send_gcode(sock: socket.socket, gcode: str) -> None:
    """Send G-code command."""
    sock.sendall(json.dumps({
        "id": 1,
        "method": "gcode/script",
        "params": {"script": gcode}
    }).encode() + ETX)

    buf = b""
    while ETX not in buf:
        buf += sock.recv(4096)


def main():
    print("=" * 50)
    print("  SIMPLE PWM SPIN TEST")
    print("=" * 50)
    print()
    print("This test will spin the motor at 5 RPS for 10 seconds")
    print("using hardware PWM at 1000 Hz.")
    print()

    # Write config
    print("Writing config...")
    write_config()

    # Restart Klipper
    print("Restarting Klipper...")
    restart_klipper()

    # Connect
    print("Connecting...")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(SOCKET_PATH)
    sock.settimeout(10.0)

    # Enable motor
    print()
    print("Enabling motor...")
    send_gcode(sock, "SET_PIN PIN=enable_test VALUE=1")
    time.sleep(0.5)

    # Start spinning
    print("Starting PWM (1000 Hz = 5 RPS)...")
    print()
    send_gcode(sock, "SET_PIN PIN=step_pwm VALUE=0.5")

    # Run for 10 seconds
    for i in range(10):
        print(f"  Spinning... {i+1}/10 seconds")
        time.sleep(1)

    # Stop
    print()
    print("Stopping PWM...")
    send_gcode(sock, "SET_PIN PIN=step_pwm VALUE=0")
    time.sleep(0.2)

    # Disable
    print("Disabling motor...")
    send_gcode(sock, "SET_PIN PIN=enable_test VALUE=0")

    sock.close()

    print()
    print("=" * 50)
    print("  TEST COMPLETE")
    print("=" * 50)
    print()
    print("Did the motor spin for 10 seconds?")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted - stopping motor...")
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCKET_PATH)
            sock.settimeout(2.0)
            sock.sendall(json.dumps({
                "id": 1,
                "method": "gcode/script",
                "params": {"script": "SET_PIN PIN=step_pwm VALUE=0"}
            }).encode() + ETX)
            sock.close()
        except:
            pass
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
