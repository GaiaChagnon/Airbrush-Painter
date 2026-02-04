#!/usr/bin/env python3
"""
Manual stepper test with speed ramp and rapid direction changes.

Tests motor using manual_stepper configuration which bypasses Klipper's
kinematics system entirely - no homing required, no coordinate limits.

Test sequence:
1. Speed ramp up (forward) - accelerate to max speed
2. Speed ramp up (reverse) - accelerate to max speed
3. Rapid 360s - 10 fast full rotations alternating direction
"""
import socket
import json
import time

SOCKET_PATH = "/tmp/klippy_uds"
ETX = b"\x03"

# Motor configuration
# 0.9° stepper = 400 native steps/rev
# Driver set to 1600 steps/rev
# rotation_distance=160 means MOVE=160 = 1 full revolution
ROTATION_DISTANCE = 160.0  # mm per revolution (scaled for driver)


def write_config() -> None:
    """Write printer.cfg with manual_stepper configuration."""
    config = '''# Manual stepper test configuration
# Bypasses kinematics - no homing required

[mcu]
serial: /dev/serial/by-id/usb-Klipper_stm32h723xx_130028001051313234353230-if00

[printer]
kinematics: none
max_velocity: 300
max_accel: 3000

# Manual stepper - allows movement without homing
# 0.9° motor = 400 native steps/rev, driver = 1600 steps/rev
# rotation_distance: 160mm means MOVE=160 = 1 full revolution
[manual_stepper test_motor]
step_pin: PF13
dir_pin: PF12
enable_pin: PF14
microsteps: 16
rotation_distance: 160
velocity: 400
accel: 8000
step_pulse_duration: 0.000005
'''
    with open('/home/gaia/printer.cfg', 'w') as f:
        f.write(config)


def restart_klipper() -> None:
    """Restart Klipper to load new config."""
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


def send_gcode(sock: socket.socket, gcode: str, timeout: float = 30.0) -> bool:
    """Send G-code command and wait for response."""
    sock.sendall(json.dumps({
        "id": 1,
        "method": "gcode/script",
        "params": {"script": gcode}
    }).encode() + ETX)

    sock.settimeout(timeout)
    buf = b""
    deadline = time.monotonic() + timeout
    
    while ETX not in buf and time.monotonic() < deadline:
        buf += sock.recv(4096)
    
    msg = json.loads(buf[:buf.index(ETX)].decode())
    
    if "error" in msg:
        print(f"  !! Error: {str(msg['error'])[:100]}")
        return False
    
    return True


def speed_ramp(sock: socket.socket, direction: str, max_speed: float = 80.0, steps: int = 8) -> None:
    """
    Perform a speed ramp test.
    
    Parameters
    ----------
    sock : socket
        Connected socket.
    direction : str
        "forward" or "reverse"
    max_speed : float
        Maximum speed in mm/s (320 mm/s = 2 RPS with rotation_distance=160)
    steps : int
        Number of speed increments.
    """
    print()
    print("=" * 60)
    print(f"  SPEED RAMP ({direction.upper()})")
    print("=" * 60)
    print()
    print(f"  Ramping from slow to {max_speed} mm/s ({max_speed/ROTATION_DISTANCE:.1f} RPS)")
    print()
    
    # Calculate positions
    # Each step moves a bit, accumulating distance
    position = 0.0
    sign = 1 if direction == "forward" else -1
    
    for i in range(1, steps + 1):
        speed = (i / steps) * max_speed
        rps = speed / ROTATION_DISTANCE
        
        # Move 0.5 rotations at this speed
        move_dist = 0.5 * ROTATION_DISTANCE  # 20mm = half rotation
        position += sign * move_dist
        
        pct = (i / steps) * 100
        print(f"  [{pct:3.0f}%] {speed:5.1f} mm/s = {rps:.2f} RPS")
        
        send_gcode(sock, f"MANUAL_STEPPER STEPPER=test_motor MOVE={position:.1f} SPEED={speed:.1f} SYNC=1", timeout=10.0)
    
    print()
    print(f"  Ramp complete! Final position: {position:.1f}mm")


def rapid_360s(sock: socket.socket, count: int = 10, speed: float = 100.0) -> None:
    """
    Perform rapid full rotations alternating direction.
    
    Parameters
    ----------
    sock : socket
        Connected socket.
    count : int
        Number of 360s to perform.
    speed : float
        Speed in mm/s.
    """
    print()
    print("=" * 60)
    print(f"  RAPID 360s - {count} alternating rotations")
    print("=" * 60)
    print()
    print(f"  Speed: {speed} mm/s = {speed/ROTATION_DISTANCE:.1f} RPS")
    print()
    
    one_rotation = ROTATION_DISTANCE  # 40mm = one full rotation
    position = 0.0
    
    for i in range(count):
        # Alternate direction
        direction = ">>>" if i % 2 == 0 else "<<<"
        target = one_rotation if i % 2 == 0 else 0.0
        
        print(f"  [{i+1:2}/{count}] {direction} 360° ", end="", flush=True)
        
        t0 = time.time()
        send_gcode(sock, f"MANUAL_STEPPER STEPPER=test_motor MOVE={target:.1f} SPEED={speed:.1f} SYNC=1", timeout=10.0)
        elapsed = time.time() - t0
        
        print(f"({elapsed:.2f}s)")
        
        position = target
        
        # 500ms pause between direction changes
        time.sleep(0.5)
    
    print()
    print("  Rapid 360s complete!")


def main():
    print("=" * 60)
    print("  MANUAL STEPPER - SPEED RAMP & RAPID 360s TEST")
    print("=" * 60)
    print()
    print("Test sequence:")
    print("  1. Speed ramp (forward) - accelerate to 2 RPS")
    print("  2. Speed ramp (reverse) - accelerate to 2 RPS")
    print("  3. Rapid 360s - 10 fast alternating rotations")
    print()
    print("Configuration:")
    print(f"  - rotation_distance: {ROTATION_DISTANCE}mm")
    print(f"  - 1 rotation = {ROTATION_DISTANCE}mm")
    print(f"  - Max speed: 400 mm/s = {400/ROTATION_DISTANCE:.1f} RPS")
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
    sock.settimeout(15.0)
    
    # Enable motor
    print()
    print("Enabling motor...")
    send_gcode(sock, "MANUAL_STEPPER STEPPER=test_motor ENABLE=1")
    time.sleep(0.5)
    
    # Reset position
    send_gcode(sock, "MANUAL_STEPPER STEPPER=test_motor SET_POSITION=0")
    
    print()
    print("Starting in 2 seconds...")
    time.sleep(2)
    
    # Test 1: Speed ramp forward (up to 2 RPS = 320 mm/s)
    speed_ramp(sock, direction="forward", max_speed=320.0, steps=8)
    time.sleep(1)
    
    # Return to start
    print()
    print("Returning to start position...")
    send_gcode(sock, "MANUAL_STEPPER STEPPER=test_motor MOVE=0 SPEED=240 SYNC=1", timeout=15.0)
    time.sleep(1)
    
    # Test 2: Speed ramp reverse (up to 2 RPS = 320 mm/s)
    speed_ramp(sock, direction="reverse", max_speed=320.0, steps=8)
    time.sleep(1)
    
    # Return to start
    print()
    print("Returning to start position...")
    send_gcode(sock, "MANUAL_STEPPER STEPPER=test_motor MOVE=0 SPEED=240 SYNC=1", timeout=15.0)
    time.sleep(1)
    
    # Test 3: Rapid 360s (2.5 RPS = 400 mm/s)
    rapid_360s(sock, count=10, speed=400.0)
    
    # Return to start and disable
    print()
    print("Returning to start...")
    send_gcode(sock, "MANUAL_STEPPER STEPPER=test_motor MOVE=0 SPEED=240 SYNC=1", timeout=15.0)
    
    print("Disabling motor...")
    send_gcode(sock, "MANUAL_STEPPER STEPPER=test_motor ENABLE=0")
    
    sock.close()
    
    print()
    print("=" * 60)
    print("  ALL TESTS COMPLETE!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - Forward ramp: Did it accelerate smoothly?")
    print("  - Reverse ramp: Did it accelerate in reverse?")
    print("  - Rapid 360s: Did it snap back and forth quickly?")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted - disabling motor...")
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(SOCKET_PATH)
            sock.settimeout(2.0)
            sock.sendall(json.dumps({
                "id": 1,
                "method": "gcode/script",
                "params": {"script": "MANUAL_STEPPER STEPPER=test_motor ENABLE=0"}
            }).encode() + ETX)
            sock.close()
        except:
            pass
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
