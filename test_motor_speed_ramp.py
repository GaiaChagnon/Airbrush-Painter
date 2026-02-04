#!/usr/bin/env python3
"""
Motor speed ramp test.

Accelerates motor from 0 to target speed (1-2 RPS), then decelerates back to 0.
Uses G1 moves with cartesian kinematics for proper motion control.

Configuration assumptions:
    - rotation_distance: 40mm (40mm linear travel per revolution)
    - 200 steps/rev at driver
    - 1 RPS = 40 mm/s, 2 RPS = 80 mm/s
"""
import sys
import time

from robot_control.configs.loader import load_config
from robot_control.hardware.klipper_client import KlipperClient


def run_speed_ramp(
    client: KlipperClient,
    target_rps: float = 1.5,
    ramp_time: float = 3.0,
    hold_time: float = 2.0,
    steps: int = 10,
    rotation_distance: float = 40.0,
) -> None:
    """
    Run a speed ramp test: accelerate to target, hold, decelerate to stop.

    Parameters
    ----------
    client : KlipperClient
        Connected Klipper client.
    target_rps : float
        Target rotations per second (1.0-2.0 recommended).
    ramp_time : float
        Time to accelerate from 0 to target (seconds).
    hold_time : float
        Time to hold at target speed (seconds).
    steps : int
        Number of velocity steps in ramp (more = smoother).
    rotation_distance : float
        mm per revolution from printer.cfg.
    """
    target_velocity = target_rps * rotation_distance  # mm/s
    
    print(f"\nSpeed Ramp Configuration:")
    print(f"  Target: {target_rps:.1f} RPS = {target_velocity:.1f} mm/s")
    print(f"  Ramp time: {ramp_time:.1f}s up, {ramp_time:.1f}s down")
    print(f"  Hold time: {hold_time:.1f}s at max speed")
    print(f"  Velocity steps: {steps}")
    print()

    # Set starting position (middle of range)
    print("Setting kinematic position...")
    client.send_gcode("SET_KINEMATIC_POSITION X=125 Y=175 Z=20")
    time.sleep(0.2)

    # Calculate velocity steps
    velocities = [(i / steps) * target_velocity for i in range(1, steps + 1)]
    
    # Track position for direction changes
    position = 125.0
    direction = 1  # 1 = positive X, -1 = negative X

    print("=" * 50)
    print("  ACCELERATING")
    print("=" * 50)

    # Accelerate through velocity steps
    for i, velocity in enumerate(velocities):
        # Calculate distance for this step (time-based)
        step_time = ramp_time / steps
        move_dist = velocity * step_time
        
        # Update position and check bounds
        new_pos = position + (direction * move_dist)
        if new_pos > 240:
            direction = -1
            new_pos = position + (direction * move_dist)
        elif new_pos < 10:
            direction = 1
            new_pos = position + (direction * move_dist)
        
        position = new_pos
        feedrate = velocity * 60  # mm/s to mm/min
        
        rps = velocity / rotation_distance
        pct = ((i + 1) / steps) * 100
        print(f"  [{pct:3.0f}%] {velocity:5.1f} mm/s = {rps:.2f} RPS")
        
        client.send_gcode(f"G1 X{position:.2f} F{feedrate:.0f}")
    
    # Wait for acceleration moves to complete
    client.send_gcode("M400")

    print()
    print("=" * 50)
    print(f"  HOLDING AT {target_rps:.1f} RPS for {hold_time:.1f}s")
    print("=" * 50)

    # Hold at max speed - continuous back and forth
    hold_distance = target_velocity * hold_time
    feedrate = target_velocity * 60
    
    # Move in current direction for hold time
    new_pos = position + (direction * hold_distance)
    if new_pos > 240:
        # Split into two moves
        dist_to_end = 240 - position
        remaining = hold_distance - dist_to_end
        client.send_gcode(f"G1 X240 F{feedrate:.0f}")
        client.send_gcode(f"G1 X{240 - remaining:.2f} F{feedrate:.0f}")
        position = 240 - remaining
        direction = -1
    elif new_pos < 10:
        dist_to_end = position - 10
        remaining = hold_distance - dist_to_end
        client.send_gcode(f"G1 X10 F{feedrate:.0f}")
        client.send_gcode(f"G1 X{10 + remaining:.2f} F{feedrate:.0f}")
        position = 10 + remaining
        direction = 1
    else:
        client.send_gcode(f"G1 X{new_pos:.2f} F{feedrate:.0f}")
        position = new_pos
    
    client.send_gcode("M400")

    print()
    print("=" * 50)
    print("  DECELERATING")
    print("=" * 50)

    # Decelerate through velocity steps (reverse order)
    for i, velocity in enumerate(reversed(velocities)):
        step_time = ramp_time / steps
        move_dist = velocity * step_time
        
        new_pos = position + (direction * move_dist)
        if new_pos > 240:
            direction = -1
            new_pos = position + (direction * move_dist)
        elif new_pos < 10:
            direction = 1
            new_pos = position + (direction * move_dist)
        
        position = new_pos
        feedrate = velocity * 60
        
        rps = velocity / rotation_distance
        pct = ((steps - i) / steps) * 100
        print(f"  [{pct:3.0f}%] {velocity:5.1f} mm/s = {rps:.2f} RPS")
        
        client.send_gcode(f"G1 X{position:.2f} F{feedrate:.0f}")
    
    # Wait for all moves to complete
    client.send_gcode("M400")

    print()
    print("=" * 50)
    print("  COMPLETE")
    print("=" * 50)


def main():
    print("=" * 60)
    print("  MOTOR SPEED RAMP TEST")
    print("=" * 60)
    print()
    print("This test will:")
    print("  1. STEPPER_BUZZ to verify motor responds")
    print("  2. Accelerate motor from 0 to ~1.5 RPS")
    print("  3. Hold at max speed")
    print("  4. Decelerate back to 0")
    print("  5. STEPPER_BUZZ to confirm completion")
    print()
    print("Configuration:")
    print("  - 200 steps/rev at driver")
    print("  - rotation_distance: 40mm")
    print("  - 1 RPS = 40 mm/s = 200 Hz pulse rate")
    print("  - 1.5 RPS = 60 mm/s = 300 Hz pulse rate")
    print()

    # Load config
    config = load_config()
    
    print("Connecting to Klipper...")
    with KlipperClient(config.connection.socket_path) as client:
        status = client.get_status()
        print(f"Printer state: {status.state}")
        
        if status.state == "shutdown":
            print("Klipper in shutdown state, restarting...")
            client.send_gcode("FIRMWARE_RESTART")
            time.sleep(5)
            # Reconnect after restart
            client.connect()
        
        if status.state != "ready":
            print(f"Warning: Printer state is {status.state}")
        
        print()
        print("=" * 50)
        print("  START BUZZ TEST")
        print("=" * 50)
        print("Motor should buzz back and forth 3 times...")
        time.sleep(1)
        client.send_gcode("STEPPER_BUZZ STEPPER=stepper_x DISTANCE=10 REPEAT=3", timeout=15.0)
        print("Buzz complete.")
        
        time.sleep(2)
        
        # Run the speed ramp
        run_speed_ramp(
            client,
            target_rps=1.5,
            ramp_time=2.0,
            hold_time=1.5,
            steps=8,
            rotation_distance=40.0,
        )
        
        time.sleep(2)
        
        print()
        print("=" * 50)
        print("  END BUZZ TEST")
        print("=" * 50)
        print("Motor should buzz back and forth 3 times...")
        client.send_gcode("STEPPER_BUZZ STEPPER=stepper_x DISTANCE=10 REPEAT=3", timeout=15.0)
        print("Buzz complete.")
    
    print()
    print("=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print()
    print("If motor didn't move during ramp but buzzed at start/end:")
    print("  - G1 moves may not be working (check enable pin timing)")
    print("  - Try slower speeds or check Klipper logs")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
