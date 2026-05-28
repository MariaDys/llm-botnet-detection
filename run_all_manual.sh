#!/bin/bash
cd ~/Desktop/botnet

declare -A devices
devices[1]="device_1_danmini_doorbell"
devices[2]="device_2_ecobee_thermostat"
devices[3]="device_3_ennio_doorbell"
devices[4]="device_4_philips_baby_monitor"
devices[5]="device_5_provision_pt737e"
devices[6]="device_6_provision_pt838"
devices[7]="device_7_samsung_webcam"
devices[8]="device_8_simplehome_1002"
devices[9]="device_9_simplehome_1003"

for device in 1 2 3 4 5 6 7 8 9; do
    folder="results/${devices[$device]}"
    echo "========== ${devices[$device]} =========="
    python3 baseline/botnet_detection_manual.py --data_dir dataset --device_id $device \
        2>&1 | tee "${folder}/manual_baseline.txt"
done

echo "DONE!"