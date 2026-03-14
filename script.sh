#!/bin/bash

# Define parameters
JOURNAL_MODES=("data" "ordered" "writeback")
IO_SCHEDULERS=("bfq" "mq-deadline" "kyber" "none")
READ_AHEAD_SIZES=(128 256 512)  # Read-ahead size in KB
BARRIERS=("on" "off")
NOATIME_OPTIONS=("noatime" "atime")
COMMIT_INTERVALS=(5 30)
IO_ENGINES=("sync" "libaio")

MOUNT_POINT="/mnt/my_mount"
DEVICE="/dev/mapper/my_vg-my_lv"
OUTPUT_CSV="results.csv"

# Create the CSV file and add the header
echo "JOURNAL_MODES,IO_SCHEDULERS,READ_AHEAD_SIZES,BARRIERS,NOATIME_OPTIONS,COMMIT_INTERVALS,IO_ENGINES,latency,bandwidth,IOPS" > "$OUTPUT_CSV"

# Loop through each combination of parameters
for journal in "${JOURNAL_MODES[@]}"; do
  for scheduler in "${IO_SCHEDULERS[@]}"; do
    for read_ahead in "${READ_AHEAD_SIZES[@]}"; do
      for barrier in "${BARRIERS[@]}"; do
        for noatime in "${NOATIME_OPTIONS[@]}"; do
          for commit_interval in "${COMMIT_INTERVALS[@]}"; do
            for io_engine in "${IO_ENGINES[@]}"; do

              # Set scheduler
              echo $scheduler | sudo tee /sys/block/sdb/queue/scheduler

              # Set read-ahead size
              echo $((read_ahead * 1024)) | sudo tee /sys/block/sdb/queue/read_ahead_kb

              # Unmount if already mounted
              if mountpoint -q "$MOUNT_POINT"; then
                  sudo umount "$MOUNT_POINT"
              fi

              # Mount with current configuration
              sudo mount -o $journal,$noatime,commit=$commit_interval,barrier=$barrier "$DEVICE" "$MOUNT_POINT"

              # Run fio and save the output
              TMP_JSON="tmp.json"
              fio --name=my_mount --filename="$MOUNT_POINT/file1" --size=1G \
              --rw=randwrite --bs=4k --ioengine=$io_engine --iodepth=32 --numjobs=4 \
              --output-format=json --output="$TMP_JSON"

              # Display the JSON output to verify its contents
              cat "$TMP_JSON"

              # Extract the required metrics from the JSON file
              latency=$(jq '.jobs[0].write.lat.mean' "$TMP_JSON")
              bandwidth=$(jq '.jobs[0].write.bw' "$TMP_JSON")  # Bandwidth in KB/s
              iops=$(jq '.jobs[0].write.iops' "$TMP_JSON")

              # Append the results to the CSV file
              echo "$journal,$scheduler,$read_ahead,$barrier,$noatime,$commit_interval,$io_engine,$latency,$bandwidth,$iops" >> "$OUTPUT_CSV"

              # Pause to ensure system stability
              sleep 5

            done
          done
        done
      done
    done
  done
done

# Clean up
rm -f "$TMP_JSON"

echo "Tests completed. Results saved to $OUTPUT_CSV."

