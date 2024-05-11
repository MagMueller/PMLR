#!/bin/bash

# Define the partitions and their time slices
partitions=("gpuhe.120h" "gpuhe.24h" "gpuhe.4h" "gpu.120h" "gpu.24h" "gpu.4h" "gpupr.120h" "gpupr.24h" "gpupr.4h")

# Loop over each partition and append the output to access.txt
for partition in "${partitions[@]}"
do
  echo "Partition: $partition" >> access.txt
  scontrol show partition $partition >> access.txt
  echo "" >> access.txt  # Adding a blank line for better readability between entries
done

echo "All partition details have been written to access.txt."
