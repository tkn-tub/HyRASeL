#!/bin/bash

# Baseline
python3 omnet_data_visualization.py --method TWR CENTRALIZED DISTRIBUTED --experiment_path <Path-to-Baseline-Data-Dir> --timestamp_format noisy_timestamp --rep_limit 20

# Speed
python3 omnet_data_visualization.py --method TWR CENTRALIZED DISTRIBUTED --experiment_path <Path-to-Speed-Data-Dir> --timestamp_format noisy_timestamp --experiment_parameter speed:2,5,10,25,50 --rep_limit 20

# Number Nodes
python3 omnet_data_visualization.py --method TWR CENTRALIZED DISTRIBUTED --experiment_path <Path-to-NumNodes-Data-Dir> --timestamp_format noisy_timestamp --experiment_parameter numNodes:4,9,16,25,36,49 --rep_limit 20

# Anchor Density
python3 omnet_data_visualization.py --method TWR CENTRALIZED DISTRIBUTED HYPERBOLIC_SELF_LOCALIZATION --experiment_path <Path-to-AnchorDensity-Data-Dir> --timestamp_format noisy_timestamp --experiment_parameter interAnchorDist:130,200,270 --rep_limit 20

# Node Density
python3 omnet_data_visualization.py --method TWR CENTRALIZED DISTRIBUTED --experiment_path <Path-to-NodeDensity-Data-Dir> --timestamp_format noisy_timestamp --experiment_parameter size:80,60,40,20 --rep_limit 20

# NumSlots
python3 omnet_data_visualization.py --method CENTRALIZED DISTRIBUTED --experiment_path  <Path-to-NumSlots-Data-Dir> --timestamp_format noisy_timestamp --experiment_parameter numSlots:2,3,4,5 --rep_limit 20 


