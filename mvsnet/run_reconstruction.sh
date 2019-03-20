#!/bin/bash
ETH=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
for i in "${ETH[@]}"; do
  echo "processing $i"
  python test_tnt.py --target_set=${i}
  python depthfusion.py --dense_folder /data/outputs/eth3d/mvsnet/${i}/
done
