#!/bin/bash
for d in /data/output-dtu/*; do
  echo "processing $d"
  python test.py --dense_folder $d
  python depthfusion.py --dense_folder $d
done
