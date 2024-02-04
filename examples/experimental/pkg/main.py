#!/usr/bin/env python3
import os
import sys

for k in os.environ:
    if k.startswith("SGE") or k.startswith("JOB"):
        print(k, os.environ[k])
print(sys.argv[1:])
