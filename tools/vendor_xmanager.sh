#!/bin/bash

rm -rf lxm3/_vendor/xmanager
mkdir -p lxm3/_vendor/xmanager

cp -r third_party/xmanager/xmanager lxm3/_vendor/
rm -r lxm3/_vendor/xmanager/{docker,generated,contrib,cloud,cli,vizier,xm_local}
# this test uses xm_local
rm lxm3/_vendor/xmanager/xm/packagables_test.py

find lxm3/_vendor/xmanager -name '*.py' -exec \
  sed -i 's/^from xmanager/from lxm3\._vendor\.xmanager/' {} \;
