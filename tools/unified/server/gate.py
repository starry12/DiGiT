#!/usr/bin/python3 -I
import os
import sys
os.execv('/usr/bin/python3', ['/usr/bin/python3', '-I', '-B', '/srv/digit-ae/admin/unified_20261011_v1/cli.py', *sys.argv[1:]])
