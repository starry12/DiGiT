"""Compile the filesystem-only builder; no graph loading or GPU access."""
import subprocess
from .common import *

def build():
    require(not (HERE/'manifest.json').exists(),'Frozen version; preserve compiled binary')
    BINARY.parent.mkdir(exist_ok=True)
    cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(HERE/'reuse.cpp'),'-o',str(BINARY)]
    subprocess.run(cmd,check=True)
    write(OUT/'build.json',dict(passed=True,command=cmd,source_sha256=sha(HERE/'reuse.cpp'),
                              matching_sha256=sha(HERE/'matching.h'),binary_sha256=sha(BINARY)))

if __name__=='__main__':build()
