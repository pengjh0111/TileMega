#!/usr/bin/env python3
"""Run a DNN Python command with the user's resource-only device provider."""
import os
from pathlib import Path
import runpy
import sys
from dnn1_device import install

install()
args=sys.argv[1:]
if args[:1]==['-m']:
    sys.argv=[args[1],*args[2:]]
    runpy.run_module(args[1],run_name='__main__',alter_sys=True)
elif args[:1]==['-c']:
    sys.argv=['-c',*args[2:]]
    exec(compile(args[1],'<string>','exec'),{'__name__':'__main__'})
elif args and not args[0].startswith('-'):
    script=str(Path(args[0]).resolve())
    sys.argv=[script,*args[1:]]
    sys.path.insert(0,str(Path(script).parent))
    runpy.run_path(script,run_name='__main__')
elif args in (['--version'],['-V']):
    print(sys.version)
else:
    raise ValueError('unsupported DNN Python command: '+str(args))
