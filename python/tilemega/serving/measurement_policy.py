"""Standard-library-only power guard shared by the two benchmark processes."""
import json
import os
from pathlib import Path
import subprocess
import time


class TimingPolicy:
    def __init__(self, path=None):
        self.options = json.loads(Path(path).read_text()) if path else {}
        self.enabled = bool(self.options.get('guard', False))
        if self.enabled and self.options.get('idle_power_w') is None:
            raise ValueError('power guard requires a measured idle power')

    def observe(self, log: Path, label: str) -> bool:
        if not self.enabled:
            return True
        threshold = self.options['idle_power_w'] + self.options['power_margin_w']
        # A completed request leaves the board hot for a few seconds. Sample
        # every observation, then require the unchanged predeclared threshold
        # before accepting the round. Persistent external activity still fails.
        deadline=time.monotonic()+self.options.get('cooldown_seconds',30)
        while True:
            raw = subprocess.check_output(['nvidia-smi', '-i', os.environ.get('TILEMEGA_DEVICE_INDEX', '0'), '--query-gpu=power.draw,clocks.sm,clocks.mem,temperature.gpu',
                                           '--format=csv,noheader,nounits'], text=True).splitlines()[0]
            power, sm, mem, temp = map(float, raw.split(','))
            accepted = power <= threshold
            with Path(log).open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(), label=label, power_w=power,
                    sm_mhz=sm, memory_mhz=mem, temperature_c=temp, power_threshold_w=threshold,
                    idle_power_w=self.options['idle_power_w'], power_margin_w=self.options['power_margin_w'],
                    power_accepted=accepted)) + '\n')
            if accepted or time.monotonic()>=deadline:
                return accepted
            time.sleep(1)
