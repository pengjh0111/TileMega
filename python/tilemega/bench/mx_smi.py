"""MetaX GPU telemetry through mx-smi (replaces the nvidia-smi queries).

Inside a container without --pid=host, mx-smi lists only this container's
processes, with container PIDs (verified: MX-S hold probe). Memory of other
containers' processes shows up in `vram used` without an owner, so foreign
use is detected through unowned memory above the idle baseline.
"""
import re
import subprocess
import time

_SECTION = re.compile(r'^GPU#(\d+)\s')
_FIELD = re.compile(r'^\s+(.+?)\s+:\s+(-?[\d.]+)\s*([A-Za-z%]*)\s*$')
_PROC = re.compile(r'^\|\s+(\d+)\s+(\d+)\s+(\S.*?)\s+(\d+)\s+\|$')


def _run(args):
    return subprocess.check_output(['mx-smi', *args], text=True, stderr=subprocess.STDOUT)


def device(index):
    """One telemetry row for natural-enumeration device `index`."""
    out = _run(['--show-memory', '--show-board-power', '--show-usage', '--show-clock',
                '--show-temperature', '-i', str(index)])
    block, fields = None, {}
    for line in out.splitlines():
        if _SECTION.match(line):
            continue
        stripped = line.strip()
        if stripped and not _FIELD.match(line):
            block = stripped
            continue
        m = _FIELD.match(line)
        if m:
            fields[(block, m.group(1))] = float(m.group(2))
    total_kb = fields[('Memory', 'vram total')]
    used_kb = fields[('Memory', 'vram used')]
    clocks = {k[1]: v for k, v in fields.items() if k[0] == 'Clock'}
    temps = {k[1]: v for k, v in fields.items() if k[0] in ('Chip Temperature', 'Board Temperature')}
    return dict(time=time.time(), used_mib=used_kb / 1024, total_mib=total_kb / 1024,
                free_mib=(total_kb - used_kb) / 1024, power_w=fields[('Board Power', 'Total')],
                utilization_pct=fields[('Utilization', 'GPU')], clocks_mhz=clocks, temperature_c=temps,
                owners=processes(index))


def processes(index=None):
    """Visible device processes as {pid: MiB}, optionally for one device."""
    owners = {}
    for line in _run(['--show-process']).splitlines():
        m = _PROC.match(line.strip())
        if m and (index is None or int(m.group(1)) == int(index)):
            owners[int(m.group(2))] = owners.get(int(m.group(2)), 0) + int(m.group(4))
    return owners


def query_csv(index):
    """`used,free,total,power,util` in the order measure.py historically parsed."""
    row = device(index)
    return row['used_mib'], row['free_mib'], row['total_mib'], row['power_w'], row['utilization_pct']


if __name__ == '__main__':
    import json
    import sys
    print(json.dumps(device(int(sys.argv[1]) if len(sys.argv) > 1 else 0), indent=1))
