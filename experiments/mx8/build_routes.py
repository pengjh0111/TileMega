"""Build isolated 1024-position MoE DIAG plans without editing production headers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from tilemega.build.identity import digest,generate,sha


def build(source,out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    binary=out/'plan.so'
    if binary.exists():raise ValueError('retain previous builds; choose a fresh output')
    overlay=out/'include'
    changes={
      'tilemega/Backend/Mx/RouteTableV3.h': [('step < 64','step < 1024'),('unsigned step = 64','unsigned step = 1024')],
      'tilemega/Backend/Mx/RouterLogitsV3.h': [('p.debug_step<64','p.debug_step<1024')],
      'tilemega/Codegen/tasks/mx/MoeProgramV3.cuh': [('unsigned(past-64)',
          'unsigned(P::kPrompt?(past==63?0:1024):past-63)')],
      'tilemega/Codegen/tasks/mx/MoeRuntimeV3.cuh': [('Alloc(s,64*P::kLayers','Alloc(s,1024*P::kLayers'),
          ('positions=64ull*P::kLayers','positions=1024ull*P::kLayers')],
    }
    for name, replacements in changes.items():
        text=(ROOT/'include'/name).read_text()
        for before,after in replacements:
            if before not in text:raise ValueError('unknown capture source: '+before)
            text=text.replace(before,after)
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    Path(str(binary)+'.cu').write_bytes(source.read_bytes())
    manifest=json.loads(Path(str(source)[:-3]+'.plan.json').read_text())
    manifest.update(watchdog=0,route_capture_positions=1024,
        route_capture_positions_semantics='prefill past63 followed by decode past64..1086',
        reference_source_sha256=sha(source))
    Path(str(binary)+'.plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    inputs=[p for p in (ROOT/'include').rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cuh')]
    inputs += list(overlay.rglob('*.h'))+list(overlay.rglob('*.cuh'))+[source,Path(__file__).resolve()]
    files={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    compiler='/opt/maca/tools/cu-bridge/bin/cucc'
    snapshot=dict(root=str(ROOT),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files=files,source_digest=digest(files),compiler=compiler,compiler_sha256=sha(compiler))
    snapshot_path=Path(str(binary)+'.source.json');snapshot_path.write_text(json.dumps(snapshot,indent=2)+'\n')
    command=[compiler,'-std=c++20','-O3','-Xcompiler','-resource-usage','-fno-strict-aliasing','-mllvm','--max-mtreg-number=255',
        '-DTILEMEGA_SYNC_V3=1','-DTILEMEGA_MX_STATIC_DECODE_V2=1','-DTILEMEGA_MX_STATIC_MOE_V3=1',
        '-DTILEMEGA_WATCHDOG=0','-DTILEMEGA_TRACE_STAGE=0','-DTILEMEGA_MX_DIAG=1','-shared','-Xcompiler=-fPIC','-x','cu',
        '-I'+str(overlay),'-I'+str(ROOT/'include'),str(binary)+'.cu','-L/opt/maca/lib','-lcudart','-o',str(binary)]
    Path(str(binary)+'.build_command.txt').write_text(shlex.join(command)+'\n')
    with Path(str(binary)+'.ptxas.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    identity=generate(binary,ROOT,snapshot_path)
    print(json.dumps(dict(binary=str(binary),artifact_id=identity['artifact_id'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();build(args.source.resolve(),args.out)
