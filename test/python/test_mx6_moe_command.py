"""Reject invalid V3 contracts before invoking a device compiler."""
import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
BINARY=Path(os.environ.get('TILEMEGA_BIN',ROOT/'build/tools/tilemega'))
BRIDGE=Path(os.environ.get('MX6_MOE_BRIDGE',ROOT/'runs/mx6/export/qwen3_30b_a3b_l24_decode/bridge.json'))


@pytest.mark.parametrize('change,error',[
    ({'version':2},'version must be 3'),
    ({'threads':576},'requires 512 threads'),
    ({'attn_version':2,'split':2},'attention V2 requires split1'),
    ({'attn_version':1},'invalid attention version'),
    ({'split':3},'invalid attention split'),
    ({'combine_mode':'deferred'},'invalid combine_mode'),
    ({'combine_mode':'unit_owner'},'combine_mode unit_owner is disqualified (moe_unit_owner litmus)'),
    ({'route_table_mode':'reorder_sum'},'invalid route_table_mode'),
    ({'router':{'part':'kStreamK'}},'router logits require TileRange TN16 WN1'),
    ({'router':{'wn':2}},'router logits require TileRange TN16 WN1'),
    ({'router':{'tn':32}},'router logits require TileRange TN16 WN1'),
    ({'purpose':'selected'},'selection'),
    ({'purpose':'selected','selection':{'cost_table_sha256':'0'*64,'candidate_rank':0,'predicted_ns':1}},'positive rank'),
])
def test_invalid_pin_stops_before_build(tmp_path,change,error):
    if not BINARY.is_file() or not BRIDGE.is_file():
        pytest.skip('CPU tool and actual BF16 MoE bridge required')
    pin=copy.deepcopy(json.loads((ROOT/'configs/pins/mx6/cp1/qwen3_30b_a3b_l24_B1_manual.json').read_text()))
    for key,value in change.items():
        if key=='router':pin['mx_moe_v3']['router'].update(value)
        else:pin['mx_moe_v3'][key]=value
    path=tmp_path/'pin.json';path.write_text(json.dumps(pin))
    out=tmp_path/'plan.so'
    result=subprocess.run([str(BINARY),'compile',str(BRIDGE),str(out),'--serving','decode',
        '--emit','serving','--frontend','decoder','--batch','1','--selection','predicted',
        '--evaluate-configs',str(path)],capture_output=True,text=True,timeout=60,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','MACA_VISIBLE_DEVICES':'','CUDACXX':'/no/device/compiler'})
    assert result.returncode!=0 and error in result.stderr
    assert not list(tmp_path.glob('plan.so*'))
