"""CPU-only, plan-bound MX-5 resource policy. No builds, GPU work or fallback."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import tarfile

from .build.identity import digest, parse_resources
from .mx5_resources import inspect
from .mx5_select import check_file, record, validate_preview
from . import mx5_select as _native_selection

_TRUSTED_QUALIFIER = Path(_native_selection.__file__).resolve().parents[2] / "experiments/mx5/stage_table.py"

CLASSES = ('qkv', 'o', 'gate_up', 'down', 'head')
STAGES = (*CLASSES, 'attention')
POLICY = dict(version=1, maximum_reductions=3, largest='numeric_reducible_class_pw',
              tie_order=list(CLASSES), scope='shared_class_coordinate', floor='D',
              attention='unchanged', threads='576_only', selection_anchor='original_candidate_0',
              ranking_claim='original_baseline_top4_only')


class ContractError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise ContractError(reason)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, 'invalid integer ' + name)
    return value


def _positive(value, name):
    _require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'invalid ' + name)
    return value


@dataclass(frozen=True)
class FrozenJSON:
    encoded: bytes

    @classmethod
    def freeze(cls, value):
        return cls(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

    @property
    def data(self):
        return json.loads(self.encoded)

    @property
    def sha256(self):
        return hashlib.sha256(self.encoded).hexdigest()


@dataclass(frozen=True)
class OriginCertificate:
    frozen: FrozenJSON

    @classmethod
    def capture(cls, preview_path):
        preview = validate_preview(preview_path)
        candidates = preview['candidates']
        _require(len(candidates) == 4 and [c['index'] for c in candidates] == list(range(4)),
                 'origin must retain all four ordered native candidates')
        layers = None
        previous = -math.inf
        for candidate in candidates:
            ns = _positive(candidate['predicted_tpot_ns'], 'origin TPOT')
            _require(ns >= previous, 'origin candidates are not in native prediction order'); previous = ns
            stages = candidate['predicted_stage_ns']
            for name in STAGES:
                _positive(stages[name], 'origin stage time')
            count = (ns - stages['head']) / sum(stages[n] for n in STAGES if n != 'head')
            rounded = round(count)
            _require(rounded >= 1 and math.isclose(count, rounded, rel_tol=1e-12, abs_tol=1e-12),
                     'origin prediction does not establish an integer layer count')
            _require(layers is None or layers == rounded, 'origin layer counts differ')
            layers = rounded
        return cls(FrozenJSON.freeze(dict(schema_version=1, kind='mx5_plan_origin_certificate',
            native_preview=record(preview_path), preview=preview, layers=layers, policy=POLICY,
            final=False, default_enabled=False, selection_anchor=0,
            ranking_claim='exact original native top4; no repaired-global-ranking claim')))

    @property
    def sha256(self):
        return self.frozen.sha256

    @property
    def data(self):
        return self.frozen.data

    def verify(self):
        path = check_file(self.data['native_preview'])
        _require(OriginCertificate.capture(path).frozen == self.frozen, 'origin certificate differs from immutable native preview/policy')

    def candidate(self, index):
        _require(type(index) is int and index in range(4), 'invalid origin candidate index')
        return self.data['preview']['candidates'][index]


def affected_edges(name, layers):
    edges = {
        'qkv': [dict(from_stage='entry', to_stage='qkv', to_layer=0),
                dict(from_stage='head', to_stage='qkv', to_layer=0, step='next')]
               + [dict(from_stage='down', from_layer=l, to_stage='qkv', to_layer=l + 1) for l in range(layers - 1)],
        'o': [dict(from_stage='attention', from_layer=l, to_stage='o', to_layer=l) for l in range(layers)],
        'gate_up': [dict(from_stage='o', from_layer=l, to_stage='gate_up', to_layer=l) for l in range(layers)],
        'down': [dict(from_stage='gate_up', from_layer=l, to_stage='down', to_layer=l) for l in range(layers)],
        'head': [dict(from_stage='down', from_layer=layers - 1, to_stage='head')],
    }
    return edges[name]


def _same_geometry(origin_pin, effective_pin):
    def without_width(pin):
        value = FrozenJSON.freeze(pin).data
        for name in CLASSES:
            value['mx_decode_v2'][name].pop('prefetch')
        return value
    _require(FrozenJSON.freeze(without_width(origin_pin)) == FrozenJSON.freeze(without_width(effective_pin)),
             'retry changed geometry, R_next, threads, attention PF or non-width metadata')
    for name in CLASSES:
        c = effective_pin['mx_decode_v2'][name]
        _require(_integer(c['prefetch'], 'Pw') >= _integer(c['d'], 'D', 1), 'retry Pw is below D')


def _key(origin, candidate_id, name, pin):
    base = origin.candidate(candidate_id)['evidence'][name][0]
    key = dict(base['key']); key['prefetch'] = pin['mx_decode_v2'][name]['prefetch']
    return dict(kind=base['kind'], key=key)


def _files(value):
    for name in ('pin', 'resolved_pin', 'generated_source', 'command', 'source_snapshot', 'source_patch',
                 'compiler', 'producer', 'compile_receipt', 'log'):
        check_file(value[name])
    for source in value.get('stage_sources', []):
        check_file(source['archive'])
    for name in ('binary', 'resources', 'occupancy'):
        if value.get(name) is not None:
            check_file(value[name])


def _snapshot_inputs(snapshot):
    root = Path(snapshot['root']).resolve()
    for name, expected in snapshot['source_files'].items():
        relative = Path(name)
        _require(not relative.is_absolute() and '..' not in relative.parts, 'invalid snapshot source path')
        check_file(dict(path=str(root / relative), sha256=expected))


def _generated_config(text, origin, candidate_id, pin):
    request = origin.data['preview']['request']; candidate = origin.candidate(candidate_id)
    expected_constants = dict(kLayers=origin.data['layers'], kThreads=576, kGrid=104,
        kBatch=request['batch'], kCapacity=request['capacity'], kPastLo=request['past_lo'], kPastHi=request['past_hi'])
    for name, expected in expected_constants.items():
        found = re.findall(r'\b' + name + r'\s*=\s*(\d+)', text)
        _require(len(found) == 1 and int(found[0]) == expected, 'generated constant differs from actual plan: ' + name)
    def alias(name, expected):
        found = re.findall(r'\busing\s+' + name + r'\s*=\s*([^;]+);', text)
        _require(len(found) == 1 and re.sub(r'\s+', '', found[0]) == expected,
                 'generated body/shape differs from actual pin: ' + name)
    config = pin['mx_decode_v2']
    for name, emitted in zip(CLASSES, ('Qkv','O','GateUp','Down','Head')):
        c = config[name]; key = candidate['evidence'][name][0]['key']
        args = '%d,%d,%d' % (c['tn'], c['wn'], c['d'])
        if name == 'head':
            body = 'MxHead<' + args + ',' + str(c['prefetch']) + '>'
        else:
            op = ('kStore','kResidual','kSwiGLU')[key['op']]
            body = 'MxGemm<' + args + ',MxOpV2::' + op + ',' + ('true' if key['norm'] else 'false') + ',MxPart::' + c['part'] + ',' + str(c['prefetch']) + '>'
        alias(emitted, body)
        alias(emitted + 'Shape', 'MxStaticShapeV2<%d,%d,kBatch,kThreads>' % (key['n'], key['k']))
        if name == 'down': alias('DownLast', body)
    key = candidate['evidence']['attention'][0]['key']
    alias('Attn', 'MxAttentionV2<%d,%d,%d,%s,%d,kCapacity>' %
          (key['dim'], key['q'], config['b_min'], 'true' if key['qk_norm'] else 'false', config['attn_prefetch']))
    alias('AttnShape', 'MxStaticAttentionShapeV2<kBatch,%d,kThreads>' % key['hkv'])


def _attempt(value, origin, candidate_id, expected_pin, ordinal):
    _require(value.get('schema_version') == 1 and value.get('kind') == 'mx5_plan_attempt', 'wrong plan-attempt schema')
    _require(value.get('origin_sha256') == origin.sha256 and value.get('candidate_id') == candidate_id,
             'attempt origin/candidate identity mismatch')
    _require(value.get('ordinal') == ordinal, 'attempt ordinals must be contiguous')
    _files(value)
    pin = json.loads(check_file(value['pin']).read_text())
    _same_geometry(origin.candidate(candidate_id)['pin'], pin)
    _require(FrozenJSON.freeze(pin) == FrozenJSON.freeze(expected_pin), 'attempt pin differs from authorized effective pin')
    resolved = json.loads(check_file(value['resolved_pin']).read_text())['mx_decode_v2']
    for name in CLASSES:
        for field in ('tn','wn','d','part','prefetch','r_next'):
            _require(type(resolved[name].get(field)) is type(pin['mx_decode_v2'][name][field]) and resolved[name].get(field) == pin['mx_decode_v2'][name][field], 'actual resolved pin differs: ' + name + '.' + field)
    for field in ('threads','b_min','attn_prefetch'):
        _require(type(resolved.get(field)) is type(pin['mx_decode_v2'][field]) and resolved.get(field) == pin['mx_decode_v2'][field], 'actual resolved pin differs: ' + field)
    _require(pin['mx_decode_v2']['threads'] == 576, '512 execution remains blocked pending the explicit proof contract')
    snapshot = json.loads(check_file(value['source_snapshot']).read_text())
    _require(snapshot['source_digest'] == digest(snapshot['source_files']), 'source snapshot digest mismatch')
    _snapshot_inputs(snapshot)
    _require(value['source_patch']['sha256'] == snapshot['worktree_diff_sha256'], 'source patch digest mismatch')
    _require(snapshot['compiler_sha256'] == value['producer']['sha256'], 'producer compiler identity mismatch')
    _require(value['producer']['sha256'] == origin.data['preview']['producer']['sha256'], 'producer differs from immutable native origin')
    _require(Path(snapshot['compiler']).resolve() == check_file(value['producer']), 'producer path mismatch')
    argv = shlex.split(check_file(value['command']).read_text())
    _require(argv and Path(argv[0]).resolve() == check_file(value['compiler']), 'device compiler identity mismatch')
    _require('-o' in argv and '-print-after=stack-frame-layout' in argv, 'command lacks output or same-compile MIR capture')
    macros = {}
    for argument in argv:
        if argument.startswith('-D'):
            name, _, setting = argument[2:].partition('=')
            _require(name not in macros, 'duplicate compiler define ' + name)
            macros[name] = setting
    for name, setting in (('TILEMEGA_MX_STATIC_DECODE_V2','1'), ('TILEMEGA_WATCHDOG','0'),
                          ('TILEMEGA_TRACE_STAGE','0'), ('TILEMEGA_MX_DIAG','0')):
        _require(macros.get(name) == setting, 'actual resource attempt has wrong instrumentation define ' + name)
    generated = check_file(value['generated_source']); output = Path(argv[argv.index('-o') + 1]).resolve()
    _require(str(generated) in argv, 'command does not compile the recorded generated source')
    if value.get('binary') is not None:
        _require(output == check_file(value['binary']), 'compiled binary path mismatch')
    receipt = json.loads(check_file(value['compile_receipt']).read_text())
    _require(receipt.get('argv') == argv and receipt.get('same_compile_mir') == value['log']['sha256'],
             'compile receipt does not bind command and same-compile MIR')
    for name in ('pin', 'resolved_pin', 'generated_source', 'source_snapshot', 'compiler', 'producer', 'log'):
        _require(receipt.get(name + '_sha256') == value[name]['sha256'], 'compile receipt identity mismatch: ' + name)
    _require(receipt.get('binary_sha256') == (value['binary']['sha256'] if value.get('binary') else None),
             'compile receipt binary identity mismatch')
    options = [('<generated>' if arg == str(generated) else '<output>' if arg == str(output) else arg) for arg in argv]
    epoch = digest(dict(source_digest=snapshot['source_digest'], compiler=value['compiler']['sha256'],
                        producer=value['producer']['sha256'], options=options))
    _require(value.get('program_source_epoch') == epoch, 'declared program source epoch mismatch')
    _generated_config(generated.read_text(), origin, candidate_id, pin)
    checked_sources = set()
    for rows in origin.candidate(candidate_id)['evidence'].values():
        for row in rows:
            source = row['provenance']['source_sha256']
            if source not in checked_sources:
                _verify_archive(row, value.get('stage_sources', []), snapshot['source_files'])
                checked_sources.add(source)
    if receipt.get('returncode') != 0:
        return dict(epoch=epoch, classification='blocked', reason='device compiler failed; no resource retry is authorized')
    _require(value.get('binary') is not None and value.get('resources') is not None,
             'successful device compile lacks binary/resource metadata')
    actual = inspect(check_file(value['binary']), check_file(value['log']), 576)
    parsed = json.loads(check_file(value['resources']).read_text())
    _require(parsed == actual, 'resource report differs from actual device ELF/same-compile MIR')
    kernel = actual['kernel']
    _integer(kernel['registers'], 'actual MTreg', 1)
    for field in ('scalar_registers','stack_bytes','static_smem_bytes'):
        _integer(kernel[field], field)
    _require(set(actual['kernel_instructions']) == {kernel['name']}, 'resource and same-compile MIR kernel names differ')
    if kernel['static_smem_bytes']:
        return dict(epoch=epoch, classification='blocked', reason='static shared memory failure is not a Pw retry trigger')
    occupancy = None
    if value.get('occupancy'):
        occupancy = json.loads(check_file(value['occupancy']).read_text())
        _require(occupancy.get('kind') == 'mx5_plan_occupancy'
                 and occupancy.get('binary_sha256') == value['binary']['sha256']
                 and occupancy.get('threads') == 576, 'occupancy proof does not bind the actual plan')
        _integer(occupancy['query_status'], 'occupancy query status')
        _integer(occupancy['resident_ctas'], 'resident CTA count')
        if occupancy.get('query_status') != 0 or occupancy.get('resident_ctas') != 1:
            return dict(epoch=epoch, classification='blocked', reason='actual occupancy query failed or did not establish one CTA')
    failure = kernel['registers'] > 227 or kernel['stack_bytes'] > 0
    if failure:
        return dict(epoch=epoch, classification='resource_failure', kernel=kernel,
                    reason='actual MTreg=%d cap=227 stack=%d' % (kernel['registers'], kernel['stack_bytes']))
    if occupancy is None:
        return dict(epoch=epoch, classification='blocked', reason='resource-fit program lacks bound actual occupancy metadata')
    return dict(epoch=epoch, classification='resource_fit', kernel=kernel, reason='actual program resources fit')


def program_epoch(receipt):
    """Compute the declared epoch without executing a compiler or touching a GPU."""
    snapshot = json.loads(check_file(receipt['source_snapshot']).read_text())
    argv = shlex.split(check_file(receipt['command']).read_text())
    generated = str(check_file(receipt['generated_source'])); output = str(Path(argv[argv.index('-o') + 1]).resolve())
    options = [('<generated>' if x == generated else '<output>' if x == output else x) for x in argv]
    return digest(dict(source_digest=snapshot['source_digest'], compiler=receipt['compiler']['sha256'],
                       producer=receipt['producer']['sha256'], options=options))


@dataclass(frozen=True)
class PlanLedger:
    origin: OriginCertificate
    candidate_id: int
    pin: FrozenJSON
    attempts: tuple[FrozenJSON, ...] = ()
    authorizations: tuple[FrozenJSON, ...] = ()
    cost_bindings: tuple[FrozenJSON, ...] = ()
    terminal_error: str | None = None

    @classmethod
    def start(cls, origin, candidate_id):
        pin = origin.candidate(candidate_id)['pin']
        return cls(origin, candidate_id, FrozenJSON.freeze(pin), terminal_error=(
            '512 execution remains blocked pending the explicit proof contract' if pin['mx_decode_v2']['threads'] != 576 else None))

    @property
    def sha256(self):
        return digest(dict(origin=self.origin.sha256, candidate_id=self.candidate_id, pin=self.pin.sha256,
            attempts=[a.sha256 for a in self.attempts], authorizations=[a.sha256 for a in self.authorizations],
            bindings=[b.sha256 for b in self.cost_bindings], terminal_error=self.terminal_error))

    def append_attempt(self, receipt_path):
        if self.terminal_error:
            return self
        try:
            self.origin.verify()
            _require(len(self.attempts) == len(self.authorizations), 'another retry authorization is required before a further attempt')
            receipt = json.loads(Path(receipt_path).read_text())
            verified = _attempt(receipt, self.origin, self.candidate_id, self.pin.data, len(self.attempts))
            if self.attempts:
                _require(verified['epoch'] == self.attempts[0].data['verified']['epoch'],
                         'program source epoch changed; start a new origin/attempt chain')
            value = FrozenJSON.freeze(dict(receipt=record(receipt_path), payload=receipt, verified=verified))
            return replace(self, attempts=(*self.attempts, value))
        except (ValueError, KeyError, OSError, TypeError) as error:
            descriptor = record(receipt_path) if Path(receipt_path).is_file() else dict(path=str(receipt_path), missing=True)
            failed = FrozenJSON.freeze(dict(receipt=descriptor, verified=dict(classification='blocked', reason=str(error))))
            return replace(self, attempts=(*self.attempts, failed), terminal_error=str(error))

    def as_dict(self):
        return dict(schema_version=1, kind='mx5_plan_attempt_ledger', origin=self.origin.data,
                    origin_sha256=self.origin.sha256, candidate_id=self.candidate_id, effective_pin=self.pin.data,
                    attempts=[x.data for x in self.attempts], authorizations=[x.data for x in self.authorizations],
                    cost_bindings=[x.data for x in self.cost_bindings], terminal_error=self.terminal_error,
                    ledger_sha256=self.sha256, final=False, default_enabled=False)

    def write(self, path):
        with Path(path).open('x') as output:
            json.dump(self.as_dict(), output, indent=2, allow_nan=False); output.write('\n')

    @classmethod
    def restore(cls, path):
        saved = json.loads(Path(path).read_text())
        _require(saved.get('schema_version') == 1 and saved.get('kind') == 'mx5_plan_attempt_ledger', 'wrong ledger schema')
        origin = OriginCertificate(FrozenJSON.freeze(saved['origin']))
        _require(origin.sha256 == saved['origin_sha256'], 'restored origin certificate hash mismatch')
        origin.verify(); ledger = cls.start(origin, saved['candidate_id'])
        _require(len(saved['authorizations']) == len(saved['cost_bindings']), 'ledger authorization/cost counts differ')
        _require(len(saved['attempts']) in (len(saved['authorizations']), len(saved['authorizations']) + 1),
                 'ledger attempt/authorization counts are not contiguous')
        for index, attempt in enumerate(saved['attempts']):
            ledger = ledger.append_attempt(attempt['receipt']['path'])
            _require(ledger.attempts[-1].data == attempt, 'restored attempt differs from immutable actual receipt')
            if index < len(saved['authorizations']):
                catalogs = {json.dumps(v['catalog'], sort_keys=True) for v in saved['cost_bindings'][index]['effective_variant_rows'].values()}
                _require(len(catalogs) == 1, 'binding lacks one immutable exact-width catalog')
                catalog = VariantCatalog.load(check_file(json.loads(next(iter(catalogs)))))
                action = next_plan_retry(ledger, catalog)
                _require(action.kind == ActionKind.BUILD_REDUCED_PLAN
                         and action.details.data['authorization'] == saved['authorizations'][index],
                         'restored authorization differs from deterministic prescribed retry')
                ledger = ledger.commit_retry(action, catalog)
                _require(ledger.cost_bindings[-1].data == saved['cost_bindings'][index], 'restored exact cost binding differs')
        _require(ledger.pin.data == saved['effective_pin'] and ledger.terminal_error == saved['terminal_error']
                 and ledger.sha256 == saved['ledger_sha256'], 'restored ledger state/digest differs')
        return ledger

    def commit_retry(self, action, catalog):
        expected = next_plan_retry(self, catalog)
        _require(action.kind == ActionKind.BUILD_REDUCED_PLAN and action == expected,
                 'retry action/cost binding differs from the verified deterministic action')
        data = action.details.data
        return replace(self, pin=FrozenJSON.freeze(data['next_pin']),
                       authorizations=(*self.authorizations, FrozenJSON.freeze(data['authorization'])),
                       cost_bindings=(*self.cost_bindings, FrozenJSON.freeze(data['prediction'])))


class ActionKind(str, Enum):
    NEEDS_STAGE_MEASUREMENTS = 'needs_stage_measurements'
    BUILD_REDUCED_PLAN = 'build_reduced_plan'
    NEEDS_D7 = 'needs_d7'
    RESOURCE_FIT = 'resource_fit'
    BLOCKED = 'blocked'


@dataclass(frozen=True)
class RetryAction:
    kind: ActionKind
    reason: str
    details: FrozenJSON


def _action(kind, reason, **details):
    return RetryAction(kind, reason, FrozenJSON.freeze(dict(final=False, default_enabled=False, execution_512_allowed=False, **details)))


def next_plan_retry(ledger, catalog=None):
    try:
        ledger.origin.verify()
        if ledger.terminal_error:
            return _action(ActionKind.BLOCKED, ledger.terminal_error)
        _require(ledger.attempts, 'no actual program attempt')
        for stored in ledger.attempts:
            check_file(stored.data['receipt'])
            _files(stored.data['payload'])
            _snapshot_inputs(json.loads(check_file(stored.data['payload']['source_snapshot']).read_text()))
        for index, binding in enumerate(ledger.cost_bindings):
            latest_auth = {a.data['shared_class']: a.data for a in ledger.authorizations[:index + 1]}
            source_files = json.loads(check_file(ledger.attempts[0].data['payload']['source_snapshot']).read_text())['source_files']
            for name, bound in binding.data['effective_variant_rows'].items():
                prior_catalog = VariantCatalog.load(check_file(bound['catalog']))
                row = bound['row']; current = prior_catalog.lookup(dict(kind=row['kind'], key=row['key']), latest_auth[name], source_files)
                _require(current is not None and current.data == bound, 'committed exact row binding changed')
        _require(len(ledger.attempts) == len(ledger.authorizations) + 1,
                 'no completed actual program attempt for the current pin')
        latest = ledger.attempts[-1].data
        state = latest['verified']
        if state['classification'] == 'blocked':
            return _action(ActionKind.BLOCKED, state['reason'])
        if state['classification'] == 'resource_fit':
            return _action(ActionKind.RESOURCE_FIT, state['reason'], original_candidate_id=ledger.candidate_id,
                           original_prediction_ns=ledger.origin.candidate(ledger.candidate_id)['predicted_tpot_ns'],
                           effective_prediction=ledger.cost_bindings[-1].data if ledger.cost_bindings else dict(
                               origin_predicted_tpot_ns=ledger.origin.candidate(ledger.candidate_id)['predicted_tpot_ns'],
                               effective_predicted_tpot_ns=ledger.origin.candidate(ledger.candidate_id)['predicted_tpot_ns'],
                               effective_stage_ns=ledger.origin.candidate(ledger.candidate_id)['predicted_stage_ns']),
                           scope='original_baseline_top4_descendant', selection_anchor=0, ledger_sha256=ledger.sha256,
                           actual_attempt_receipt=ledger.attempts[-1].data['receipt'])
        pin = ledger.pin.data; config = pin['mx_decode_v2']
        reducible = [name for name in CLASSES if config[name]['prefetch'] > config[name]['d']]
        if not reducible:
            if state['kernel']['registers'] <= 227:
                return _action(ActionKind.BLOCKED, 'all Pw equal D but stack-only 512 fallback remains unspecified; retain D7 evidence')
            return _action(ActionKind.NEEDS_D7, 'all Pw equal D and actual MTreg exceeds 227; 512 remains blocked',
                           failed_attempt_sha256=ledger.attempts[-1].sha256, program_source_epoch=state['epoch'],
                           confirmed_boolean_accepted=False, ledger_sha256=ledger.sha256, origin_sha256=ledger.origin.sha256,
                           candidate_id=ledger.candidate_id, failed_attempt_receipt=ledger.attempts[-1].data['receipt'])
        if len(ledger.authorizations) >= 3:
            return _action(ActionKind.BLOCKED, 'three whole-plan reductions exhausted while some Pw remain above D; no extra reduction or 512 fallback')
        name = min(reducible, key=lambda n: (-config[n]['prefetch'], CLASSES.index(n)))
        old = config[name]['prefetch']; new = max(config[name]['d'], old // 2)
        config[name]['prefetch'] = new
        _same_geometry(ledger.origin.candidate(ledger.candidate_id)['pin'], pin)
        authorization = dict(kind='plan_resource_retry', schema_version=1, origin_sha256=ledger.origin.sha256,
            candidate_id=ledger.candidate_id, parent_attempt_sha256=ledger.attempts[-1].sha256,
            program_source_epoch=state['epoch'], reduction_step=len(ledger.authorizations) + 1,
            shared_class=name, from_pw=old, to_pw=new, previous_pin_sha256=ledger.pin.sha256,
            next_pin_sha256=FrozenJSON.freeze(pin).sha256, affected_edges=affected_edges(name, ledger.origin.data['layers']),
            policy=POLICY)
        chain = (*ledger.authorizations, FrozenJSON.freeze(authorization))
        binding = bind_plan_costs(ledger, pin, chain, catalog)
        details = dict(ledger_sha256=ledger.sha256, next_pin=pin, authorization=authorization, **binding.data)
        if binding.data['missing']:
            return _action(ActionKind.NEEDS_STAGE_MEASUREMENTS, 'exact source-qualified three-process width rows are missing', **details)
        return _action(ActionKind.BUILD_REDUCED_PLAN, 'one prescribed largest numeric reducible class Pw halving', **details)
    except (ValueError, KeyError, OSError, TypeError) as error:
        return _action(ActionKind.BLOCKED, str(error))


def _load_qualifier(path):
    path = check_file(path)
    _require(path == _TRUSTED_QUALIFIER.resolve(), 'variant catalog cannot select arbitrary qualifier code')
    spec = importlib.util.spec_from_file_location('mx5_variant_stage_qualifier', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _validate_row(row, qualifier):
    kind = row['kind']; key = row['key']
    _require(kind in ('gemm', 'head'), 'this bounded variant catalog changes GEMM/head Pw only')
    _require(qualifier.canonical(dict(key=key), kind) == key, 'noncanonical variant key')
    _require(set(key) == set(qualifier.FIELDS[kind]), 'variant key must be canonical and complete')
    _require('resource_selection' not in row, 'plan retry rows must not claim fabricated stage-bench resource selection')
    _positive(row['median_ns'], 'variant median'); _positive(row['p90_ns'], 'variant p90')
    _integer(row['r_next'], 'variant R_next', 1)
    _require(type(row['spread']) in (int,float) and math.isfinite(row['spread']) and row['spread'] >= 0, 'invalid variant spread')
    _require(row.get('processes') == 3 and len(row['rounds']) == 3, 'variant requires three fresh measured processes')
    medians, p90s, paths, rounds = [], [], set(), set()
    expected_provenance = qualifier.identity(row['provenance'])
    _require(qualifier.kind_of(expected_provenance) == kind, 'variant kind/provenance mismatch')
    expected_resources = qualifier.resources(row['resources'], key['threads'])
    for sample in row['rounds']:
        number = _integer(sample['round'], 'round', 1); measurement = sample['measurement']
        k, coordinates, provenance, resources = qualifier.validate_measurement(measurement)
        _require(k == kind and coordinates == key and provenance == expected_provenance
                 and resources == expected_resources and measurement['round'] == number, 'variant round identity/key/resource mismatch')
        _require(number in (1, 2, 3) and number not in rounds, 'variant process rounds must be exactly 1/2/3')
        rounds.add(number)
        input_path = check_file(sample['input'])
        _require(str(input_path) not in paths, 'variant processes reuse an input file'); paths.add(str(input_path))
        lines = input_path.read_text().splitlines(); line = _integer(sample['input']['line'], 'input line', 1)
        _require(line <= len(lines) and json.loads(lines[line - 1]) == measurement, 'variant raw process line differs from measurement')
        medians.append(measurement['median_ns']); p90s.append(measurement['p90_ns'])
    records, _ = qualifier.load_runs([sample['input']['path'] for sample in row['rounds']])
    selected = [r for r in records if r['kind'] == kind and r['key'] == key]
    _require(len(selected) == 3 and {r['row']['round'] for r in selected} == {1,2,3},
             'variant raw batches do not contain exactly three unique key/process rounds')
    _require(statistics.median(medians) == row['median_ns'] and statistics.median(p90s) == row['p90_ns'],
             'variant aggregate differs from process medians')
    _require(math.isfinite(row['spread']) and abs(row['spread'] - (max(medians)-min(medians))/row['median_ns']) <= .000050001,
             'variant spread differs from raw processes')
    ref_kind, ref_key, ref_provenance, ref_resources = qualifier.validate_reference(row['r_next_reference'])
    expected = dict(key, prefetch=0)
    _require(ref_kind == kind and ref_key == expected and ref_resources['mtreg'] == row['r_next'], 'variant R_next reference mismatch')
    _require(all(ref_provenance[f] == expected_provenance[f] for f in ('executor', 'implementation', 'source_sha256')),
             'variant zero-prefetch source identity mismatch')


def _verify_archive(row, sources, source_files):
    digest_value = row['provenance']['source_sha256']
    matches = [s for s in sources if s['source_sha256'] == digest_value]
    _require(len(matches) == 1, 'missing or ambiguous stage source archive')
    archive = check_file(matches[0]['archive']); h = hashlib.sha256(); shared = 0
    with tarfile.open(archive, 'r:gz') as source:
        members = [m for m in source.getmembers() if m.isfile()]
        _require(len({m.name for m in members}) == len(members), 'duplicate stage source archive member')
        for member in sorted(members, key=lambda m: m.name):
            body = source.extractfile(member).read(); h.update(member.name.encode()+b'\0'+body+b'\0')
            if member.name.startswith('include/tilemega/'):
                shared += 1
                _require(source_files.get(member.name) == hashlib.sha256(body).hexdigest(),
                         'variant shared body source differs from actual program epoch: ' + member.name)
    _require(shared > 0 and h.hexdigest() == digest_value, 'variant source archive digest mismatch')


@dataclass(frozen=True)
class VariantCatalog:
    frozen: FrozenJSON

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text())
        _require(document.get('schema_version') == 1 and document.get('kind') == 'mx5_decode_stage_variants',
                 'wrong variant catalog schema; never replace the baseline table')
        _require(document.get('qualification_scope') == 'plan_bound_measured_widths', 'wrong variant qualification scope')
        qualifier = _load_qualifier(document['qualification_tool'])
        seen = set()
        for entry in document['rows']:
            _require(entry['authorization'].get('kind') == 'plan_resource_retry', 'variant needs distinct plan_resource_retry authorization')
            _require(entry['row_sha256'] == digest(entry['row']), 'variant exact row hash mismatch')
            _validate_row(entry['row'], qualifier)
            identity = digest(dict(kind=entry['row']['kind'], key=entry['row']['key'], authorization=entry['authorization']))
            _require(identity not in seen, 'ambiguous duplicate exact authorized variant row'); seen.add(identity)
        for item in document['binaries']:
            check_file(item)
        return cls(FrozenJSON.freeze(dict(catalog=record(path), document=document)))

    def lookup(self, request, authorization, source_files):
        data = self.frozen.data; check_file(data['catalog'])
        document = data['document']; check_file(document['qualification_tool'])
        for entry in document['rows']:
            row = entry['row']
            if row['kind'] != request['kind'] or row['key'] != request['key'] or FrozenJSON.freeze(entry['authorization']) != FrozenJSON.freeze(authorization):
                continue
            _require(entry['row_sha256'] == digest(row), 'variant exact row hash mismatch')
            _validate_row(row, _load_qualifier(document['qualification_tool']))
            _verify_archive(row, document['sources'], source_files)
            binaries = {b['sha256']: b for b in document['binaries']}
            for provenance in (row['provenance'], row['r_next_reference']['provenance']):
                _require(provenance['binary_sha256'] in binaries, 'variant/reference binary is absent from immutable proof bundle')
                check_file(binaries[provenance['binary_sha256']])
            return FrozenJSON.freeze(dict(row=row, row_sha256=entry['row_sha256'], catalog=data['catalog']))
        return None


def bind_plan_costs(ledger, pin, authorizations, catalog):
    _require(len(authorizations) == len(ledger.authorizations) + 1
             and authorizations[:-1] == ledger.authorizations, 'cost binding must extend exactly the actual plan retry chain')
    _require(ledger.attempts and ledger.attempts[-1].data['verified']['classification'] == 'resource_failure',
             'cost binding lacks an actual program register/stack failure')
    authorization = authorizations[-1].data
    _require(authorization['kind'] == 'plan_resource_retry'
             and authorization['parent_attempt_sha256'] == ledger.attempts[-1].sha256
             and authorization['origin_sha256'] == ledger.origin.sha256
             and authorization['candidate_id'] == ledger.candidate_id
             and authorization['program_source_epoch'] == ledger.attempts[-1].data['verified']['epoch']
             and authorization['previous_pin_sha256'] == ledger.pin.sha256
             and authorization['next_pin_sha256'] == FrozenJSON.freeze(pin).sha256
             and authorization['reduction_step'] == len(authorizations) <= 3,
             'cost binding authorization does not bind the actual parent attempt/epoch/pin')
    old = ledger.pin.data; reducible = [n for n in CLASSES if old['mx_decode_v2'][n]['prefetch'] > old['mx_decode_v2'][n]['d']]
    _require(reducible, 'cost binding has no reducible Pw')
    name = min(reducible, key=lambda n: (-old['mx_decode_v2'][n]['prefetch'], CLASSES.index(n)))
    before = old['mx_decode_v2'][name]['prefetch']; after = max(old['mx_decode_v2'][name]['d'], before // 2)
    old['mx_decode_v2'][name]['prefetch'] = after
    _require(FrozenJSON.freeze(old) == FrozenJSON.freeze(pin)
             and authorization['shared_class'] == name and authorization['from_pw'] == before and authorization['to_pw'] == after
             and authorization['affected_edges'] == affected_edges(name, ledger.origin.data['layers'])
             and authorization['policy'] == POLICY, 'cost binding is not the prescribed shared-coordinate largest-Pw reduction')
    _same_geometry(ledger.origin.candidate(ledger.candidate_id)['pin'], pin)
    original = ledger.origin.candidate(ledger.candidate_id)
    stages = dict(original['predicted_stage_ns']); missing, rows = [], {}
    latest = {a.data['shared_class']: a.data for a in authorizations}
    source_files = json.loads(check_file(ledger.attempts[-1].data['payload']['source_snapshot']).read_text())['source_files']
    for name, authorization in latest.items():
        request = _key(ledger.origin, ledger.candidate_id, name, pin)
        bound = catalog.lookup(request, authorization, source_files) if catalog else None
        if bound is None:
            missing.append(dict(shared_class=name, **request, authorization_sha256=digest(authorization), required_fresh_processes=3, source_sha256=original['evidence'][name][0]['provenance']['source_sha256'],
                                r_next=original['evidence'][name][0]['r_next']))
            continue
        row = bound.data['row']; base = original['evidence'][name][0]
        _require(row['provenance']['source_sha256'] == base['provenance']['source_sha256']
                 and row['r_next'] == base['r_next'], 'variant changed the original measured source/R_next epoch')
        stages[name] = row['median_ns']; rows[name] = bound.data
    prediction = None
    if not missing:
        total = stages['head'] + ledger.origin.data['layers'] * sum(stages[n] for n in STAGES if n != 'head')
        prediction = dict(origin_sha256=ledger.origin.sha256, original_candidate_id=ledger.candidate_id,
            origin_predicted_tpot_ns=original['predicted_tpot_ns'], effective_predicted_tpot_ns=total,
            effective_stage_ns=stages, effective_variant_rows=rows, selection_anchor=0,
            ranking_claim='descendant of original exact top4; not globally exact after repair')
    return FrozenJSON.freeze(dict(missing=missing, prediction=prediction))
