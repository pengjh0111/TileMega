"""Dataset correctness gates for identified forward megakernels (no timing)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .export import DEFAULT_NAFNET, model
from tilemega.serving.plan import FORWARD, PlanLibrary
from tilemega.serving.weights import load_weights


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(actual, reference):
    return torch.nn.functional.cosine_similarity(actual.float(), reference.float(), dim=-1)


class NativeForward:
    """Keep bindings alive and check repeated L1/L2 launches on every input."""
    def __init__(self, library, export, bridge, batch):
        self.library = PlanLibrary(library)
        info = self.library.info
        if info.phase != FORWARD or info.capacity or not info.batch_lo <= batch <= info.batch_hi:
            raise ValueError('correctness requires a compatible stateless forward library')
        self.batch = batch
        self.export = Path(export)
        self.bridge_path = Path(bridge)
        self.bridge = json.loads(self.bridge_path.read_text())
        identity_path = Path(str(self.library.path) + '.identity.json')
        self.identity = json.loads(identity_path.read_text())
        if self.identity['binary_sha256'] != sha(self.library.path):
            raise ValueError('library identity mismatch')
        nodes = {node['name']: node for node in self.bridge['nodes']}
        signature = self.bridge['signature']
        self.inputs = [item['name'] for item in signature['inputs'] if item['kind'] == 'USER_INPUT']
        self.outputs = [item['name'] for item in signature['outputs'] if item['kind'] == 'USER_OUTPUT']
        self.tensors = load_weights(self.export / 'checkpoint', self.library, device='cuda')
        for name in self.inputs:
            node = nodes[name]
            shape = [batch, *map(int, node['shape'][1:])]
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.int64': torch.int64}[node['dtype']]
            self.tensors[name] = torch.empty(shape, dtype=dtype, device='cuda')
        # Output shapes are bound by the selected plan, independently of the
        # symbolic batch spelling retained in the export bridge.
        for name in self.outputs:
            node = nodes[name]
            self.tensors[name] = torch.empty([batch, *map(int, node['shape'][1:])],
                                             dtype=torch.bfloat16, device='cuda')
        external = {buffer.name for buffer in self.library.buffers if buffer.role == 1}
        if external != self.tensors.keys():
            raise ValueError('native bindings differ from the exported graph')
        self.plan = self.library.create(batch, {name: value.data_ptr()
                                               for name, value in self.tensors.items()}, 0)
        self.plan.set_steps([0])
        self.launches = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plan.__exit__(*args)

    def __call__(self, arguments):
        if len(arguments) != len(self.inputs):
            raise ValueError('input arity differs from the exported graph')
        for name, value in zip(self.inputs, arguments):
            self.tensors[name].copy_(value)
        first = None
        for mode in (1, 2):
            if not self.library.info.modes & mode:
                continue
            for name in self.outputs:
                self.tensors[name].fill_(float('nan'))
            self.plan.launch(0, mode, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            self.launches += 1
            values = [self.tensors[name].clone() for name in self.outputs]
            if any(not torch.isfinite(value).all() for value in values):
                raise AssertionError('native output is nonfinite or unwritten')
            if first is None:
                first = values
            elif any(not torch.equal(a, b) for a, b in zip(first, values)):
                raise AssertionError('native L1 and L2 outputs differ bitwise')
        if first is None:
            raise ValueError('library advertises no executable forward mode')
        return first


def references(name, export, checkpoint=None, nafnet_weights=DEFAULT_NAFNET):
    """Load original FP32 weights; verify native BF16 weights are their cast."""
    from safetensors import safe_open
    fp32, source = model(name, checkpoint=checkpoint, nafnet_weights=nafnet_weights,
                         dtype=torch.float32)
    bf16 = copy.deepcopy(fp32).to(torch.bfloat16)
    state = bf16.state_dict()
    path = Path(export) / 'checkpoint/model.safetensors'
    with safe_open(str(path), framework='pt', device='cpu') as tensors:
        if set(tensors.keys()) != state.keys():
            raise ValueError('reference and exported checkpoint names differ')
        for key, value in state.items():
            actual = tensors.get_tensor(key)
            if actual.dtype != value.dtype or not torch.equal(actual, value):
                raise ValueError('export is not the BF16 cast of reference checkpoint: ' + key)
    return fp32.cuda(), bf16.cuda(), dict(source=source, exported_checkpoint_sha256=sha(path),
        fp32='original upstream checkpoint', bf16='same checkpoint cast to BF16')


def classification_inputs(name, checkpoint, dataset):
    from PIL import Image
    if name == 'mbv1':
        import timm
        module, _ = model(name, checkpoint=checkpoint, dtype=torch.float32)
        transform = timm.data.create_transform(**timm.data.resolve_model_data_config(module),
                                               is_training=False)
    else:
        from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
        transform = (ResNet18_Weights.IMAGENET1K_V1 if name == 'resnet18' else
                     MobileNet_V2_Weights.IMAGENET1K_V2).transforms()
    root = Path(dataset)
    manifest = root / 'dm1-selected.json'
    selection = json.loads(manifest.read_text())
    paths = [str(item['path']) for item in selection]
    if len(paths) != 1000 or paths != sorted(paths) or len(set(paths)) != 1000:
        raise ValueError('ImageNetV2 gate requires exactly the first 1000 sorted paths')
    all_paths = sorted(str(path.relative_to(root)) for path in
                       (root / 'imagenetv2-matched-frequency-format-val').rglob('*.jpeg'))
    if paths != all_paths[:1000]:
        raise ValueError('ImageNetV2 selection is not the prescribed path prefix')
    for path in paths:
        image = root / path
        with Image.open(image) as value:
            tensor = transform(value.convert('RGB'))
        yield tensor.to(torch.bfloat16), dict(path=path, sha256=sha(image))


def bert_inputs(dataset, checkpoint, masked):
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer
    path = Path(dataset)
    if path.is_dir():
        path = path / 'wikitext-103-raw-v1/validation-00000-of-00001.parquet'
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    texts = parquet.read_table(path, columns=['text'])['text'].to_pylist()
    tokens = tokenizer('\n\n'.join(texts), add_special_tokens=False,
                       truncation=False)['input_ids']
    if len(tokens) < 256 * 128:
        raise ValueError('WikiText-103 validation has insufficient tokens')
    generator = torch.Generator().manual_seed(20261007)
    for index in range(256):
        ids = torch.tensor(tokens[index * 128:(index + 1) * 128], dtype=torch.int64)
        arguments = [ids, torch.zeros_like(ids)]
        length = int(torch.randint(64, 129, (), generator=generator)) if masked else 128
        if masked:
            arguments.append((torch.arange(128) < length).long())
        yield arguments, dict(index=index, valid_length=length,
                              ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())


def sidd_inputs(dataset):
    import cv2
    import lmdb
    import numpy as np
    root = Path(dataset)
    if (root / 'validation').is_dir():
        root = root / 'validation'
    noisy_path, clean_path = [root / (name + '.lmdb') for name in ('input_crops', 'gt_crops')]
    names = []
    for path in (noisy_path, clean_path):
        names.append([line.split()[0].rsplit('.', 1)[0] for line in
                      (path / 'meta_info.txt').read_text().splitlines() if line.strip()])
    if names[0] != names[1] or len(names[0]) != 1280 or len(set(names[0])) != 1280:
        raise ValueError('SIDD requires the 1280 matching official validation blocks')
    environments = [lmdb.open(str(path), readonly=True, lock=False, readahead=False)
                    for path in (noisy_path, clean_path)]
    try:
        with environments[0].begin() as noisy, environments[1].begin() as clean:
            for key in names[0]:
                payloads = [transaction.get(key.encode()) for transaction in (noisy, clean)]
                if any(value is None for value in payloads):
                    raise ValueError('SIDD block is missing: ' + key)
                images = [cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                          for value in payloads]
                if any(value is None or value.shape != (256, 256, 3) for value in images):
                    raise ValueError('SIDD validation block has wrong shape: ' + key)
                tensors = [torch.from_numpy(value[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
                           for value in images]
                yield tensors[0].to(torch.bfloat16), tensors[1], dict(key=key,
                    noisy_sha256=hashlib.sha256(payloads[0]).hexdigest(),
                    gt_sha256=hashlib.sha256(payloads[1]).hexdigest())
    finally:
        for environment in environments:
            environment.close()


def batches(iterator, batch):
    pending = []
    for item in iterator:
        pending.append(item)
        if len(pending) == batch:
            yield pending, batch
            pending = []
    if pending:
        valid = len(pending)
        yield pending + [pending[-1]] * (batch - valid), valid


def psnr(left, right):
    # RGB [0,1], without clamping or uint8 quantization, preserves the emitted
    # numerical differences. Each validation block contributes one PSNR.
    return -10 * torch.log10((left.float() - right.float()).square().flatten(1).mean(-1))


@torch.inference_mode()
def check(args):
    torch.set_num_threads(4)
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fp32, bf16, reference = references(args.model, args.export, args.checkpoint, args.nafnet_weights)
    if args.model in ('resnet18', 'mbv1', 'mbv2'):
        fp32, bf16 = [module.to(memory_format=torch.channels_last) for module in (fp32, bf16)]
        iterator = classification_inputs(args.model, args.checkpoint, args.data)
        total = 1000
    elif args.model == 'bert':
        iterator = bert_inputs(args.data, args.checkpoint, args.mask)
        total = 256
    else:
        iterator = sidd_inputs(args.data)
        total = 1280
    records = []
    with NativeForward(args.library, args.export, args.bridge, args.batch) as native:
        for items, valid in batches(iterator, args.batch):
            if args.model == 'bert':
                arguments = [torch.stack([item[0][index] for item in items]).cuda()
                             for index in range(len(items[0][0]))]
                kwargs = dict(input_ids=arguments[0], token_type_ids=arguments[1])
                if args.mask:
                    kwargs['attention_mask'] = arguments[2]
                truth, baseline = fp32(**kwargs), bf16(**kwargs)
                actual = native(arguments)
                hidden, pooler = actual
                for i in range(valid):
                    length = items[i][1]['valid_length']
                    records.append(dict(items[i][1], hidden_cosine_min=float(cosine(
                        hidden[i, :length], truth.last_hidden_state[i, :length]).min()),
                        pooler_cosine=float(cosine(pooler[i], truth.pooler_output[i])),
                        bf16_hidden_cosine_min=float(cosine(baseline.last_hidden_state[i, :length],
                            truth.last_hidden_state[i, :length]).min()),
                        bf16_pooler_cosine=float(cosine(baseline.pooler_output[i],
                            truth.pooler_output[i])),
                        tm_bf16_hidden_cosine_min=float(cosine(hidden[i, :length],
                            baseline.last_hidden_state[i, :length]).min())))
            else:
                image = torch.stack([item[0] for item in items]).cuda()
                truth, baseline = fp32(image.float()), bf16(image)
                actual, = native([image])
                if args.model == 'nafnet':
                    gt = torch.stack([item[1] for item in items]).cuda()
                    metrics = [psnr(left, right) for left, right in
                               ((actual, truth), (baseline, truth), (actual, gt), (baseline, gt))]
                    for i in range(valid):
                        records.append(dict(items[i][2], **{name: float(value[i]) for name, value in zip(
                            ('tm_fp32_psnr', 'bf16_fp32_psnr', 'tm_gt_psnr', 'bf16_gt_psnr'), metrics)}))
                else:
                    similarity = cosine(actual, truth)
                    top1, baseline_top1 = actual.argmax(-1), baseline.argmax(-1)
                    oracle_top1 = truth.argmax(-1)
                    for i in range(valid):
                        records.append(dict(items[i][1], cosine=float(similarity[i]),
                            bf16_cosine=float(cosine(baseline[i],truth[i])),
                            tm_top1_match=bool(top1[i] == oracle_top1[i]),
                            bf16_top1_match=bool(baseline_top1[i] == oracle_top1[i])))
        if len(records) != total:
            raise ValueError('dataset gate did not visit every prescribed sample')
        mean = lambda key: sum(row[key] for row in records) / total
        if args.model == 'bert':
            metrics = {key: min(row[key] for row in records)
                       for key in ('hidden_cosine_min', 'pooler_cosine')}
            passed = all(value >= .999 for value in metrics.values())
            metrics.update({key: min(row[key] for row in records) for key in
                ('bf16_hidden_cosine_min', 'bf16_pooler_cosine', 'tm_bf16_hidden_cosine_min')})
        elif args.model == 'nafnet':
            metrics = {key: mean(key) for key in
                       ('tm_fp32_psnr', 'bf16_fp32_psnr', 'tm_gt_psnr', 'bf16_gt_psnr')}
            passed = (metrics['tm_fp32_psnr'] >= metrics['bf16_fp32_psnr'] - 1 and
                      abs(metrics['tm_gt_psnr'] - metrics['bf16_gt_psnr']) <= .05)
        else:
            metrics = {key: mean(key) for key in ('cosine', 'bf16_cosine', 'tm_top1_match', 'bf16_top1_match')}
            passed = (metrics['cosine'] >= .999 and
                      metrics['tm_top1_match'] >= metrics['bf16_top1_match'] - .002)
        return dict(evidence='verified', scope='G-DNN dataset correctness', passed=passed,
            model=args.model, masked=args.mask, batch=args.batch, samples=total,
            artifact_id=native.identity['artifact_id'], identity=native.identity,
            reference=reference, bridge_sha256=sha(args.bridge),
            input_protocol='identical BF16 images or int64 ids; original FP32 upstream reference',
            launches=native.launches, metrics=metrics, samples_detail=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('resnet18', 'mbv1', 'mbv2', 'bert', 'nafnet'), required=True)
    for option in ('library', 'export', 'bridge', 'data', 'out'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--nafnet-weights', type=Path, default=DEFAULT_NAFNET)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--mask', action='store_true')
    args = parser.parse_args(argv)
    if args.mask and args.model != 'bert':
        parser.error('--mask requires BERT')
    result = check(args)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
        ('passed', 'model', 'samples', 'artifact_id', 'metrics')}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
