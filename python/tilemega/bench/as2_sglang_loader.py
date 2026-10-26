"""Canonical data transport only; SGLang owns model math and weight packing."""
import hashlib
import json
from pathlib import Path
import torch
from sglang.srt.model_loader.loader import DefaultModelLoader


class CanonicalModelLoader(DefaultModelLoader):
    def download_model(self,model_config):
        model_path=Path(model_config.model_path).resolve()
        root=Path('/home/d50064638/junhuip/TileMega/runs/as2')
        if not model_path.is_relative_to(root):raise ValueError('Only a local AS2 model is admitted')
        identity=json.loads((model_path/'as2_weights.json').read_text())
        directory=Path('/home/d50064638/junhuip/tilemega-as2/weights')/identity['model_id']
        def digest(path):
            with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
        manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
        if manifest['status']!='complete' or digest(manifest_path)!=identity['manifest_sha256'] or digest(model_path/'config.json')!=manifest['config_sha256']:
            raise ValueError('Canonical model/config identity mismatch')
        return model_path,directory,manifest,digest

    def _get_all_weights(self,model_config,model):
        path,directory,manifest,digest=self.download_model(model_config)
        consumed=[]
        for record in manifest['tensors']:
            file=directory/record['file']
            if file.resolve().parent!=directory or file.stat().st_size!=record['bytes'] or digest(file)!=record['sha256']:
                raise ValueError('Canonical weight identity mismatch: '+record['name'])
            cpu=torch.from_file(str(file),shared=False,size=record['bytes']//2,dtype=torch.bfloat16).reshape(record['shape'])
            consumed.append(record['name']);yield record['name'],cpu
        (path.parent/'weight_load.json').write_text(json.dumps(dict(
            manifest_sha256=digest(directory/'manifest.json'),config_sha256=manifest['config_sha256'],
            consumed=consumed,aliases=manifest['aliases'],
            loader='native SGLang model.load_weights and postprocess; canonical BF16 mmap'),indent=2)+'\n')
