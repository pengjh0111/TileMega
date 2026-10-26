"""Fail-closed compilation receipts for the independent NPU baseline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers.cache_utils import StaticCache


class NpuStaticCache(StaticCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        position=(cache_kwargs or {}).get('cache_position')
        key,value=self.key_cache[layer_idx],self.value_cache[layer_idx]
        if position is None:
            key.copy_(key_states.to(key.dtype));value.copy_(value_states.to(value.dtype))
        else:
            # Positions are the same unique runtime indices as HF StaticCache.
            # Installed TorchAir has no aten.index_copy GE converter.
            torch.ops.npu.scatter_update_(key,position,key_states.to(key.dtype),2)
            torch.ops.npu.scatter_update_(value,position,value_states.to(value.dtype),2)
        return key,value


class CompileAudit:
    def __init__(self, backend, directory: Path, identity, max_graphs=4):
        self.backend=backend;self.directory=directory;self.identity=identity
        self.max_graphs=max_graphs;self.graphs=[]
        self.in_timed_request=False;self.runtime_captures={}
        directory.mkdir(exist_ok=False)

    def __call__(self, graph, inputs, **kwargs):
        if len(self.graphs)>=self.max_graphs:
            raise RuntimeError('Bounded compile graph count exceeded; per-token recompilation is not admitted')
        code=graph.code;index=len(self.graphs)
        row=dict(index=index,fx_sha256=hashlib.sha256(code.encode()).hexdigest(),status='compiling')
        self.graphs.append(row);(self.directory/f'graph_{index}.py').write_text(code)
        self.save();started=time.monotonic()
        try:
            compiled=self.backend(graph,inputs,**kwargs)
        except Exception as exc:
            row.update(status='failed',error=repr(exc));self.save();raise
        row.update(status='backend_returned',backend_seconds=time.monotonic()-started)
        self.save();return compiled

    def save(self):
        (self.directory/'compile.json').write_text(json.dumps(dict(
            schema='tilemega.as2.torchair_compilation.v1',backend=self.identity,
            fullgraph=True,dynamic=False,suppress_errors=False,max_graphs=self.max_graphs,
            graphs=self.graphs,runtime_captures=self.runtime_captures),indent=2)+'\n')

    def check_timed(self, prepared_graphs):
        if not prepared_graphs or len(self.graphs)!=prepared_graphs:
            raise RuntimeError('Compilation occurred in a timed request; preserve samples but reject performance admission')
        if self.identity.get('name')=='npugraph_ex.get_npu_backend' and not self.runtime_captures:
            raise RuntimeError('No actual NPUGraph capture was observed')
        self.save()

    def guard_graph_runtime(self, graph_class):
        original=graph_class.compile
        audit=self
        def checked(graph,*args,**kwargs):
            before=len(graph._graphs_meta)
            key=original(graph,*args,**kwargs)
            if graph.fallback_to_eager or graph.config.get('force_eager',False):
                raise RuntimeError('NPUGraph eager fallback is forbidden in this compiled baseline')
            count=len(graph._graphs_meta)
            if audit.in_timed_request and count!=before:
                raise RuntimeError('NPUGraph captured a new shape during request timing')
            audit.runtime_captures[str(id(graph))]=count
            return key
        graph_class.compile=checked


def npu_compile(model, directory, backend_name='torchair-ge'):
    import torch
    import torch_npu
    if backend_name=='torchair-ge':
        import torchair as implementation
        backend=implementation.get_npu_backend();name='torchair.get_npu_backend'
    elif backend_name=='npugraph-ex':
        import npugraph_ex as implementation
        config=implementation.CompilerConfig();config.mode='npugraph_ex'
        config.force_eager=False;config.capture_limit=4;config.force_recapture=False
        backend=implementation.get_npu_backend(compiler_config=config);name='npugraph_ex.get_npu_backend'
    else:raise ValueError('Unknown NPU compiler backend')
    # Explicit factories avoid torch_npu's optional eager-only 'npu' fallback.
    module=Path(implementation.__file__).resolve()
    audit=CompileAudit(backend,directory,dict(name=name,
        module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
        torch=torch.__version__,torch_npu=torch_npu.__version__))
    if backend_name=='npugraph-ex':
        from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
        audit.guard_graph_runtime(AclGraph)
    torch._dynamo.config.suppress_errors=False
    return torch.compile(model,backend=audit,fullgraph=True,dynamic=False),audit
