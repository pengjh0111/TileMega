# SPDX-License-Identifier: BSD-3-Clause
"""Validate native runner arguments on the CPU before creating weights."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
from pathlib import Path


def validate_native_arguments(runner, arguments):
    runner = Path(runner)
    source = runner.read_bytes()
    module = ast.parse(source, filename=str(runner))
    main = next((node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'main'), None)
    if main is None:
        raise ValueError('native runner has no main argument parser')
    prefix = []
    for original in main.body:
        node = copy.deepcopy(original)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (isinstance(call.func, ast.Attribute) and
                    isinstance(call.func.value, ast.Name) and
                    call.func.value.id == 'parser' and call.func.attr == 'parse_args'):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ValueError('unsupported native argument assignment')
                call.args = [ast.Name(id='argv', ctx=ast.Load())]
                prefix.extend([node, ast.Return(value=ast.Name(
                    id=node.targets[0].id, ctx=ast.Load()))])
                break
        # Only parser construction and argument declarations execute here.
        construction = (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == 'argparse'
                        and node.value.func.attr == 'ArgumentParser')
        declaration = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Attribute)
                       and isinstance(node.value.func.value, ast.Name)
                       and node.value.func.value.id == 'parser'
                       and node.value.func.attr == 'add_argument')
        if not (construction or declaration):
            raise ValueError('native runner argument prefix has unsupported statements')
        prefix.append(node)
    else:
        raise ValueError('native runner has no direct parser.parse_args call')
    function = ast.FunctionDef(name='parse', args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg='argv')], kwonlyargs=[], kw_defaults=[],
        defaults=[]), body=prefix, decorator_list=[])
    namespace = dict(argparse=argparse, Path=Path, __doc__=ast.get_docstring(module))
    executable = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(executable, str(runner), 'exec'), namespace)
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            parsed = namespace['parse'](list(arguments))
    except SystemExit as error:
        raise ValueError('native runner rejected arguments: ' + errors.getvalue().strip()) from error
    return dict(runner=str(runner.resolve()),
                runner_sha256=hashlib.sha256(source).hexdigest(),
                arguments=list(arguments),
                parsed={key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(parsed).items()})
