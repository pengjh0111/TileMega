"""Bind native forward timing receipts to their actual compiled entry point."""
from tilemega.build.identity import digest


def execution_identity(plan, identity, mode):
    if 'native_forward' not in identity['plan']:
        return plan.execution_identity(mode)
    if mode not in (1, 2):
        raise ValueError('native forward needs L1 or L2 mode')
    kernels = [row for name, row in identity['resources'].items()
               if 'sm90_forward_native_kernel' in name]
    if len(kernels) != 1:
        raise ValueError('native forward requires one identified compiled entry point')
    # Both public modes dispatch the same typed forward program. Its entry is
    # distinct from the legacy tilemega_l1_kernel and tilemega_l2_kernel names.
    result = dict(artifact_id=identity['artifact_id'], executor={1: 'L1', 2: 'L2'}[mode],
                  loop=False, pdl=False, kernel=kernels[0]['name'],
                  spill=kernels[0]['spill'], trace=identity['trace'], native_forward=True)
    result['execution_id'] = digest(result)
    return result
