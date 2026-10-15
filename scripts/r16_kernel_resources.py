"""Parse compiler resources for the serving entry actually measured."""
import re


def resource_functions(log):
    entries = {}
    functions = {}
    active = None
    for line in log.splitlines():
        compiled = re.search(r"Compiling entry function '([^']+)'", line)
        if compiled:
            name = compiled[1]
            if name in entries:
                raise ValueError('ambiguous duplicate CUDA entry: '+name)
            entries[name] = {'symbol': name}
            functions.setdefault(name, entries[name])
        properties = re.search(r'Function properties for (\S+)', line)
        if properties:
            active = properties[1]
            functions.setdefault(active, {'symbol': active})
        spills = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if spills and active is not None:
            # A device helper can be reported under multiple entry functions.
            # Preserve its worst occurrence rather than overwriting a spill
            # from an earlier entry with a later zero-spill occurrence.
            for key, value in (('stack_bytes', spills[1]),
                               ('spill_store_bytes', spills[2]),
                               ('spill_load_bytes', spills[3])):
                functions[active][key] = max(functions[active].get(key, 0), int(value))
        registers = re.search(r'Used (\d+) registers', line)
        if registers and active is not None:
            functions[active]['registers'] = int(registers[1])
    return entries, functions


def serving_entries(log):
    return resource_functions(log)[0]


def validate_selected_entry(log, executor, device_loop):
    if executor not in ('L1', 'L2') or device_loop not in (0, 1):
        raise ValueError('unsupported serving executor or device loop')
    kernel = ('tilemega_l1_loop_kernel' if executor == 'L1' else
              'tilemega_loop_kernel') if device_loop else (
              'tilemega_l1_kernel' if executor == 'L1' else 'tilemega_l2_kernel')
    entries, functions = resource_functions(log)
    matches = [row for name, row in entries.items()
               if name == kernel or name.startswith(
                   '_ZN8tilemega7codegen'+str(len(kernel))+kernel+'E')]
    if len(matches) != 1:
        raise ValueError('selected CUDA entry is absent or ambiguous: '+kernel)
    selected = matches[0]
    for row in entries.values():
        if not all(key in row for key in ('stack_bytes', 'spill_store_bytes',
                                         'spill_load_bytes', 'registers')):
            raise ValueError('incomplete CUDA entry resource receipt')
    non_entry_spills = [row for name, row in functions.items()
                        if name not in entries and (row.get('spill_store_bytes', 0)
                                                   or row.get('spill_load_bytes', 0))]
    if non_entry_spills:
        raise ValueError('non-entry device function spills require callgraph attribution')
    if selected['spill_store_bytes'] or selected['spill_load_bytes']:
        raise ValueError('selected CUDA entry has compiler spills: '+kernel)
    return {'evidence': 'verified compiler entry resources; runtime selection must be checked separately',
            'executor': executor, 'device_loop': device_loop,
            'selected_kernel': kernel, 'selected_entry': selected,
            'all_entries': list(entries.values()),
            'non_entry_spills': non_entry_spills,
            'whole_binary_entry_spills_zero': all(
                not row['spill_store_bytes'] and not row['spill_load_bytes']
                for row in entries.values()),
            'scope': 'selected entry only; not a claim about whole-binary local memory or unexecuted entries'}
