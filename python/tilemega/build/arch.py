"""Compiler targets for base architecture labels, matching ArchDispatch.h."""


def nvcc_arch(arch):
    tag = str(arch)
    if not tag.startswith('sm_'):
        tag = 'sm_' + tag
    return {'sm_90': 'sm_90a', 'sm_100': 'sm_100a'}.get(tag, tag)


def nvcc_flag(arch):
    tag = nvcc_arch(arch)
    if tag.endswith('a'):
        return '-gencode=arch=compute_' + tag[3:] + ',code=' + tag
    return '-arch=' + tag


def compiler_arch(options):
    for index, value in enumerate(options):
        if value.startswith('-arch='):
            return value.split('=', 1)[1]
        if value == '-arch' and index + 1 < len(options):
            return options[index + 1]
        if value.startswith(('-gencode=', '--generate-code=')) or value in ('-gencode', '--generate-code'):
            specification = value.split('=', 1)[1] if '=' in value else (
                options[index + 1] if index + 1 < len(options) else '')
            code = specification.partition(',code=')[2]
            if code.startswith('sm_'):
                return code
    return None


def architecture_identity(options, target_spec):
    compiled = compiler_arch(options)
    if compiled is None:
        raise ValueError('compiler command lacks an explicit architecture')
    logical = target_spec.get('arch_tag', {'sm_90a': 'sm_90', 'sm_100a': 'sm_100'}.get(compiled, compiled))
    expected = target_spec.get('nvcc_arch', nvcc_arch(logical))
    if compiled != expected:
        raise ValueError('compiler architecture differs from the recorded target')
    return logical, compiled
