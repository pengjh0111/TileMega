#!/usr/bin/env python3
"""Compare source, resources, and normalized SASS for existing kernels."""
import argparse,difflib,json,re,subprocess
from pathlib import Path
def sass(binary):
    text=subprocess.check_output(['cuobjdump','-sass',str(binary)],text=True)
    functions={};name=None
    for line in text.splitlines():
        m=re.search(r'Function\s*:\s*(\S+)',line)
        if m:name=m.group(1);functions[name]=[]
        elif name:
            instruction=re.sub(r'/\*[^*]*\*/','',line).strip()
            if instruction and not instruction.startswith(('.', 'Fatbin','code for')):
                functions[name].append(instruction)
    return functions
def resources(path):
    result={};name=None;entry=None
    for line in path.read_text().splitlines():
        kernel=re.search(r"Compiling entry function\s+'([^']+)'",line)
        if kernel:entry=kernel.group(1)
        m=re.search(r"(?:Compiling entry function|Function properties for)\s+'?([^'\s]+)",line)
        # ptxas emits called-function resources once per kernel context.
        # A new loop kernel must not change the comparison of an old caller.
        if m:name=(entry or '<unscoped>')+'::'+m.group(1);result.setdefault(name,[])
        if name and re.search(r'(Used \d+ registers|bytes stack frame|bytes spill|bytes smem)',line):
            result[name].append(re.sub(r'\s+',' ',line.strip()))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ignore-watchdog-macro',action='store_true');p.add_argument('--ignore-pdl-macro',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    def source(so):
        text=Path(str(so)+'.cu').read_text()
        if a.ignore_watchdog_macro:text=re.sub(r'^#define TILEMEGA_WATCHDOG [01]\n','',text,flags=re.M)
        if a.ignore_pdl_macro:text=re.sub(r'^#define TILEMEGA_PDL [01]\n','',text,flags=re.M)
        return text
    old,new=sass(a.reference),sass(a.candidate);differences={}
    for name,lines in old.items():
        # All reference functions, including called task bodies, are compared.
        if lines!=new.get(name):differences[name]=list(difflib.unified_diff(lines,new.get(name,[])))
    resource_old=resources(Path(str(a.reference)+'.ptxas.log'))
    resource_new=resources(Path(str(a.candidate)+'.ptxas.log'))
    report=dict(source_equal=source(a.reference)==source(a.candidate),
                sass_equal=not differences,
                resources_equal=all(value==resource_new.get(name) for name,value in resource_old.items()),
                existing_functions=list(old),new_functions=sorted(set(new)-set(old)),
                resource_scope='entry kernel and called function; new kernel contexts excluded')
    (a.out/'sass_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    (a.out/'resources.json').write_text(json.dumps(dict(reference=resource_old,candidate=resource_new),indent=2)+'\n')
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));raise SystemExit(0 if all(report[k] for k in ('source_equal','sass_equal','resources_equal')) else 1)
if __name__=='__main__':main()
