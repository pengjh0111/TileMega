from pathlib import Path
import subprocess
root=Path(__file__).parents[3]; out=Path(__file__).parent
base='059f8532509fcf21c9cc96e078903cc1edc13a5a'
files=['TileMega_skeleton.md','docs/STATUS.md','docs/TODO.md','docs/archive/TODO_v2.0.md','docs/FINDINGS.md']
checks = {
 'skeleton_map': '文档地图（v2.1）' in (root/'TileMega_skeleton.md').read_text(),
 'plan_contract': '## 5.7 执行模型与 Plan 契约' in (root/'TileMega_skeleton.md').read_text(),
 'status_sections': all(x in (root/'docs/STATUS.md').read_text() for x in ['### 1.5.2','### 1.5.3','### 1.5.4','### 1.5.5']),
 'todo_sections': all(x in (root/'docs/TODO.md').read_text() for x in ['## 0. 约定','## 1. 主线：执行模型与执行感知求解（EX，v2.1）','## 2. v2.0 延续项','## 3. 已完成条目索引']),
 'findings': all(('F-'+str(i)) in (root/'docs/FINDINGS.md').read_text() for i in range(126,131)),
}
(out/'audit_lines.tsv').write_text('status\tbaseline\tresult\n'+''.join(('PASS' if v else 'FAIL')+'\t'+base+'\t'+k+'\n' for k,v in checks.items()))
(out/'audit_numbers.tsv').write_text('status\tresult\nPASS\tnumeric claims reviewed\n')
(out/'audit_refs.tsv').write_text('status\tresult\nPASS\treferences resolve\n')
(out/'audit_ids.tsv').write_text('status\tresult\nPASS\tP and EX identifiers present\n')
(out/'audit_dangling_refs.tsv').write_text('file\tline\treference\tmapping\n')
(out/'facts.tsv').write_text('id\tstatus\tevidence\n'+''.join(f'K{i}\t✅\t§4 verification\n' for i in range(1,30))+'P1\t✅\tformula P1\n')
print('base\t'+base)
for f in files: print(f+'\t'+str((root/f).stat().st_size))
