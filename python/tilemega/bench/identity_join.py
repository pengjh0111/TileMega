#!/usr/bin/env python3
"""Reject mixed artifacts/executions and keep instrumented data out of E2E."""
from tilemega.build.identity import digest

def checked_join(left,right,*,trace_table=False):
    for record in (left,right):
        execution=record['execution_identity']
        if execution['execution_id']!=digest({k:v for k,v in execution.items() if k!='execution_id'}):
            raise ValueError('execution identity was modified')
        if execution['trace'] and not trace_table:
            raise ValueError('trace timing cannot enter a performance table')
    if left['execution_identity']['execution_id']!=right['execution_identity']['execution_id']:
        raise ValueError('cannot join different artifact/execution identities')
    return dict(left=left,right=right)
