"""Native receipt binding rejects absent/ambiguous entries before reporting."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from tilemega.dnn.execution import execution_identity


class NativeIdentity(unittest.TestCase):
    def test_native_entry_and_legacy_delegation(self):
        plan = Mock()
        identity = dict(plan={'native_forward': {}}, artifact_id='artifact', trace=False,
            resources={'typed_sm90_forward_native_kernel':
                       dict(name='typed_sm90_forward_native_kernel', spill=False)})
        value = execution_identity(plan, identity, 1)
        self.assertEqual(value['kernel'], 'typed_sm90_forward_native_kernel')
        self.assertTrue(value['native_forward'])
        plan.execution_identity.assert_not_called()
        for resources in ({}, {**identity['resources'], 'other_sm90_forward_native_kernel':
                               dict(name='other_sm90_forward_native_kernel', spill=True)}):
            changed = copy.deepcopy(identity);changed['resources'] = resources
            with self.assertRaises(ValueError):execution_identity(plan, changed, 1)
        identity['plan'] = {}
        self.assertEqual(execution_identity(plan, identity, 1), plan.execution_identity.return_value)


if __name__ == '__main__':unittest.main()
