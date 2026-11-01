"""Tensor parallel shard recipes (SM90R5 B2.6): the ranks' packs reassemble TP1's."""
import unittest

import torch

from tilemega.serving.weights import _packed_gpu, _shard_parts


def shard(recipe, axis, parts):
    return dict(recipe, shard=dict(axis=axis, parts=parts))


class TpShardRecipeTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        # Qwen3-1.7B's geometry scaled down: 8 KV heads of 2 queries, head_dim 4.
        self.hkv, self.qperkv, self.dim, self.hidden, self.inter = 8, 2, 4, 32, 48
        self.tensors = {
            'q': torch.randn(self.hkv * self.qperkv * self.dim, self.hidden, generator=g),
            'k': torch.randn(self.hkv * self.dim, self.hidden, generator=g),
            'v': torch.randn(self.hkv * self.dim, self.hidden, generator=g),
            'gate': torch.randn(self.inter, self.hidden, generator=g),
            'up': torch.randn(self.inter, self.hidden, generator=g),
            'o': torch.randn(self.hidden, self.hkv * self.qperkv * self.dim, generator=g),
            'down': torch.randn(self.hidden, self.inter, generator=g),
            'head': torch.randn(40, self.hidden, generator=g),
            'norm': torch.randn(self.hidden, generator=g),
        }
        self.tensors = {k: v.to(torch.bfloat16) for k, v in self.tensors.items()}
        self.source = self.tensors.__getitem__

    def qkv(self, hkv):
        return dict(kind='qkv_group_interleave', sources=['q', 'k', 'v'], hkv=hkv,
                    qperkv=self.qperkv, head_dim=self.dim)

    def assert_reassembles(self, whole, per_rank, parts, axis):
        ranks = [_packed_gpu(per_rank, self.source, r) for r in range(parts)]
        self.assertTrue(torch.equal(torch.cat(ranks, dim=axis), _packed_gpu(whole, self.source)))

    def test_column_parallel_recipes(self):
        for parts in (2, 4):
            self.assert_reassembles(self.qkv(self.hkv), shard(self.qkv(self.hkv // parts), 0, parts),
                                    parts, 0)
            gu = dict(kind='gate_up_interleave', sources=['gate', 'up'], u=4)
            self.assert_reassembles(gu, shard(gu, 0, parts), parts, 0)
            head = dict(kind='alias', source='head')
            self.assert_reassembles(head, shard(head, 0, parts), parts, 0)

    def test_row_parallel_recipes(self):
        for parts in (2, 4):
            for name in ('o', 'down'):
                alias = dict(kind='alias', source=name)
                self.assert_reassembles(alias, shard(alias, 1, parts), parts, 1)

    def test_shard_under_fold_and_pages(self):
        # The norm spans the unsharded K, so folding commutes with an axis-0 shard.
        for parts in (2, 4):
            inner = shard(self.qkv(self.hkv // parts), 0, parts)
            fold = dict(kind='fold_rmsnorm', norm='norm', source=inner)
            whole = dict(kind='fold_rmsnorm', norm='norm', source=self.qkv(self.hkv))
            self.assert_reassembles(whole, fold, parts, 0)
            self.assertEqual(_shard_parts(dict(kind='tile_pages', source=fold)), {parts})
            # Tile pages of a rank's shard equal tile pages of its slice of TP1.
            for rank in range(parts):
                rows = self.hkv * (self.qperkv + 2) * self.dim // parts
                expected = _packed_gpu(whole, self.source)[rank * rows:(rank + 1) * rows]
                paged = _packed_gpu(dict(kind='tile_pages', tile_n=8, tile_k=16, source=fold),
                                    self.source, rank)
                reference = _packed_gpu(dict(kind='tile_pages', tile_n=8, tile_k=16,
                                             source=dict(kind='alias', source='x')),
                                        {'x': expected}.__getitem__)
                self.assertTrue(torch.equal(paged, reference))

    def test_invalid_shards_are_rejected(self):
        alias = dict(kind='alias', source='gate')
        with self.assertRaisesRegex(ValueError, 'does not split'):
            _packed_gpu(shard(alias, 0, 5), self.source, 0)
        with self.assertRaisesRegex(ValueError, 'invalid shard'):
            _packed_gpu(shard(alias, 0, 2), self.source, 2)
        with self.assertRaisesRegex(ValueError, 'cannot carry'):
            _packed_gpu(dict(kind='fold_rmsnorm', norm='norm', source=alias,
                             shard=dict(axis=0, parts=2)), self.source, 0)
        self.assertEqual(_shard_parts(alias), set())


if __name__ == '__main__':
    unittest.main()
