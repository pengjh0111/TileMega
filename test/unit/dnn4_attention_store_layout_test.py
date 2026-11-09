#!/usr/bin/env python3
"""Prove packed quad stores cover the original attention output coordinates."""
import unittest


class StoreLayoutTest(unittest.TestCase):
    def test_every_output_once_with_contiguous_aligned_vectors(self):
        checked = 0
        for rows in (16, 32, 64, 128):
            actual = {}
            for image in range(2):
                for head in range(12):
                    for query in range(0, 128, rows):
                        for warp in range(rows // 16):
                            for group in range(2):
                                for half in range(2):
                                    # Symbols name the original MMA coordinates;
                                    # the reference is the dense output tensor.
                                    words = [[tuple((query + warp*16 + lane//4 + half*8,
                                        group*32 + block*8 + 2*(lane % 4) + e)
                                        for e in range(2)) for block in range(4)]
                                        for lane in range(32)]
                                    first = [list(w) for w in words]
                                    for lane in range(32):
                                        for index in range(4):
                                            if (index & 1) != (lane & 1):
                                                first[lane][index] = words[lane ^ 1][index ^ 1]
                                    second = [list(w) for w in first]
                                    for lane in range(32):
                                        for index in range(4):
                                            if (index & 2) != (lane & 2):
                                                second[lane][index] = first[lane ^ 2][index ^ 2]
                                    for lane, packed in enumerate(second):
                                        row = query + warp*16 + lane//4 + 8*half
                                        col = group*32 + 8*(lane % 4)
                                        address = ((image*128 + row)*12 + head)*64 + col
                                        self.assertEqual(address % 8, 0)
                                        for e, coordinate in enumerate(x for pair in packed for x in pair):
                                            self.assertEqual(coordinate, (row, col + e))
                                            self.assertNotIn(address + e, actual)
                                            actual[address + e] = coordinate
                                            checked += 1
            self.assertEqual(len(actual), 2*128*12*64)
            self.assertEqual(set(actual), set(range(2*128*12*64)))
        self.assertEqual(checked, 786432)


if __name__ == '__main__':
    unittest.main()
