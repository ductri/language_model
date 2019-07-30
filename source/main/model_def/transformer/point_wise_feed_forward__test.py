import unittest

import torch

from model_def.transformer.point_wise_feed_forward import PointWiseFeedForward


class TestPointWiseFeedForward(unittest.TestCase):

    def test_forward(self):
        mha = PointWiseFeedForward(d_model=512)
        x = torch.rand(5, 100, 512)
        output = mha(x)
        self.assertListEqual(list(output.size()), [5, 100, 512])


if __name__ == '__main__':
    unittest.main()

