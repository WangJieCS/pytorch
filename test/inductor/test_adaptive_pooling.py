# Owner(s): ["module: inductor"]

import torch
import torch.nn.functional as F
from torch._inductor.test_case import TestCase


class AdaptivePoolingTest(TestCase):
    def test_adaptive_avg_pool_noncontiguous_input_fused_argmax(self):
        torch.manual_seed(123)
        x = torch.rand((4, 4, 3), dtype=torch.float64).transpose(0, 1)
        self.assertFalse(x.is_contiguous())

        for output_size in (2, 3):
            with self.subTest(output_size=output_size):

                def fn(inp):
                    return torch.argmax(F.adaptive_avg_pool1d(inp, output_size))

                expected = fn(x)
                actual = torch.compile(fn, backend="inductor")(x)
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
