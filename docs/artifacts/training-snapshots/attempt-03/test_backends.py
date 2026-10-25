# SPDX-License-Identifier: Apache-2.0
"""Contract, double finite differences, and actual Qwen layer/parameter checks."""
import unittest

import torch

from .backends import GDNBackend, use_backend
from .validation import inputs


class BackendTests(unittest.TestCase):
    def test_double_gradcheck_with_normalization_and_state(self):
        args = tuple(x.double().requires_grad_() for x in inputs(
            batch=1, length=3, key_heads=1, value_heads=2, key_dim=2, value_dim=3))
        operation = GDNBackend(chunk_size=2)
        self.assertTrue(torch.autograd.gradcheck(
            lambda q, k, v, g, beta, state: operation(
                q, k, v, g, beta, initial_state=state, output_final_state=True,
                scale=0.37, use_qk_l2norm_in_kernel=True),
            args, fast_mode=True, atol=1e-5, rtol=1e-3))

    def test_state_layout_handoff_and_gradients(self):
        args = tuple(x.requires_grad_() for x in inputs(length=13))
        operation = GDNBackend(chunk_size=4)
        full = operation(*args[:5], initial_state=args[5], output_final_state=True,
                         state_v_first=False)
        first, state = operation(
            *(x[:, :5] for x in args[:5]), initial_state=args[5].transpose(-1, -2),
            output_final_state=True, state_v_first=True)
        tail, state = operation(
            *(x[:, 5:] for x in args[:5]), initial_state=state,
            output_final_state=True, state_v_first=True)
        torch.testing.assert_close(torch.cat((first, tail), 1), full[0], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(state.transpose(-1, -2), full[1], atol=2e-6, rtol=1e-5)
        grad_full = torch.autograd.grad(full[0].square().sum() + full[1].square().sum(),
                                        args, retain_graph=True)
        grad_split = torch.autograd.grad(first.square().sum() + tail.square().sum()
                                         + state.square().sum(), args)
        for left, right in zip(grad_full, grad_split):
            torch.testing.assert_close(left, right, atol=3e-5, rtol=3e-4)

    def test_reference_scaling_and_final_state_option(self):
        args = inputs(length=9)
        for scale in (None, 0.27):
            reference = GDNBackend("recurrent")(
                *args[:5], scale=scale, initial_state=args[5],
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            for name in ("torch", "deltaforge"):
                actual = GDNBackend(name, chunk_size=4)(
                    *args[:5], scale=scale, initial_state=args[5],
                    output_final_state=True, use_qk_l2norm_in_kernel=True)
                for left, right in zip(actual, reference):
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
        self.assertIsNone(GDNBackend()(*args[:5])[1])

    def test_rejects_unsupported_and_invalid(self):
        args = inputs(length=9)
        operation = GDNBackend()
        for options in ({"cu_seqlens": torch.tensor([0, 9])},
                        {"use_beta_sigmoid": True}, {"head_first": True}):
            with self.assertRaises(NotImplementedError):
                operation(*args[:5], **options)
        for options in ({"scale": -1}, {"initial_state": args[5].bfloat16()}):
            with self.assertRaises(ValueError):
                operation(*args[:5], **options)

    def test_collinear_keys_weak_decay_forward_backward(self):
        # Actual models can produce aligned keys and nearly open gates. A dense
        # nilpotent power expansion suffers catastrophic FP32 cancellation here.
        from .ops import GDN, recurrent, block_inverse_unit_lower
        torch.manual_seed(23)
        lower = torch.ones(32, 32).tril(-1) * 0.999
        target = torch.linalg.solve_triangular(torch.eye(32) + lower, torch.eye(32),
                                               upper=False, unitriangular=True)
        torch.testing.assert_close(block_inverse_unit_lower(lower), target, atol=2e-6, rtol=2e-5)
        q = torch.ones(1, 64, 1, 4) / 2
        args = (q, q.clone(), torch.randn(1, 64, 1, 3),
                torch.full((1, 64, 1), -1e-6), torch.full((1, 64, 1), 0.999),
                torch.zeros(1, 1, 4, 3))
        args = tuple(x.requires_grad_() for x in args)
        reference = recurrent(*args)
        actual = GDN(32)(*args)
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)
        target_grad = torch.autograd.grad(sum(x.square().sum() for x in reference), args)
        actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), args)
        for left, right in zip(actual_grad, target_grad):
            torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-3)

    def test_real_qwen_layer_outputs_gradients_restore_checkpoint(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
        torch.manual_seed(15)
        config = Qwen3_5TextConfig(hidden_size=16, linear_num_key_heads=1,
                                   linear_num_value_heads=2, linear_key_head_dim=4,
                                   linear_value_head_dim=6)
        layer = Qwen3_5GatedDeltaNet(config, layer_idx=0)
        layer.causal_conv1d_fn = None
        original = (layer.chunk_gated_delta_rule, layer.recurrent_gated_delta_rule)
        x = torch.randn(2, 7, 16, requires_grad=True)
        with use_backend(layer, "torch", chunk_size=4):
            reference = layer(x)
            gradients = torch.autograd.grad(reference.square().sum(), (x, *layer.parameters()))
        for checkpoint in (False, True):
            with use_backend(layer, chunk_size=4):
                if checkpoint:
                    from torch.utils.checkpoint import checkpoint as recompute
                    actual = recompute(layer, x, use_reentrant=False)
                else:
                    actual = layer(x)
                candidate = torch.autograd.grad(actual.square().sum(), (x, *layer.parameters()))
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
                for left, right in zip(candidate, gradients):
                    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-4)
        self.assertEqual(original, (layer.chunk_gated_delta_rule, layer.recurrent_gated_delta_rule))
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with use_backend(layer):
                raise RuntimeError("restore")
        self.assertEqual(original, (layer.chunk_gated_delta_rule, layer.recurrent_gated_delta_rule))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
