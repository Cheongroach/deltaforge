# SPDX-License-Identifier: Apache-2.0
"""Independent numerical and contract checks; also callable on either device."""
import json
import unittest

import torch

from .compiler import export_lower
from .ops import GDN, inverse_unit_lower, recurrent, validate_inputs


def inputs(batch=2, length=17, key_heads=2, value_heads=4, key_dim=8,
           value_dim=12, device="cpu", dtype=torch.float32, seed=20261025):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def normal(*shape):
        return torch.randn(*shape, generator=generator)
    q = torch.nn.functional.normalize(normal(batch, length, key_heads, key_dim), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(normal(batch, length, key_heads, key_dim), dim=-1).to(dtype)
    v = normal(batch, length, value_heads, value_dim).to(dtype)
    g = -torch.rand(batch, length, value_heads, generator=generator) * 0.15
    beta = torch.rand(batch, length, value_heads, generator=generator)
    state = normal(batch, value_heads, key_dim, value_dim) * 0.05
    return tuple(x.to(device) for x in (q, k, v, g, beta, state))


def errors(actual, expected):
    a, b = actual.float(), expected.float()
    diff = a - b
    return {
        "max_abs": diff.abs().max().item(),
        "relative_l2": (diff.norm() / b.norm().clamp(min=1e-12)).item(),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def verify_pair(actual, expected, dtype=torch.float32):
    atol, rtol = (5e-3, 1e-2) if dtype == torch.bfloat16 else (5e-5, 5e-4)
    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(actual[1], expected[1], atol=5e-5, rtol=5e-4)
    return {"output": errors(actual[0], expected[0]), "state": errors(actual[1], expected[1])}


def device_validation(device, dtype=torch.float32):
    rows = []
    # Strong/zero decay, odd tail, GQA, K!=V, initial state and mixed lengths.
    for length, chunk in ((1, 1), (7, 4), (17, 8), (63, 16), (65, 32)):
        args = inputs(length=length, device=device, dtype=dtype)
        lengths = torch.tensor([length, max(0, length - 5)], device=device)
        reference = recurrent(*args, lengths=lengths)
        candidate = GDN(chunk)(*args, lengths=lengths)
        row = {"length": length, "chunk": chunk, "dtype": str(dtype), "device": device}
        row.update(verify_pair(candidate, reference, dtype))
        rows.append(row)
    for gate in (0.0, -80.0):
        args = list(inputs(length=33, device=device, dtype=dtype))
        args[3] = torch.full_like(args[3], gate)
        rows.append({"gate": gate, **verify_pair(GDN(16)(*args), recurrent(*args), dtype)})
    return rows


class NumericalTests(unittest.TestCase):
    def test_oracle_contracts(self):
        rows = device_validation("cpu")
        self.assertEqual(len(rows), 7)

    def test_nilpotent_inverse(self):
        for size in (1, 2, 3, 8, 17, 32, 64):
            torch.manual_seed(size)
            lower = (torch.randn(2, size, size, dtype=torch.float64) * 0.03).tril(-1)
            inverse = inverse_unit_lower(lower)
            identity = torch.eye(size, dtype=torch.float64).expand_as(lower)
            torch.testing.assert_close((identity + lower) @ inverse, identity, atol=1e-12, rtol=1e-12)

    def test_inverse_gradient(self):
        # Double precision finite-difference check on the transformed primitive.
        lower = (torch.randn(1, 4, 4, dtype=torch.float64) * 0.02).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: inverse_unit_lower(x.tril(-1)), (lower,)))

    def test_full_backward(self):
        originals = inputs(length=11)
        left = tuple(x.detach().requires_grad_() for x in originals)
        right = tuple(x.detach().requires_grad_() for x in originals)
        expected = recurrent(*left)
        actual = GDN(4)(*right)
        weights = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
        grad_left = torch.autograd.grad(sum((x * w).sum() for x, w in zip(expected, weights)), left)
        grad_right = torch.autograd.grad(sum((x * w).sum() for x, w in zip(actual, weights)), right)
        for a, b in zip(grad_left, grad_right):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=5e-4)

    def test_state_handoff(self):
        args = inputs(length=35)
        output, final = GDN(8)(*args)
        first = tuple(x[:, :13].contiguous() for x in args[:5]) + (args[5],)
        second = tuple(x[:, 13:].contiguous() for x in args[:5])
        out_a, state_a = GDN(8)(*first)
        out_b, state_b = GDN(8)(*second, state_a)
        verify_pair((torch.cat((out_a, out_b), dim=1), state_b), (output, final))

    def test_initial_state_not_mutated(self):
        args = inputs(length=9)
        saved = args[5].clone()
        GDN(8)(*args)
        torch.testing.assert_close(args[5], saved, atol=0, rtol=0)

    def test_lowering(self):
        args = inputs(length=17)
        graph, info = export_lower(args, chunk_size=8)
        self.assertEqual(info["lowered_domain_ops"], 1)
        self.assertNotIn("deltaforge.unit_lower_solve", graph.code)
        verify_pair(graph(*args), recurrent(*args))

    def test_reject_invalid_contract(self):
        args = list(inputs())
        validate_inputs(*args)
        args[3] = torch.ones_like(args[3])
        with self.assertRaises(ValueError):
            validate_inputs(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
