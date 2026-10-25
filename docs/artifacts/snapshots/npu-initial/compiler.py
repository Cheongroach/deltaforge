# SPDX-License-Identifier: Apache-2.0
"""torch.export/FX domain lowering followed by a vendor compiler."""
import math
import time

import torch

from .ops import GDN


def lower_unit_lower_solve(module):
    """Replace the semantic solve with O(log C) nilpotent matrix doubling.

    A general triangular solve is deliberately NOT rewritten. Only the
    domain op, whose contract supplies a unit diagonal and strictly lower L,
    can prove nilpotency. torch.export guards the fixed shape and dtype.
    """
    graph = module.graph
    count = 0
    before = len(list(graph.nodes))
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.deltaforge.unit_lower_solve.default:
            continue
        lower, rhs = node.args
        meta = lower.meta["val"]
        size = int(meta.shape[-1])
        with graph.inserting_before(node):
            eye = graph.call_function(
                torch.ops.aten.eye.default,
                (size,),
                {"dtype": meta.dtype, "device": meta.device, "pin_memory": False},
            )
            if size == 1:
                result = rhs
            else:
                power = graph.call_function(torch.ops.aten.neg.default, (lower,))
                inverse = graph.call_function(torch.ops.aten.add.Tensor, (eye, power))
                for _ in range(1, math.ceil(math.log2(size))):
                    power = graph.call_function(torch.ops.aten.matmul.default, (power, power))
                    update = graph.call_function(torch.ops.aten.matmul.default, (power, inverse))
                    inverse = graph.call_function(torch.ops.aten.add.Tensor, (inverse, update))
                result = graph.call_function(torch.ops.aten.matmul.default, (inverse, rhs))
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        count += 1
    graph.eliminate_dead_code()
    graph.lint()
    module.recompile()
    return {
        "lowered_domain_ops": count,
        "fx_nodes_before": before,
        "fx_nodes_after": len(list(graph.nodes)),
        "algorithm": "finite_nilpotent_doubling",
        "approximate_gate_truncation": False,
    }


def export_lower(args, chunk_size=32):
    start = time.perf_counter()
    exported = torch.export.export(GDN(chunk_size, semantic_solve=True), args)
    module = exported.module()
    info = lower_unit_lower_solve(module)
    info["export_and_lower_seconds"] = time.perf_counter() - start
    return module, info


def compile_gdn(args, chunk_size=32):
    module, info = export_lower(args, chunk_size)
    device = args[0].device.type
    if device == "npu":
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        backend = torchair.get_npu_backend(compiler_config=config)
        info["backend"] = "torchair"
    elif device in ("cpu", "cuda"):
        backend = "inductor"
        info["backend"] = "inductor"
    else:
        raise ValueError(f"Unsupported compiler target {device}")
    compiled = torch.compile(module, backend=backend, fullgraph=True, dynamic=False)
    start = time.perf_counter()
    compiled(*args)
    if device != "cpu":
        getattr(torch, device).synchronize()
    info["compile_and_first_execution_seconds"] = time.perf_counter() - start
    return compiled, info, module.code
