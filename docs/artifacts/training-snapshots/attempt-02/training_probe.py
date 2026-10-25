# SPDX-License-Identifier: Apache-2.0
"""Bounded real-model LoRA training and forward/backward backend comparison."""
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import torch

from .backends import GDNBackend, use_backend
from .validation import errors, inputs


def emit(event, **fields):
    row = {"event": event, "unix_seconds": time.time(), **fields}
    print("DELTAFORGE " + json.dumps(row, ensure_ascii=False), flush=True)
    return row


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def synchronize():
    torch.cuda.synchronize()


def seed(value):
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def comparison(actual, expected, tolerance):
    result = errors(actual, expected)
    if not result["finite"] or result["relative_l2"] > tolerance:
        raise AssertionError(f"Numerical gate failed ({tolerance}): {result}")
    return result


@contextlib.contextmanager
def model_backend(model, name):
    if name == "default":
        class NativeSelection:
            def stats(self):
                return {"backend": "default", "chunk_functions": sorted({
                    layer.chunk_gated_delta_rule.__module__ + "."
                    + layer.chunk_gated_delta_rule.__name__
                    for layer in model.modules()
                    if type(layer).__name__ == "Qwen3_5GatedDeltaNet"})}
        yield NativeSelection()
    else:
        with use_backend(model, name) as adapter:
            yield adapter


def kernel_probe(output):
    rows = []
    # Qwen3.5-2B's real GDN dimensions, plus a longer single-layer workload.
    for length in (256, 2048):
        args = inputs(batch=1, length=length, key_heads=16, value_heads=16,
                      key_dim=128, value_dim=128, device="cuda", dtype=torch.bfloat16)
        references = None
        for name in ("torch", "deltaforge", "compiled", "fla"):
            operation = GDNBackend(name, chunk_size=32)
            values = tuple(x.detach().requires_grad_() for x in args)
            weights = (torch.ones_like(values[2]) / values[2].numel(),
                       torch.ones_like(values[5]) / values[5].numel())
            started = time.perf_counter()
            result = operation(*values[:5], initial_state=values[5], output_final_state=True,
                               use_qk_l2norm_in_kernel=True)
            gradients = torch.autograd.grad(sum((x * w).sum() for x, w in zip(result, weights)),
                                            values)
            synchronize()
            first_ms = (time.perf_counter() - started) * 1000
            checks = {}
            if references is None:
                references = (tuple(x.detach().float() for x in result),
                              tuple(x.detach().float() for x in gradients))
            else:
                checks["output"] = [errors(x, y)
                                    for x, y in zip(result, references[0])]
                checks["gradients"] = [errors(x, y)
                                       for x, y in zip(gradients, references[1])]
            passed = (all(x["finite"] and x["relative_l2"] <= 0.05 for x in checks.get("output", []))
                      and all(x["finite"] and x["relative_l2"] <= 0.08
                              for x in checks.get("gradients", [])))
            if not passed:
                rows.append(emit("kernel_result", backend=name, length=length, checks=checks,
                                 gate_passed=False, samples=[], adapter=operation.stats()))
                (output / "kernel-results.json").write_text(json.dumps(rows, indent=2))
                del result, gradients, values
                gc.collect()
                torch.cuda.empty_cache()
                continue
            timings = []
            saved_bytes = []
            for iteration in range(12):
                values = tuple(x.detach().requires_grad_() for x in args)
                saved = []
                def pack(tensor):
                    saved.append(tensor.numel() * tensor.element_size())
                    return tensor
                synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                # Hooks are for the activation metric only; timing samples use no hooks.
                result = operation(*values[:5], initial_state=values[5], output_final_state=True,
                                   use_qk_l2norm_in_kernel=True)
                synchronize()
                forward_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                torch.autograd.grad(sum((x * w).sum() for x, w in zip(result, weights)), values)
                synchronize()
                backward_ms = (time.perf_counter() - start) * 1000
                if iteration >= 2:
                    timings.append({"forward_ms": forward_ms, "backward_ms": backward_ms,
                                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
                del result
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                values = tuple(x.detach().requires_grad_() for x in args)
                result = operation(*values[:5], initial_state=values[5], output_final_state=True,
                                   use_qk_l2norm_in_kernel=True)
                torch.autograd.grad(sum((x * w).sum() for x, w in zip(result, weights)), values)
            saved_bytes.append(sum(saved))
            rows.append(emit("kernel_result", backend=name, length=length,
                             gate_passed=True,
                             shape=[1, length, 16, 128, 128], first_forward_backward_ms=first_ms,
                             checks=checks, samples=timings, saved_tensor_bytes=saved_bytes,
                             **{"adapter": operation.stats()}))
            del result, gradients, values
            gc.collect()
            torch.cuda.empty_cache()
            (output / "kernel-results.json").write_text(json.dumps(rows, indent=2))
    (output / "kernel-results.json").write_text(json.dumps(rows, indent=2))


def make_model(model_path, initialization_seed):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    seed(initialization_seed)
    config = Qwen3_5TextConfig.from_pretrained(model_path)
    model, information = Qwen3_5ForCausalLM.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
        output_loading_info=True, local_files_only=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if information.get(key):
            raise RuntimeError(f"Pretrained loading failed: {key}={information[key][:5]}")
    # Hold these non-GDN paths constant even when FLA is installed.
    for layer in model.modules():
        if type(layer).__name__ == "Qwen3_5GatedDeltaNet":
            layer.causal_conv1d_fn = None
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                        "up_proj", "down_proj", "in_proj_qkv", "in_proj_z",
                        "in_proj_b", "in_proj_a", "out_proj"]))
    model.to("cuda")
    model.config.use_cache = False
    model.train()
    return model


def initialization_hash(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            result.update(name.encode())
            result.update(parameter.detach().cpu().float().numpy().tobytes())
    return result.hexdigest()


def layer_diagnostic(model_path, data_path, output):
    """Fixed real-layer inputs AND upstream gradient isolate local numerical error."""
    from transformers import AutoTokenizer
    from fla.modules.l2norm import l2norm as fla_l2norm
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    data = batches(data_path, tokenizer, 256)
    model = make_model(model_path, 20261025)
    layers = [layer for layer in model.modules() if type(layer).__name__ == "Qwen3_5GatedDeltaNet"]
    selected = {layers[0].layer_idx, layers[len(layers) // 2].layer_idx, layers[-1].layer_idx}
    captured = {}
    saved = []
    for layer in layers:
        original = layer.chunk_gated_delta_rule
        saved.append((layer, original))
        def capture(q, k, v, g, beta, _index=layer.layer_idx, **options):
            result = GDNBackend("torch", chunk_size=32)(q, k, v, g, beta, **options)
            if _index in selected:
                record = {"inputs": tuple(x.detach().clone() for x in (q, k, v, g, beta)),
                          "native": saved_function[_index]}
                captured[_index] = record
                def store_gradient(gradient):
                    record["upstream"] = gradient.detach().clone()
                result[0].register_hook(store_gradient)
            return result
        layer.chunk_gated_delta_rule = capture
    saved_function = {layer.layer_idx: function for layer, function in saved}
    try:
        result = model(**data["train"][0])
        result.loss.backward()
    finally:
        for layer, function in saved:
            layer.chunk_gated_delta_rule = function
    del result, model, data
    gc.collect()
    torch.cuda.empty_cache()
    rows = []
    for index, record in sorted(captured.items()):
        reference = None
        for name in ("torch32", "torch64", "deltaforge", "fla_common", "native",
                     "fla_norm_torch", "fp32_norm_torch"):
            try:
                values = tuple(x.detach().requires_grad_() for x in record["inputs"])
                options = {"use_qk_l2norm_in_kernel": True, "output_final_state": False}
                if name == "native":
                    result = record["native"](*values, **options)[0]
                else:
                    q, k, v, g, beta = values
                    if name in ("fla_norm_torch", "fp32_norm_torch"):
                        if name == "fla_norm_torch":
                            q, k = (fla_l2norm(x.contiguous()) for x in (q, k))
                        else:
                            q, k = ((x.float() * torch.rsqrt(
                                x.float().square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
                                    for x in (q, k))
                        options["use_qk_l2norm_in_kernel"] = False
                    backend = {"fla_common": "fla", "deltaforge": "deltaforge"}.get(name, "torch")
                    size = 64 if name == "torch64" else 32
                    result = GDNBackend(backend, chunk_size=size)(q, k, v, g, beta, **options)[0]
                gradient = torch.autograd.grad((result * record["upstream"]).sum(), values)
                if reference is None:
                    reference = (result.detach().float(), tuple(x.detach().float() for x in gradient))
                rows.append(emit("layer_diagnostic", layer=index, backend=name,
                                 input_shape=list(values[0].shape), fixed_upstream=True,
                                 output=errors(result, reference[0]),
                                 input_gradients={key: errors(left, right)
                                     for key, left, right in zip(("q", "k", "v", "g", "beta"),
                                                                 gradient, reference[1])}))
                del result, gradient, values
            except Exception as error:
                rows.append(emit("layer_diagnostic", layer=index, backend=name,
                                 error=type(error).__name__, message=str(error)))
            (output / "layer-diagnostics.json").write_text(json.dumps(rows, indent=2))
    emit("layer_diagnostic_complete", cases=len(rows))


def batches(data_path, tokenizer, max_length):
    value = json.loads(Path(data_path).read_text())
    prepared = {}
    for split in ("train", "validation"):
        rows = []
        for sample in value[split]:
            # Use public source text for causal language modeling; no padding loss.
            ids = tokenizer(sample["text"], add_special_tokens=False)["input_ids"][:max_length]
            if len(ids) < 8:
                raise ValueError("Dataset sample has fewer than eight tokens")
            length = len(ids)
            padding = max_length - length
            padded = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * padding
            mask = [1] * length + [0] * padding
            labels = ids + [-100] * padding
            rows.append({key: torch.tensor([values], dtype=torch.long, device="cuda")
                         for key, values in (("input_ids", padded), ("attention_mask", mask),
                                             ("labels", labels))})
        prepared[split] = rows
    return prepared


@torch.no_grad()
def validation_loss(model, data):
    model.eval()
    total = count = 0
    for batch in data:
        tokens = (batch["labels"][:, 1:] != -100).sum().item()
        total += model(**batch).loss.float().item() * tokens
        count += tokens
    model.train()
    return total / count


def parameter_gate(model_path, data_path, output):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    data = batches(data_path, tokenizer, 256)
    model = make_model(model_path, 20261025)
    initial_hash = initialization_hash(model)
    reference = None
    rows = []
    for name in ("torch", "default", "deltaforge", "compiled", "fla"):
        result = None
        try:
            with model_backend(model, name) as adapter:
                model.zero_grad(set_to_none=True)
                result = model(**data["train"][0])
                result.loss.backward()
                gradients = {key: value.grad.detach().cpu().float()
                             for key, value in model.named_parameters()
                             if value.requires_grad and value.grad is not None}
                expected_keys = {key for key, value in model.named_parameters() if value.requires_grad}
                finite = (set(gradients) == expected_keys
                          and all(torch.isfinite(x).all() for x in gradients.values()))
                logits = result.logits.detach().cpu().float()
                loss = result.loss.item()
                checks = {"gradients_complete_and_finite": bool(finite)}
                passed = finite and bool(torch.isfinite(result.loss)) and bool(torch.isfinite(logits).all())
                if reference is None:
                    if not passed:
                        raise AssertionError("Reference gradients/loss/logits failed")
                    reference = (logits, loss, gradients)
                else:
                    checks["logits"] = errors(logits, reference[0])
                    checks["loss_abs_difference"] = abs(loss - reference[1])
                    squared_norm = sum(x.square().sum().item() for x in reference[2].values())
                    per_parameter = []
                    for key, target in reference[2].items():
                        if key not in gradients:
                            continue
                        delta = (gradients[key] - target).square().sum().item()
                        norm = target.square().sum().item()
                        per_parameter.append({"name": key, "squared_error": delta,
                                              "squared_reference_norm": norm,
                                              "relative_l2": (delta / max(norm, 1e-24)) ** 0.5})
                    aggregate = (sum(x["squared_error"] for x in per_parameter)
                                 / max(squared_norm, 1e-24)) ** 0.5
                    checks["parameter_gradient_relative_l2"] = aggregate
                    checks["largest_gradient_error_contributors"] = sorted(
                        per_parameter, key=lambda x: x["squared_error"], reverse=True)[:20]
                    passed = (passed and checks["logits"]["finite"]
                              and checks["logits"]["relative_l2"] <= 0.03
                              and checks["loss_abs_difference"] <= 0.05 and aggregate <= 0.08)
                rows.append(emit("model_gate", backend=name, loss=loss, checks=checks,
                                 gate_passed=bool(passed), reference="torch",
                                 trainable_gradient_tensors=len(gradients),
                                 initial_adapter_sha256=initial_hash, adapter=adapter.stats()))
        except Exception as error:
            rows.append(emit("model_gate", backend=name, gate_passed=False,
                             error=type(error).__name__, message=str(error)))
            if name == "torch":
                raise
        finally:
            del result
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            (output / "model-gates.json").write_text(json.dumps(rows, indent=2))
    emit("model_gate_summary", accepted=[row["backend"] for row in rows if row["gate_passed"]],
         rejected=[row["backend"] for row in rows if not row["gate_passed"]])
    del model, reference, data
    gc.collect()
    torch.cuda.empty_cache()


def train_arm(model_path, data_path, output, backend, repetition, steps=10):
    from transformers import AutoTokenizer
    initialization_seed = 20261025
    model = make_model(model_path, initialization_seed)
    initial_hash = initialization_hash(model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    data = batches(data_path, tokenizer, 256)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    rows = []
    with model_backend(model, backend) as adapter:
        initial_validation = validation_loss(model, data["validation"])
        # Two warmup updates and eight measured updates; budgets and order match.
        for step in range(steps):
            batch = data["train"][step % len(data["train"])]
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = model(**batch)
            loss = result.loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize()
            elapsed = time.perf_counter() - start
            row = {"step": step, "warmup": step < 2, "loss": loss.item(),
                   "gradient_norm": norm.item(), "seconds": elapsed,
                   "effective_tokens": (batch["labels"][:, 1:] != -100).sum().item(),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite training loss")
            rows.append(row)
            emit("train_step", backend=backend, repetition=repetition, **row)
            del result, loss
        final_validation = validation_loss(model, data["validation"])
        final_hash = initialization_hash(model)
        if final_hash == initial_hash:
            raise AssertionError("No adapter parameter update occurred")
        result = emit("training_result", backend=backend, repetition=repetition,
                      model="Qwen/Qwen3.5-2B", scope="LoRA rank8; not full-parameter training",
                      initial_adapter_sha256=initial_hash, final_adapter_sha256=final_hash,
                      trainable_parameters=sum(x.numel() for x in parameters),
                      total_parameters=sum(x.numel() for x in model.parameters()),
                      max_length=256, batch_size=1, optimizer="AdamW",
                      learning_rate=1e-4, checkpoint="nonreentrant", steps=rows,
                      validation_loss_before=initial_validation,
                      validation_loss_after=final_validation, adapter=adapter.stats(),
                      data_sha256=digest(data_path))
        (output / f"training-{backend}-{repetition}.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["kernel", "gate", "train", "diagnostic"], required=True)
    parser.add_argument("--backend", choices=["default", "torch", "deltaforge", "compiled", "fla"])
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "triton", "transformers", "peft", "fla-core")}
    emit("training_environment", phase=args.phase, versions=versions,
         device=torch.cuda.get_device_name(), cuda=torch.version.cuda,
         capability=torch.cuda.get_device_capability(),
         source_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(20261025)
    if args.phase == "kernel":
        kernel_probe(args.output)
    elif args.phase == "gate":
        parameter_gate(args.model, args.data, args.output)
    elif args.phase == "diagnostic":
        layer_diagnostic(args.model, args.data, args.output)
    else:
        train_arm(args.model, args.data, args.output, args.backend, args.repetition)


if __name__ == "__main__":
    main()
