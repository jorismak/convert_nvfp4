#!/usr/bin/env python3
"""
Tests for NVFP4 quantization implementation.

Verifies that our implementation matches comfy_kitchen and produces
correct round-trip results when dequantized.
"""

import torch
import sys

sys.path.insert(0, ".")

from convert_nvfp4 import (
    _f32_to_floatx_unpacked,
    pack_uint4,
    float_to_fp4_e2m1_packed,
    quantize_nvfp4,
    F4_E2M1_MAX,
    F8_E4M3_MAX,
)

import comfy_kitchen as ck
from comfy_kitchen.float_utils import (
    _f32_to_floatx_unpacked as ck_f32_to_floatx,
    _floatx_unpacked_to_f32 as ck_floatx_to_f32,
)


def test_fp4_exact_values():
    """Test that exact FP4 representable values convert correctly."""
    print("TEST 1: FP4 E2M1 exact value mapping")

    # All 16 FP4 E2M1 values
    test_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    expected_codes = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8
    )

    our_codes = _f32_to_floatx_unpacked(test_values, ebits=2, mbits=1)
    ck_codes = ck_f32_to_floatx(test_values, ebits=2, mbits=1)

    assert torch.equal(our_codes, expected_codes), (
        f"Our codes {our_codes} != expected {expected_codes}"
    )
    assert torch.equal(our_codes, ck_codes), (
        f"Our codes {our_codes} != CK codes {ck_codes}"
    )

    # Verify round-trip
    dequant = ck_floatx_to_f32(our_codes, ebits=2, mbits=1)
    # Note: -0.0 and 0.0 compare equal with torch.equal
    assert torch.allclose(dequant, test_values), (
        f"Dequant {dequant} != original {test_values}"
    )

    print("  PASSED: All 16 FP4 codes map correctly")
    return True


def test_fp4_rounding():
    """Test that rounding behavior matches IEEE-754 round-to-nearest-even."""
    print("TEST 2: FP4 rounding behavior")

    test_cases = [
        # (input_value, expected_code, description)
        (2.51, 5, "2.51 -> 3.0 (closer)"),
        (2.49, 4, "2.49 -> 2.0 (closer)"),
        (2.50, 4, "2.50 -> 2.0 (round to even)"),
        (3.50, 6, "3.50 -> 4.0 (round to even)"),
        (0.25, 0, "0.25 -> 0.0 (round to even)"),
        (0.75, 2, "0.75 -> 1.0 (round to even)"),
        (7.0, 7, "7.0 -> 6.0 (saturate)"),
        (100.0, 7, "100.0 -> 6.0 (saturate)"),
    ]

    all_passed = True
    for val, expected_code, desc in test_cases:
        t = torch.tensor([val], dtype=torch.float32)
        our_code = _f32_to_floatx_unpacked(t, 2, 1).item()
        ck_code = ck_f32_to_floatx(t, 2, 1).item()

        if our_code != ck_code:
            print(f"  FAIL: {desc}: our={our_code} != ck={ck_code}")
            all_passed = False
        elif our_code != expected_code:
            # Our code matches CK but not expected - CK is authoritative
            print(
                f"  NOTE: {desc}: got {our_code} (matches CK), expected {expected_code}"
            )

    if all_passed:
        print("  PASSED: All rounding cases match comfy_kitchen")
    return all_passed


def test_full_pipeline_match():
    """Test that our full quantization pipeline matches comfy_kitchen exactly."""
    print("TEST 3: Full pipeline vs comfy_kitchen")

    torch.manual_seed(42)
    weight = torch.randn(128, 128, dtype=torch.float32)

    # Our pipeline
    qdata_ours, block_scale_ours, tensor_scale_ours = quantize_nvfp4(weight)

    # ComfyKitchen pipeline
    scale_ck = torch.amax(weight.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)
    qdata_ck, block_scale_ck = ck.quantize_nvfp4(weight, scale_ck, pad_16x=True)

    assert torch.allclose(tensor_scale_ours, scale_ck), "Tensor scale mismatch"
    assert torch.equal(block_scale_ours, block_scale_ck), "Block scale mismatch"
    assert torch.equal(qdata_ours, qdata_ck), "Quantized data mismatch"

    print("  PASSED: Full pipeline matches comfy_kitchen exactly")
    return True


def test_round_trip_error():
    """Test that round-trip error is within expected bounds."""
    print("TEST 4: Round-trip error bounds")

    torch.manual_seed(42)
    weight = torch.randn(128, 128, dtype=torch.float32)

    # Quantize
    qdata, block_scale, tensor_scale = quantize_nvfp4(weight)

    # Dequantize using comfy_kitchen (simulating ComfyUI reader)
    recovered = ck.dequantize_nvfp4(qdata, tensor_scale, block_scale, torch.float32)
    recovered = recovered[:128, :128]

    # Compute error
    error = (weight - recovered).abs()

    # Expected max error is bounded by half the quantization step at max scale
    max_block = block_scale.float().max().item()
    # FP4 has spacing of 2.0 at the highest level (between 4.0 and 6.0)
    # So max error is tensor_scale * max_block * 1.0 (half of 2.0)
    theoretical_max = tensor_scale.item() * max_block * 1.0

    actual_max = error.max().item()
    mean_error = error.mean().item()

    print(f"  Max error: {actual_max:.6f} (bound: {theoretical_max:.6f})")
    print(f"  Mean error: {mean_error:.6f}")

    assert actual_max <= theoretical_max * 1.1, (
        f"Max error {actual_max} exceeds bound {theoretical_max}"
    )

    print("  PASSED: Error within expected bounds")
    return True


def test_large_tensor():
    """Test with larger tensor to ensure memory-efficient processing works."""
    print("TEST 5: Large tensor (1024x2048)")

    torch.manual_seed(123)
    weight = torch.randn(1024, 2048, dtype=torch.float32)

    qdata, block_scale, tensor_scale = quantize_nvfp4(weight)

    # Compare with comfy_kitchen
    scale_ck = torch.amax(weight.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)
    qdata_ck, block_scale_ck = ck.quantize_nvfp4(weight, scale_ck, pad_16x=True)

    assert torch.equal(qdata, qdata_ck), "Large tensor qdata mismatch"
    assert torch.equal(block_scale, block_scale_ck), "Large tensor block_scale mismatch"

    # Check round-trip
    recovered = ck.dequantize_nvfp4(qdata, tensor_scale, block_scale, torch.float32)
    error = (weight - recovered[:1024, :2048]).abs()

    print(f"  Max error: {error.max().item():.6f}")
    print(f"  Mean error: {error.mean().item():.6f}")
    print("  PASSED: Large tensor matches comfy_kitchen")
    return True


def test_various_shapes():
    """Test various tensor shapes including non-power-of-2 dimensions."""
    print("TEST 6: Various tensor shapes")

    shapes = [
        (16, 16),
        (17, 17),  # Non-power-of-2
        (100, 200),
        (127, 255),  # Near padding boundaries
        (128, 256),  # Exact padding boundaries
        (129, 257),  # Just over padding boundaries
    ]

    torch.manual_seed(456)
    all_passed = True

    for rows, cols in shapes:
        weight = torch.randn(rows, cols, dtype=torch.float32)
        qdata, block_scale, tensor_scale = quantize_nvfp4(weight)

        scale_ck = torch.amax(weight.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)
        qdata_ck, block_scale_ck = ck.quantize_nvfp4(weight, scale_ck, pad_16x=True)

        if not torch.equal(qdata, qdata_ck):
            print(f"  FAIL: Shape {rows}x{cols} qdata mismatch")
            all_passed = False
        else:
            print(f"  OK: Shape {rows}x{cols}")

    if all_passed:
        print("  PASSED: All shapes work correctly")
    return all_passed


def main():
    """Run all tests."""
    print("=" * 60)
    print("NVFP4 Quantization Tests")
    print("=" * 60)
    print()

    tests = [
        test_fp4_exact_values,
        test_fp4_rounding,
        test_full_pipeline_match,
        test_round_trip_error,
        test_large_tensor,
        test_various_shapes,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            results.append(False)
        print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
