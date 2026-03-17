"""
RKNN Runtime wrapper using ctypes for direct librknn.so binding.
Designed for embedded systems (Luckfox Pico, etc.) running 32-bit Linux.
"""

import ctypes
import sys
from pathlib import Path
from typing import Optional, List


class RKNNError(Exception):
    """Base exception for RKNN runtime errors."""
    pass


class RKNNLite:
    """Lightweight RKNN Runtime wrapper using ctypes to call librknn.so."""

    # RKNN return codes
    RKNN_SUCC = 0
    
    # Core masks
    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4
    NPU_CORE_0_1_2 = 7

    def __init__(self, verbose: bool = False, lib_path: Optional[str] = None):
        """
        Initialize RKNN runtime.
        
        Args:
            verbose: Enable verbose logging
            lib_path: Path to librknn_runtime.so (auto-detect if None)
        """
        self.verbose = verbose
        self.context = None
        self._lib = None
        self._load_library(lib_path)

    def _load_library(self, lib_path: Optional[str] = None) -> None:
        """Load librknn_runtime.so using ctypes."""
        if lib_path is None:
            # Try common library locations on embedded Linux
            candidates = [
                "librknn_runtime.so",
                "/usr/lib/librknn_runtime.so",
                "/usr/local/lib/librknn_runtime.so",
                "/opt/rknn/lib/librknn_runtime.so",
                "./lib/librknn_runtime.so",
            ]
            
            for candidate in candidates:
                try:
                    self._lib = ctypes.CDLL(candidate)
                    if self.verbose:
                        print(f"[RKNN] Loaded library from: {candidate}")
                    return
                except OSError:
                    continue
            
            raise RKNNError(
                "Could not find librknn_runtime.so. Tried: " + ", ".join(candidates)
            )
        else:
            try:
                self._lib = ctypes.CDLL(lib_path)
                if self.verbose:
                    print(f"[RKNN] Loaded library from: {lib_path}")
            except OSError as e:
                raise RKNNError(f"Failed to load librknn_runtime.so from {lib_path}: {e}")

        self._setup_function_signatures()

    def _setup_function_signatures(self) -> None:
        """Define ctypes function signatures for RKNN API."""
        # rknn_context is typically a void pointer
        rknn_context = ctypes.c_void_p
        
        # rknn_create(rknn_context *ctx)
        self._rknn_create = self._lib.rknn_create
        self._rknn_create.argtypes = [ctypes.POINTER(rknn_context)]
        self._rknn_create.restype = ctypes.c_int
        
        # rknn_destroy(rknn_context ctx)
        self._rknn_destroy = self._lib.rknn_destroy
        self._rknn_destroy.argtypes = [rknn_context]
        self._rknn_destroy.restype = ctypes.c_int
        
        # rknn_load_rknn(rknn_context ctx, const char *path)
        self._rknn_load_rknn = self._lib.rknn_load_rknn
        self._rknn_load_rknn.argtypes = [rknn_context, ctypes.c_char_p]
        self._rknn_load_rknn.restype = ctypes.c_int
        
        # rknn_init_runtime(rknn_context ctx, rknn_core_mask core_mask)
        self._rknn_init_runtime = self._lib.rknn_init_runtime
        self._rknn_init_runtime.argtypes = [rknn_context, ctypes.c_int]
        self._rknn_init_runtime.restype = ctypes.c_int
        
        # rknn_inputs_set(rknn_context ctx, uint32_t io_num, rknn_input *inputs)
        self._rknn_inputs_set = self._lib.rknn_inputs_set
        self._rknn_inputs_set.argtypes = [rknn_context, ctypes.c_uint32, ctypes.c_void_p]
        self._rknn_inputs_set.restype = ctypes.c_int
        
        # rknn_run(rknn_context ctx)
        self._rknn_run = self._lib.rknn_run
        self._rknn_run.argtypes = [rknn_context]
        self._rknn_run.restype = ctypes.c_int
        
        # rknn_outputs_get(rknn_context ctx, uint32_t io_num, rknn_output *outputs, rknn_perf_detail *perf)
        self._rknn_outputs_get = self._lib.rknn_outputs_get
        self._rknn_outputs_get.argtypes = [
            rknn_context,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p
        ]
        self._rknn_outputs_get.restype = ctypes.c_int
        
        # rknn_outputs_release(rknn_context ctx, uint32_t io_num, rknn_output *outputs)
        self._rknn_outputs_release = self._lib.rknn_outputs_release
        self._rknn_outputs_release.argtypes = [rknn_context, ctypes.c_uint32, ctypes.c_void_p]
        self._rknn_outputs_release.restype = ctypes.c_int

    def _check_error(self, ret: int, msg: str = "") -> None:
        """Check return code and raise exception on error."""
        if ret != self.RKNN_SUCC:
            error_msg = f"RKNN error {ret}"
            if msg:
                error_msg += f": {msg}"
            raise RKNNError(error_msg)

    def create(self) -> None:
        """Create RKNN context."""
        ctx = ctypes.c_void_p()
        ret = self._rknn_create(ctypes.byref(ctx))
        self._check_error(ret, "Failed to create RKNN context")
        self.context = ctx
        if self.verbose:
            print("[RKNN] Context created")

    def load_rknn(self, model_path: str) -> int:
        """
        Load RKNN model from file.
        
        Args:
            model_path: Path to .rknn model file
            
        Returns:
            RKNN return code (0 = success)
        """
        if self.context is None:
            self.create()
        
        model_path_bytes = str(model_path).encode('utf-8')
        ret = self._rknn_load_rknn(self.context, model_path_bytes)
        if ret == self.RKNN_SUCC and self.verbose:
            print(f"[RKNN] Model loaded: {model_path}")
        return ret

    def init_runtime(self, core_mask: int = NPU_CORE_AUTO) -> int:
        """
        Initialize RKNN runtime.
        
        Args:
            core_mask: NPU core mask selection
            
        Returns:
            RKNN return code (0 = success)
        """
        if self.context is None:
            raise RKNNError("Context not initialized. Call load_rknn first.")
        
        ret = self._rknn_init_runtime(self.context, core_mask)
        if ret == self.RKNN_SUCC and self.verbose:
            print(f"[RKNN] Runtime initialized with core mask {core_mask}")
        return ret

    def inference(
        self,
        inputs: List,
        data_type: Optional[str] = None
    ) -> Optional[List]:
        """
        Run inference on input data.
        
        Args:
            inputs: List of input numpy arrays (float32)
            data_type: Data type hint (currently ignored, assumed float32)
            
        Returns:
            List of output numpy arrays, or None on error
        """
        import numpy as np
        
        if self.context is None:
            raise RKNNError("Context not initialized")
        
        if not inputs:
            raise ValueError("No inputs provided")
        
        try:
            # Prepare input structures
            # rknn_input structure: uint32 index, void *buf, uint32 size, uint8 pass_through
            input_size = len(inputs)
            
            # For simplicity, we assume single input
            # Adapt as needed for multiple inputs
            input_data = inputs[0]
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data, dtype=np.float32)
            else:
                input_data = input_data.astype(np.float32)
            
            # Create input buffer
            input_buf = input_data.ctypes.data_as(ctypes.c_void_p)
            input_size_bytes = input_data.nbytes
            
            # Set inputs (simplified - assumes single input)
            # Would need proper rknn_input structure definition for full implementation
            # This is a placeholder - actual implementation would require RKNN header definitions
            
            # Run inference
            ret = self._rknn_run(self.context)
            if ret != self.RKNN_SUCC:
                if self.verbose:
                    print(f"[RKNN] Inference failed with code {ret}")
                return None
            
            # Get outputs (simplified)
            # Would need output structure and query output count/sizes first
            outputs = []
            # Placeholder for output extraction
            
            return outputs if outputs else None
            
        except Exception as e:
            if self.verbose:
                print(f"[RKNN] Inference error: {e}")
            return None

    def release(self) -> None:
        """Release RKNN context."""
        if self.context is not None:
            ret = self._rknn_destroy(self.context)
            if ret == self.RKNN_SUCC and self.verbose:
                print("[RKNN] Context released")
            self.context = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
