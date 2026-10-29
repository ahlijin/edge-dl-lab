"""TensorRT 引擎封装（cuda-python 版），设备端多个脚本共用一份。

用法:
    from trt_engine import TRTEngine
    engine = TRTEngine("wakeword.engine")
    logits = engine.infer(feat[None, None])     # (1,1,64,101) float32 -> (1,2)
"""

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


class TRTEngine:
    def __init__(self, engine_path, verbose=False):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"无法反序列化引擎 {engine_path}（版本不匹配或文件损坏）")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            size = trt.volume(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            host_mem = np.empty(size, dtype=dtype)
            err, device_mem = cudart.cudaMalloc(host_mem.nbytes)
            assert err == cudart.cudaError_t.cudaSuccess

            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "name": name})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "name": name})

        err, self.stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

    def infer(self, input_data):
        np.copyto(self.inputs[0]["host"], np.asarray(input_data, dtype=np.float32).ravel())

        err, = cudart.cudaMemcpyAsync(
            self.inputs[0]["device"], self.inputs[0]["host"].ctypes.data,
            self.inputs[0]["host"].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], inp["device"])
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], out["device"])

        self.context.execute_async_v3(stream_handle=self.stream)

        err, = cudart.cudaMemcpyAsync(
            self.outputs[0]["host"].ctypes.data, self.outputs[0]["device"],
            self.outputs[0]["host"].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        err, = cudart.cudaStreamSynchronize(self.stream)
        assert err == cudart.cudaError_t.cudaSuccess

        return self.outputs[0]["host"].copy()

    def prob_positive(self, feat2d):
        """(64,101) 特征 -> 唤醒词概率"""
        logits = self.infer(np.asarray(feat2d, dtype=np.float32)[None, None]).ravel()
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())
