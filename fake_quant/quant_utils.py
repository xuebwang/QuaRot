import math
import transformers
import torch
import utils
import hadamard_utils
# import fast_hadamard_transform



MXFP_FORMAT_CACHE = {
    # data type: ebits, mbits, emax, max_norm, min_norm
    "mx_int8": (0, 8, 0, 1.984375, 0),
    "mx_int4": (0, 4, 0, 1.75, 0),
    "mx_int2": (0, 2, 0, 1.0, 0),
    "mx_fp8e5m2": (5, 4, 15, 57344.0, 6.103515625e-05),
    "mx_fp8": (4, 5, 8, 448.0, 0.015625),
    "mx_fp8e4m3": (4, 5, 8, 448.0, 0.015625),
    "mx_fp6e3m2": (3, 4, 4, 28.0, 0.25),
    "mx_fp6": (2, 5, 2, 7.5, 1.0),
    "mx_fp6e2m3": (2, 5, 2, 7.5, 1.0),
    "mx_fp4": (2, 3, 2, 6.0, 1.0),
    "mx_fp4e2m1": (2, 3, 2, 6.0, 1.0),
    "mx_float16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_fp16": (5, 12, 15, 65504.0, 6.103515625e-05),
    "mx_bfloat16": (8, 9, 127, 3.3895313892515355e+38, 1.1754943508222875e-38),
    "mx_bf16": (8, 9, 127, 3.3895313892515355e+38, 1.1754943508222875e-38),
}

def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """Straight-Through Estimator for floor.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.floor() - x).detach() + x

def reshape_pad_tensor_by_group_size(data: torch.Tensor, group_size: int):
    """Reshapes and pads the tensor to ensure that it can be quantized in groups of `group_size`.

    This function adjusts the input tensor's shape so that its last dimension is a multiple
    of the specified `group_size`. If padding is required, it adds padding to the tensor
    to achieve this. If the tensor's last dimension is already divisible by `group_size`,
    no padding is applied.

    Args:
        data (torch.Tensor): The input tensor to be reshaped and padded.
        group_size (int): The size of the groups that the tensor should be reshaped into.

    Returns:
        torch.Tensor: The reshaped and padded tensor, if necessary.
        tuple: The original shape of the input tensor.
        int: The padding length applied to the tensor. Returns 0 if no padding is applied.
    """
    orig_shape = data.shape
    pad_len = 0
    if len(data.shape) > 2:
        data = data.reshape(-1, orig_shape[-1])
    if group_size == -1 or data.shape[1] < group_size:
        return data, orig_shape, pad_len
    elif data.shape[1] % group_size == 0:
        data = data.reshape(-1, group_size)
        return data, orig_shape, pad_len
    else:
        pad_len = (data.shape[1] + group_size - 1) // group_size * group_size - data.shape[1]
        data_new = torch.nn.functional.pad(data, (0, pad_len))
        data_new = data_new.reshape(-1, group_size)
        return data_new, orig_shape, pad_len

def revert_tensor_by_pad(data: torch.Tensor, orig_shape: tuple, pad_len: int):
    """Reverts the tensor to its original shape by removing padding.

    This function removes the padding added during reshaping and returns the tensor to
    its original shape.

    Args:
        data (torch.Tensor): The reshaped and possibly padded tensor.
        orig_shape (tuple): The original shape of the tensor before reshaping.
        pad_len (int): The length of the padding to be removed.

    Returns:
        torch.Tensor: The tensor restored to its original shape.
    """
    if pad_len == 0:
        return data.reshape(orig_shape)
    else:
        if len(orig_shape) > 2:
            tmp_shape = torch.prod(torch.tensor(orig_shape[:-1])).item()
        else:
            tmp_shape = orig_shape[0]
        data_new = data.reshape(tmp_shape, -1)
        data_new = data_new[:, :-pad_len]
        data_new = data_new.reshape(orig_shape)
        return data_new

FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

def quant_mx(tensor, bits=4, group_size=32, v=0, max_scale=1.0,
         mantissa_rounding="even", data_type="mx_fp4e2m1", **kwargs):
    """Quantize the given tensor using the specified parameters.

    This function performs quantization on the `tensor` tensor according to the
    given bit width (`bits`), data type (`data_type`), and additional parameters.
    The quantization process involves scaling the tensor values and adjusting
    the exponent and mantissa to fit within the specified format.

    Args:
        tensor (torch.Tensor): The tensor containing the tensors to be quantized.
        bits (int): The bit width to be used for quantization.
        group_size (int): The group size of sharing scale and exponent.
        data_type (str): The data type for quantization (e.g., 'mx_fp4').
        v (float): A value used for adjusting the tensors.
        max_scale (float or torch.Tensor): The maximum scale to be applied to the tensors.
        mantissa_rounding (str): rounding method for mantissa,currently support even,nearest,floor

    Returns:
        tuple: A tuple containing the quantized tensors, shared exponent, and None (reserved for future use).

    Raises:
        KeyError: If `data_type` is not found in `MXFP_FORMAT_CACHE`.
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    orig_dtype = tensor.dtype
    shared_exp, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)
    if isinstance(max_scale, torch.Tensor):
        shared_exp *= (max_scale.unsqueeze(dim=-1))
    else:
        shared_exp *= max_scale
    scale_emax = 2 ** (8 - 1) - 1
    shared_exp = torch.log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype))
    shared_exp[shared_exp == torch.inf] = scale_emax + emax
    shared_exp[shared_exp == -torch.inf] = -scale_emax + emax
    shared_exp = (shared_exp - emax)
    shared_exp = floor_ste(shared_exp)
    shared_exp[shared_exp > scale_emax] = scale_emax  ##changed Nan
    shared_exp[shared_exp < -scale_emax] = -scale_emax
    if (shared_exp.dtype == torch.float16 and (torch.any(shared_exp > 15) or torch.any(shared_exp < -24))) or (
            shared_exp.dtype == torch.bfloat16 and torch.any((shared_exp < -126))):
        tensor = tensor.to(torch.float32)
        shared_exp = shared_exp.to(torch.float32)
    tensor = tensor / (2 ** shared_exp)
    tensor = tensor + v
    if ebits != 0:
        private_exp = floor_ste(torch.log2(torch.abs(tensor) + (tensor == 0).type(tensor.dtype)))

        # The minimum representable exponent for 8 exp bits is -126
        min_exp = -(2 ** (ebits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
    else:
        private_exp = None

    # Scale up so appropriate number of mbits are in the integer portion of the number
    tensor = tensor * (2 ** (mbits - 2)) if private_exp is None else tensor / (2 ** private_exp) * (2 ** (mbits - 2))

    if mantissa_rounding == "even":
        abs_tensor = torch.abs(tensor)
        mask_tensor = ((abs_tensor - 0.5) % 2 == torch.zeros_like(abs_tensor)).type(tensor.dtype)
        tensor = torch.sign(tensor) * (floor_ste(abs_tensor + 0.5) - mask_tensor)
    elif mantissa_rounding == "nearest":
        tensor = round_ste(tensor)
    elif mantissa_rounding == "floor":
        tensor = floor_ste(tensor)
    else:
        raise ValueError("mantissa_rounding only supports even, nearest or floor.")
    max_mantissa = 2 ** (mbits - 1) - 1
    tensor = torch.clamp(tensor, -max_mantissa, max_mantissa)

    # Undo scaling
    tensor = tensor / (2 ** (mbits - 2)) if private_exp is None else tensor / (2 ** (mbits - 2)) * (2 ** private_exp)

    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    tensor = tensor * (2 ** shared_exp)
    tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
    return tensor.to(orig_dtype)


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=4, zero_point=True, q_group_size=32, inplace=True, get_scale_zp=False
):
    # org_w_shape = w.shape
    # if q_group_size > 0:
    #     assert org_w_shape[-1] % q_group_size == 0
    #     w = w.reshape(-1, q_group_size)
    # assert w.dim() == 2

    w, org_w_shape, pad_len = reshape_pad_tensor_by_group_size(w, q_group_size)

    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

# from torchao.prototype.mx_formats.mx_tensor import MXTensor, ScaleCalculationMode
# # Note: MX int8 is not implemented yet
# from torchao.prototype.mx_formats.constants import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2, DTYPE_FP4

from  quark.torch.kernel.hw_emulation.hw_emulation_interface import fake_quantize_mx
from quark.torch.quantization.config.type import QSchemeType, Dtype

def quant_mx_quark(x):
    return fake_quantize_mx(input_tensor=x,
                            mx_element_dtype=Dtype.fp4,
                            axis=-1,
                            block_size=32)

def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0>=8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1>=8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)

class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        else:
            x = quant_mx_quark(x).to(x_dtype)
            return x

        #     return x
        # elif self.sym:
        #     return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        # return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            utils.cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

# class ActQuantWrapper(torch.nn.Module):
#     '''
#         This class is a wrapper for the activation quantization.
#         We extract the FP features in the forward pass and quantize the rest using
#         the self.quantizer object.
#         If a rotation Q is provided, the weight matrix will be rotated,
#         a pre-forward hook will be registerd to rotate the activation before quantization.
#     '''

#     def __init__(self, module:torch.nn.Linear):
#         super(ActQuantWrapper, self).__init__()
#         assert isinstance(module, torch.nn.Linear)
#         self.module = module
#         self.weight = module.weight
#         self.bias = module.bias
#         self.quantizer = ActQuantizer()
#         self.out_quantizer = ActQuantizer()
#         self.register_buffer('had_K', torch.tensor(0))
#         self._buffers['had_K'] = None
#         self.K = 1
#         self.online_full_had = False
#         self.online_partial_had = False
#         self.had_dim = 0
#         self.fp32_had = False

#     def extra_repr(self) -> str:
#         str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
#         if self.quantizer.bits < 16:
#             str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

#         str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
#         if self.out_quantizer.bits < 16:
#             str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

#         return str_

#     def forward(self, x):
#         x_dtype = x.dtype

#         # Rotate, if needed
#         if self.online_full_had:
            
#             if self.fp32_had: # Full Hadamard in FP32
#                 x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
#             else: # Full Hadamard in FP16
#                 x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
            
#         elif self.online_partial_had:
#             # todo: implement this in QAttention to avoid reshaping!
            
#             if self.fp32_had:
#                 x = x.float()
                
#             init_shape = x.shape
#             if self.K == 1:
#                 x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
#                                                                scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
#             else:
#                 x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
                
#             if self.fp32_had:
#                 x = x.to(x_dtype)
#             x = x.reshape(init_shape)

#         if self.quantizer.bits < 16: #Quantize, if needed
#             self.quantizer.find_params(x)
#             x = self.quantizer(x).to(x_dtype)
#             self.quantizer.free()

#         x = self.module(x).to(x_dtype)

#         if self.out_quantizer.bits < 16: #Quantize the output, if needed
#             self.out_quantizer.find_params(x)
#             x = self.out_quantizer(x).to(x_dtype)
#             self.out_quantizer.free()

#         return x

class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear=None):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.name = None
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.runtime_smooth = True
        self.act_scale_g128 = True

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype
        # Rotate, if needed
        if self.online_full_had:
            
            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
            
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            
            if self.fp32_had:
                x = x.float()
                
            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
                
            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16: #Quantize, if needed
            # if self.runtime_smooth:
            #     if len(x.shape) == 2:
            #         act_scales = x.abs().max(dim=0,keepdim=True)[0]
            #     else:
            #         act_scales = x.abs().max(dim=1,keepdim=True)[0]
            #     act_scales.clamp_(min=1e-5)

            #     # breakpoint()

            #     if self.act_scale_g128:
            #         index = torch.argsort(act_scales, dim=-1, descending=True)
            #         act_scales = torch.gather(act_scales, -1, index)
            #         sg = 128
            #         if len(x.shape) == 2:
            #             act_scales = act_scales.reshape(1,x.shape[1]//sg,sg)
            #             act_scales = act_scales.max(dim=-1,keepdim=True)[0].repeat(1,1,sg)
            #             act_scales = act_scales.reshape(1,-1)
            #         else:
            #             act_scales = act_scales.reshape(x.shape[0],1,x.shape[2]//sg,sg)
            #             act_scales = act_scales.max(dim=-1,keepdim=True)[0].repeat(1,1,1,sg)
            #             act_scales = act_scales.reshape(x.shape[0],1,-1)
            #         reverse_index = torch.argsort(index, dim=-1)
            #         act_scales = torch.gather(act_scales, -1, reverse_index)
            #     x = x / act_scales
            if self.runtime_smooth:
                if len(x.shape) == 2:
                    act_scales = x.abs().mean(dim=0, keepdim=True)  # 将 max 改为 mean
                else:
                    act_scales = x.abs().mean(dim=1, keepdim=True)  # 将 max 改为 mean
                act_scales.clamp_(min=1e-5)

                # breakpoint()

                if self.act_scale_g128:
                    index = torch.argsort(act_scales, dim=-1, descending=True)
                    act_scales = torch.gather(act_scales, -1, index)
                    sg = 128
                    if len(x.shape) == 2:
                        act_scales = act_scales.reshape(1, x.shape[1] // sg, sg)
                        act_scales = act_scales.mean(dim=-1, keepdim=True).repeat(1, 1, sg)  # 将 max 改为 mean
                        act_scales = act_scales.reshape(1, -1)
                    else:
                        act_scales = act_scales.reshape(x.shape[0], 1, x.shape[2] // sg, sg)
                        act_scales = act_scales.mean(dim=-1, keepdim=True).repeat(1, 1, 1, sg)  # 将 max 改为 mean
                        act_scales = act_scales.reshape(x.shape[0], 1, -1)
                    reverse_index = torch.argsort(index, dim=-1)
                    act_scales = torch.gather(act_scales, -1, reverse_index)
                x = x / act_scales

            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            if self.runtime_smooth:
                x = x * act_scales
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            if self.runtime_smooth:
                if len(x.shape) == 2:
                    act_scales = x.abs().max(dim=0,keepdim=True)[0]
                else:
                    act_scales = x.abs().max(dim=1,keepdim=True)[0]
                act_scales.clamp_(min=1e-5)
                if self.act_scale_g128:
                    index = torch.argsort(act_scales, dim=-1, descending=True)
                    act_scales = torch.gather(act_scales, -1, index)
                    sg = 128
                    if len(x.shape) == 2:
                        act_scales = act_scales.reshape(1,x.shape[1]//sg,sg)
                        act_scales = act_scales.max(dim=-1,keepdim=True)[0].repeat(1,1,sg)
                        act_scales = act_scales.reshape(1,-1)
                    else:
                        act_scales = act_scales.reshape(x.shape[0],1,x.shape[2]//sg,sg)
                        act_scales = act_scales.max(dim=-1,keepdim=True)[0].repeat(1,1,1,sg)
                        act_scales = act_scales.reshape(x.shape[0],1,-1)
                    reverse_index = torch.argsort(index, dim=-1)
                    act_scales = torch.gather(act_scales, -1, reverse_index)
                x = x / act_scales

            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            if self.runtime_smooth:
                x = x * act_scales

            self.out_quantizer.free()

        return x



class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.bits < 16:
            # if self.sym:
            #     return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            # return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
            x = quant_mx_quark(x).to(x_dtype)
            return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)



def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
