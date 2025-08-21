import torch

from quant.quant_mode import QuantizeMode

from ultralytics.nn.tasks import DetectionModel  # type: ignore
from ultralytics.nn.modules import Conv  # type: ignore

from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA  # type: ignore

from quant.module import (
    LSQConv,
    lsqconv_from_conv,
    LSQC3k2,
    lsqc3k2_from_c3k2,
    LSQSPPF,
    lsqsppf_from_sppf,
    LSQC2PSA,
    lsqc2psa_from_c2psa,
)

convert_dict = {
    Conv: lsqconv_from_conv,
    C3k2: lsqc3k2_from_c3k2,
    SPPF: lsqsppf_from_sppf,
    C2PSA: lsqc2psa_from_c2psa,
}

convertible_modules = tuple(convert_dict)

converted_modules = (LSQConv, LSQC3k2, LSQSPPF, LSQC2PSA)


# By default it will copy all weights from module to new module.
def quantize_module(module, quantize_mode: QuantizeMode):
    """Convert module to quantized module.

    Parameters
    ----------
    module : nn.Module
    quantize_mode : QuantizeMode
        Please see the documentation of QuantizeMode.

    Returns
    -------
    nn.Module
    """
    if not isinstance(module, convertible_modules):
        return module
    converted_block = convert_dict[type(module)](
        module,
        mode=quantize_mode,
    )
    converted_block.f = module.f
    converted_block.i = module.i
    return converted_block


def inplace_quantize(
    model: DetectionModel, quantize_mode: QuantizeMode
) -> DetectionModel:
    """Inplace quantize a model. It changes the in-argument `model`.

    Parameters
    ----------
    model : DetectionModel
    quantize_mode : QuantizeMode

    Returns
    -------
    DetectionModel
    """
    # We don't want the autograd to track these weird operations.
    with torch.no_grad():
        if quantize_mode == "exact":
            return model
        for i, module in enumerate(model.model):
            if i == 0:
                continue
            model_to_delete = model.model[i]
            model.model[i] = quantize_module(module, quantize_mode)
            del model_to_delete
        return model
