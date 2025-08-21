from pydantic import BaseModel, Field

from typing import Annotated, Literal, Union


class ExactModel(BaseModel):
    kind: Literal["exact"]


class SymQuantizeOption(BaseModel):
    kind: Literal["quantize_sym"]
    weight_bits: int
    activation_bits: int


class AsymQuantizeOption(BaseModel):
    kind: Literal["quantize_asym"]
    weight_bits: int
    activation_bits: int


QuantizeMode = Annotated[
    Union[ExactModel, SymQuantizeOption, AsymQuantizeOption],
    Field(discriminator="kind"),
]
