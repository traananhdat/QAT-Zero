from enum import StrEnum

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Annotated, Literal


class KDMethodsKind(StrEnum):
    KL = "kl"
    DIST = "dist"


class KLMethod(BaseModel):
    kind: Literal[KDMethodsKind.KL]
    # tau (float): Temperature coefficient. Defaults to 1.0.
    tau: float = Field(4.0)


class DISTMethod(BaseModel):
    kind: Literal[KDMethodsKind.DIST]
    beta: float = Field(1.0)
    gamma: float = Field(1.0)
    tau: float = Field(1.0)


class KDModules(StrEnum):
    NONE = "none"
    CNN = "cnn"
    BN = "bn"
    CNNBN = "cnnbn"
    ALL = "all"


class KDMethods(BaseModel):
    """KDMethods describes the knowledge distillation method you are going to use."""

    # The teacher model.
    teacher_weight: Path = Field(
        Path("yolo11n.pt"), description="the model weight for teacher"
    )

    # This option controls which model KD aligns the student with the teacher.
    kd_module: KDModules = Field(KDModules.CNNBN, description="The kd modules")

    # This option controls the proportion of original loss function in KD.
    original_loss_weight: float = Field(
        0.04, description="loss of the original weight."
    )

    # This option controls the kind of KD loss function
    kd_loss: Annotated[KLMethod | DISTMethod, Field(discriminator="kind")]

    # This option controls the proportion of KD loss function
    kd_loss_weight: float = Field(0.1, description="loss of kd")

    # This option controls the proportion of MSE loss function
    mse_loss_weight: float = Field(1.0, description="loss of MSE")
