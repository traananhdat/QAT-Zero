from enum import Enum
from pathlib import Path


class ViTType(Enum):
    DeiT_T = 0
    DeiT_S = 1
    SWIN_T = 2
    SWIN_S = 3


model_weights = {
    ViTType.SWIN_T: "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth",
    ViTType.SWIN_S: "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
    ViTType.DeiT_T: "https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask_rcnn_deit_tiny_fpn_3x_coco.pth.tar",
    ViTType.DeiT_S: "https://github.com/czczup/ViT-Adapter/releases/download/v0.3.1/mask_rcnn_deit_small_fpn_3x_coco.pth.tar",
}

model_weights_name = {
    ViTType.SWIN_T: "mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth",
    ViTType.SWIN_S: "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
    ViTType.DeiT_T: "mask_rcnn_deit_tiny_fpn_3x_coco.pth.tar",
    ViTType.DeiT_S: "mask_rcnn_deit_small_fpn_3x_coco.pth.tar",
}


def _download_vit(
    model_type: ViTType, download_dir: Path = Path("./pretrained")
) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)

    import requests  # type: ignore

    resp = requests.get(model_weights[model_type])
    download_target = download_dir / model_weights_name[model_type]
    with download_target.open("wb") as f:
        f.write(resp.content)
    return download_target


def get_vit(model_type: ViTType, work_dir: Path = Path("./pretrained")) -> Path:
    vit_target = work_dir / model_weights_name[model_type]
    if vit_target.exists():
        return vit_target
    return _download_vit(model_type=model_type, download_dir=work_dir)


def download_all(download_dir: Path = Path("./pretrained")):
    """download all pretrained models from the web

    Parameters
    ----------
    download_dir : Path, optional. By default Path("./pretrained")
    """
    for i in ViTType:
        _download_vit(i, download_dir=download_dir)


# if this file is treated as a script
if __name__ == "__main__":
    download_all()
    if not Path("./data/coco").exists():
        print("please prepare coco datasets in vit/data/coco")
