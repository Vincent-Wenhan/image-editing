"""MagicBrush RGBA erase masks use inverse alpha; grayscale uses white as edit."""

from PIL import Image, ImageChops


def editable_mask(mask: Image.Image) -> tuple[Image.Image, str]:
    if "A" in mask.getbands():
        return mask.getchannel("A").point(lambda value: 255 if value < 128 else 0), "inverse_alpha"
    red, green, blue = mask.convert("RGB").split()
    if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
        raise ValueError("Colored mask without alpha has ambiguous edit semantics")
    return mask.convert("L").point(lambda value: 255 if value >= 128 else 0), "white_is_edit"
