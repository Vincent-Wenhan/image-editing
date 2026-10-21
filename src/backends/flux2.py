"""Resident BF16 FLUX.2 Klein Base pipeline with unchanged Stage 0 sampling."""

import time
from pathlib import Path

from src.backends.flux2_conditions import reference_images
from src.backends.flux2_encoding import joint_preprocessing
from src.conditioning.controls import ControlBundle
from src.core.types import EditInput, EditOutput, GenerationOptions
from src.infra.gpu import require_lease


class Flux2Backend:
    def __init__(self, model: str | Path, lora: str | None = None) -> None:
        require_lease()
        import torch
        from diffusers import Flux2KleinPipeline

        torch.cuda.reset_peak_memory_stats()
        self.pipe = Flux2KleinPipeline.from_pretrained(
            Path(model), torch_dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        if lora:
            self.pipe.load_lora_weights(lora)

    def edit(
        self, edit_input: EditInput, controls: ControlBundle, options: GenerationOptions
    ) -> EditOutput:
        images = reference_images(edit_input, controls, options)
        import torch

        generator = torch.Generator("cuda").manual_seed(options.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with (
            torch.inference_mode(),
            joint_preprocessing(self.pipe, enabled=isinstance(images, list)),
        ):
            image = self.pipe(
                image=images,
                prompt=edit_input.instruction,
                width=options.width,
                height=options.height,
                num_inference_steps=options.steps,
                guidance_scale=options.guidance,
                generator=generator,
            ).images[0]
        torch.cuda.synchronize()
        return EditOutput(
            image=image,
            seconds=time.perf_counter() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
