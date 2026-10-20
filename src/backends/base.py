"""Only the external editing call is shared; model internals remain independent."""

from typing import Protocol, runtime_checkable

from src.conditioning.controls import ControlBundle
from src.core.types import EditInput, EditOutput, GenerationOptions


@runtime_checkable
class EditingBackend(Protocol):
    def edit(
        self, edit_input: EditInput, controls: ControlBundle, options: GenerationOptions
    ) -> EditOutput: ...
