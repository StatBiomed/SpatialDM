from copy import copy
from _util_dataset import AMetadata


_mel = AMetadata(
    name="melanoma",
    doc_header="log-transformed melanoma dataset from `Thrane et al <https://doi.org/10.1158/0008-5472.CAN-18-0747>`__.",
    shape=(293, 16148),
    url="https://ndownloader.figshare.com/files/36619200",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "melanoma"]