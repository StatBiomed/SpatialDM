from copy import copy
from ._util_dataset import AMetadata


_mel = AMetadata(
    name="melanoma",
    doc_header="melanoma dataset from `Thrane et al <https://doi.org/10.1158/0008-5472.CAN-18-0747>`__.",
    shape=(293, 16148),
    url="https://ndownloader.figshare.com/files/36619200",
)

_A1 = AMetadata(
    name="A1",
    doc_header="Adult colon rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(2649, 33538),
    url="https://ndownloader.figshare.com/files/36674901",
)

_A2 = AMetadata(
    name="A2",
    doc_header="Adult colon rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(2316, 33538),
    url="https://figshare.com/ndownloader/files/36674904",
)

_A3 = AMetadata(
    name="A3",
    doc_header="12-PCW Fetus colon single rep from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(1080, 33538),
    url="https://figshare.com/ndownloader/files/36674331",
)

_A4 = AMetadata(
    name="A4",
    doc_header="19-PCW Fetus colon single rep from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(1242, 33538),
    url="https://figshare.com/ndownloader/files/36674937",
)

_A6 = AMetadata(
    name="A6",
    doc_header="12-PCW TI rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(346, 33538),
    url="https://figshare.com/ndownloader/files/36674310",
)

_A7 = AMetadata(
    name="A7",
    doc_header="12-PCW TI rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(344, 33538),
    url="https://figshare.com/ndownloader/files/36674934",
)

_A8 = AMetadata(
    name="A8",
    doc_header="12-PCW colon rep 1 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(709, 33538),
    url="https://figshare.com/ndownloader/files/36674577",
)

_A9 = AMetadata(
    name="A9",
    doc_header="12-PCW colon rep 2 from Corbett, et al. <https://doi.org/10.1016/j.cell.2020.12.016>`__.",
    shape=(644, 33538),
    url="https://figshare.com/ndownloader/files/36674313",
)

for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "melanoma",
    "A1","A2","A3","A4","A6","A7","A8","A9"]