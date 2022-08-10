from dataset import *

# _mel = AMetadata(
#     name="melanoma",
#     doc_header="log-transformed melanoma dataset from `Thrane et al <https://doi.org/10.1158/0008-5472.CAN-18-0747>`__.",
#     shape=(293, 16148),
#     url="https://figshare.com/ndownloader/files/36612477",
# )
#
# for name, var in copy(locals()).items():
#     if isinstance(var, AMetadata):
#         var._create_function(name, globals())
#
#
# __all__ = [  # noqa: F822
#     "melanoma"]

mel = melanoma()
print(mel)
print('ok')