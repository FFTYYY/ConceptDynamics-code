from .identity import make_data as make_data_identity
from .identity_truncate import make_data as make_data_identity_truncate
from .multi_identity import make_data as make_data_multi_identity
from xingyun import MyDict

tasks = {
    "identity"           : make_data_identity   , 
    "identity_truncate"  : make_data_identity_truncate   , 
    "multi_identity"      : make_data_multi_identity , 
}

def get_data(data_name: str, C: MyDict):
    return tasks[data_name](C)

