"""_summary_

_extended_summary_
"""

import os
import pprint
from pathlib import Path

from herbiedss.utils import eccoding

userprofile = os.environ["USERPROFILE"]

# datapath = Path(f"{userprofile}/data/gfs/20261016/subset_ebe45fef__gfs.t12z.pgrb2.0p25.f001")
# datapath = Path(f"{userprofile}/data/gfs/20261016/subset_eb215fef__gfs.t12z.pgrb2.0p25.f002")
datapath = Path(
    f"{userprofile}/data/gfs/20261016/subset_29817758__gfs.t00z.pgrb2.0p25.f126"
)

result = eccoding._read_eccodes_metadata(
    datapath,
    message_index=0,
)

pprint.pp(result)

resolved_result = eccoding._resolve_dss_metadata(
    result,
    model="gfs",
    product="pgrb2.0p25",
    fxx=126,
    dss_units="mm",
)

pprint.pp(resolved_result)
