from pathlib import Path
from typing import Optional

import numpy as np
from h5py import File


class HDFCache:
    def __init__(self, filename: Path):
        self.f = File(filename, "a")

    def get(self, key: str, group: str = None) -> Optional[np.ndarray]:
        try:
            if group:
                g = self.f[group]
            else:
                g = self.f
            return np.asarray(g[key])
        except KeyError:
            return None

    def set(self, key: str, data: np.ndarray, group: str = None, compressed: bool = False) -> None:
        if self.get(key, group) is not None:
            self.delete(key, group)
        if not group:
            g = self.f
        elif group not in self.f:
            g = self.f.create_group(group)
        else:
            g = self.f[group]
        if compressed:
            kwargs = {
                "compression": "gzip",
                "compression_opts": 5
            }
        else:
            kwargs = {}
        g.create_dataset(key, data=data, **kwargs)

    def delete(self, key: str, group: str = None):
        if not group:
            g = self.f
        else:
            g = self.f[group]
        del g[key]

    def delgroup(self, group: str):
        raise NotImplemented()

    def __del__(self):
        self.f.close()
