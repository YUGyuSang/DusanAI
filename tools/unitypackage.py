#!/usr/bin/env python3
"""
Helpers to read Unity .unitypackage archives (gzip tar of <guid>/{asset, pathname}).

    from unitypackage import UnityPackage
    pkg = UnityPackage("Doosanenerbility/.../robot_spot.unitypackage")
    fbx = pkg.find(".fbx")            # -> local file path of the extracted asset
    tex = pkg.find("Base_Color.png")

Extraction goes to a cache dir (default: <repo>/tools/.unity_cache/<package name>).
"""
import os
import tarfile

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ASSET_ROOT = os.path.join(REPO, "Doosanenerbility", "Doosanenerbility", "Modeling")
DEFAULT_CACHE = os.path.join(REPO, "tools", ".unity_cache")

PACKAGES = {
    "robot_spot": os.path.join(ASSET_ROOT, "object", "robot_spot.unitypackage"),
    "fac_gastank": os.path.join(ASSET_ROOT, "object", "fac_gastank.unitypackage"),
    "obs_palette": os.path.join(ASSET_ROOT, "object", "obs_palette.unitypackage"),
    "factory_hall": os.path.join(ASSET_ROOT, "map", "factory_hall.unitypackage"),
    "factory_inner_1": os.path.join(ASSET_ROOT, "map", "factory_inner_1.unitypackage"),
    "gas_spurt": os.path.join(ASSET_ROOT, "effect", "gas_spurt.unitypackage"),
    "spot_move": os.path.join(ASSET_ROOT, "animation", "spot_move.unitypackage"),
    "spot_fall": os.path.join(ASSET_ROOT, "animation", "spot_fall.unitypackage"),
}


class UnityPackage:
    def __init__(self, path_or_name, cache_dir=DEFAULT_CACHE):
        path = PACKAGES.get(path_or_name, path_or_name)
        self.path = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.dir = os.path.join(cache_dir, self.name)
        if not os.path.isdir(self.dir) or not os.listdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            with tarfile.open(path, "r:gz") as tf:
                members = [m for m in tf.getmembers()
                           if m.isfile() and not os.path.isabs(m.name) and ".." not in m.name.split("/")]
                tf.extractall(self.dir, members=members)
        self.assets = {}   # unity pathname -> local asset file
        for guid in os.listdir(self.dir):
            g = os.path.join(self.dir, guid)
            pn = os.path.join(g, "pathname")
            asset = os.path.join(g, "asset")
            if os.path.isfile(pn) and os.path.isfile(asset):
                with open(pn, "r", encoding="utf-8", errors="replace") as f:
                    self.assets[f.readline().strip()] = asset

    def find(self, suffix):
        """First asset whose unity path ends with `suffix` (case-insensitive)."""
        s = suffix.lower()
        for p, a in sorted(self.assets.items()):
            if p.lower().endswith(s):
                return a
        return None

    def find_all(self, suffix):
        s = suffix.lower()
        return [(p, a) for p, a in sorted(self.assets.items()) if p.lower().endswith(s)]


if __name__ == "__main__":
    for name in PACKAGES:
        if os.path.exists(PACKAGES[name]):
            pkg = UnityPackage(name)
            print(f"== {name}")
            for p in sorted(pkg.assets):
                print("   ", p)
