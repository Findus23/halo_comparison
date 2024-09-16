from pathlib import Path

import pandas as pd

h = 0.6777


def make_names_unique(names: list[str]) -> list[str]:
    """
    https://stackoverflow.com/a/30650847
    """
    newlist = []
    for i, v in enumerate(names):
        count = names[:i].count(v)
        newlist.append(v + f"_{count + 1}" if count > 0 else v)
    return newlist


def read_rockstar_halos(dir: Path) -> tuple[pd.DataFrame, Path]:
    file = dir / "halos_127.0.ascii"
    if not file.exists():
        file = dir / "halos_0.0.ascii"
    with file.open() as f:
        first_line = next(f)
    headers = first_line.lstrip("#").strip().split()
    headers = make_names_unique(headers)
    df = pd.read_csv(file, sep=" ", names=headers, comment="#")
    df.sort_values(by=["num_p"], inplace=True, ascending=False)
    df.set_index("id", inplace=True)

    df["r200c_kpc"] = df.r200c / h
    df["m200c_msun"] = df.m200c / h / 1e10
    df["m500c_msun"] = df.m500c / h / 1e10
    df["m2500c_msun"] = df.m2500c / h / 1e10
    df["mbound_200c_msun"] = df.mbound_200c / h / 1e10

    df["Rs_kpc"] = df.Rs / h

    df["con"] = df.r200c_kpc / df.Rs_kpc

    return df, file


def largest_halo_properties(dir: Path):
    df, file = read_rockstar_halos(dir)

    main_halo = df.iloc[0]

    print(file)
    print(f"r200c_kpc: {main_halo.r200c_kpc:.1f}")
    print(f"m200c_msun: {main_halo.m200c_msun:.2f}")
    print(f"m500c_msun: {main_halo.m500c_msun:.2f}")
    print(f"m2500c_msun: {main_halo.m2500c_msun:.2f}")
    print(f"mbound_200c_msun: {main_halo.mbound_200c_msun:.2f}")
    print(f"Rs_kpc: {main_halo.Rs_kpc:.2f}")
    print(f"con: {main_halo.con:.3f}")
    return main_halo


if __name__ == '__main__':
    data = {}
    dir = Path("/home/lukas/cosmos_data/auriga-resim/data/auriga6/7_8_10")
    data["7_8_10"] = largest_halo_properties(dir)
    dir = Path("/home/lukas/cosmos_data/auriga-resim/data/auriga6/7_10_10")
    data["7_10_10"] = largest_halo_properties(dir)
    dir = Path("/home/lukas/cosmos_data/auriga-resim/data/auriga6/7_10_12")
    data["7_10_12"] = largest_halo_properties(dir)
    dir = Path("/home/lukas/cosmos_data/auriga-resim/data/auriga6/adrian_ref_new")
    data["ref"] = largest_halo_properties(dir)
    df = pd.DataFrame(data).T
    print(df)
    df.T.to_csv(Path("~/tmp/halo_comp.csv"))
