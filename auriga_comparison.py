from cic import plot_cic, cic_from_radius
from paths import auriga_dir
from readfiles import read_file

for dir in auriga_dir.glob("*"):
    Xc_adrian = 56.50153741810241
    Yc_adrian = 49.40761085700951
    Zc_adrian = 49.634393647291695
    Xc = 58.25576087992683
    Yc = 51.34632916228137
    Zc = 51.68749302578122

    is_by_adrian = "arj" in dir.name
    if not dir.is_dir():
        continue
    input_file = dir / "output_0007.hdf5"
    if is_by_adrian:
        input_file = dir / "final_output.hdf5"
    print(input_file)
    df_ref, _ = read_file(input_file)
    X, Y, Z = df_ref.X.to_numpy(), df_ref.Y.to_numpy(), df_ref.Z.to_numpy()
    print()
    print(Yc - Yc_adrian)
    # shift: (-6, 0, -12)
    if not is_by_adrian:
        xshift = Xc - Xc_adrian
        yshift = Yc - Yc_adrian
        zshift = Zc - Zc_adrian
        print("shift", xshift, yshift, zshift)

        X -= 1.9312
        Y -= 1.7375
        Z -= 1.8978
    rho, extent = cic_from_radius(X, Z, 500, Xc_adrian, Yc_adrian, 5, periodic=False)

    plot_cic(
        rho, extent,
        title=str(dir.name)
    )
