import traceback

from compare_halos import compare_halo_resolutions

something_failed = False
for wf in ["DB2", "DB4", "DB8", "shannon"]:
    for res in [256, 512]:
        for source_wf in [wf]:
            try:
                compare_halo_resolutions(
                    ref_waveform=source_wf,
                    reference_resolution=256,
                    comp_waveform=wf,
                    comparison_resolution=res,
                    plot=False,
                    plot3d=False,
                    velo_halos=True,
                    single=False
                )
            except Exception as e:
                traceback.print_exc()
                something_failed = True
                continue

if something_failed:
    print("\n\nAt least one comparison failed")
