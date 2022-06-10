from compare_halos import compare_halo_resolutions

for wf in ["DB2", "DB4", "DB8", "shannon"]:
    for res in [128, 256, 512]:
        for source_wf in [wf, "shannon"]:
            compare_halo_resolutions(
                ref_waveform=source_wf,
                reference_resolution=128,
                comp_waveform=wf,
                comparison_resolution=res,
                plot=False,
                plot3d=False,
                velo_halos=True,
                single=False
            )
