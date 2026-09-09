# Saved Cournot 3D plots

Run: `cournot_3d_stochastic_mlp-tanh_w32_d4_p3395_basis128_seed2026_20260909T005347Z`.

These four PNGs were recovered from the embedded outputs in the [executed notebook](https://github.com/sun-mengwei/dtb-colab-experiments/blob/c84639cec27e21634880b3c1eae25a4aace84f1a/DTB_Game_Ver2/cournot_3d_nonpotential_stochastic_mlp_dtb.ipynb). The PNG bytes are preserved exactly, with no regeneration or rescaling. The notebook reports that the original run was saved under `/content/dtb-colab-experiments/DTB_Game_Ver2/saved_runs/cournot_3d_stochastic_mlp-tanh_w32_d4_p3395_basis128_seed2026_20260909T005347Z` in Colab.

| File | Plot |
| --- | --- |
| [dtb_3d_cloud_snapshots.png](dtb_3d_cloud_snapshots.png) | Neural-DTB 3D cloud snapshots |
| [em_3d_cloud_snapshots.png](em_3d_cloud_snapshots.png) | Euler–Maruyama 3D cloud snapshots |
| [dtb_vs_em_x1_x2.png](dtb_vs_em_x1_x2.png) | Matched DTB / EM snapshots on coordinates 1 and 2 |
| [tangent_diagnostics.png](tangent_diagnostics.png) | DTB tangent diagnostics |

## Recovery scope

These are notebook display images, not the original PNGs exported by `savefig(..., dpi=180)`. Raw particle histories, serialized arrays, and the other original Colab run artifacts were not retrieved. This directory is a plot recovery, not a complete run-data backup.

The [plot manifest](plot_manifest.json) records each image's notebook cell, output index, dimensions, and SHA-256 checksum.
