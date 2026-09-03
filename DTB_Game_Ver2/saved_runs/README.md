# Saved DTB runs

Each subdirectory is one saved notebook execution. The directory name records:

```text
cournot-{dimension}d-nonpotential__nn-{type}-{activation}_w{width}-d{depth}-p{parameter_count}__basis-{basis_size}__seed-{seed}__{UTC timestamp}
```

Set `SAVE_RUN = True` near the beginning of
`cournot_10d_nonpotential_mlp_dtb.ipynb` to create a new directory. Set it to
`False` to display the figures and mathematical metric table without writing
run artifacts. A newly generated directory appears on GitHub after the local
repository changes are committed and pushed.

Each saved run contains its configuration, figures, metric table, diagnostic
histories, serialized DTB state, and a run-specific README.
