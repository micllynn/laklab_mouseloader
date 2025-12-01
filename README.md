This package loads rodent visual behavior experiments with associated neuropixels
data, including visualization and analysis tools.

The package also includes tools to automatically register neuropixels data
with histology, to align behavior and electrophysiological data based on shared
TTL signals, to parse outputs from RigBox Timeline and Block files in
a Pythonic manner, and to perform high-level analysis of the resulting outputs.

[The focus now is on loading Blake Russel's ReportOpto dataset and performing
analysis on it. Future work will make this more generalizable.]

## Main usage
- Import the package:
```
import npixloader as npl
```

  - `npl.ExpObj` provides the main interface to load an experiment including behavior
  and neuropixels components.
  Typically, a dataset object is loaded first using `npl.DSetObj`, which containins
  information about all experiments performed for a project.
  It is employed in the following manner:
  - 
  ```
  dset = npl.DSetObj(path_reportopto='path_to_reportopto_datastructure.csv') 
	  # loads dataset .csv, see dset.py for more information about
	  # where this is located
  exp = npl.ExpObj(dset_obj=dset, dset_ind=5)  # loads expref 5 into exp
  exp.plt_exp()  # visualizes experiment
```
![plt_exp_output](/plt_exp_out.jpg)
	
	
## Analysis
- `npix_loader.analysis_fns` contains functions to load and plot entire datasets
comprised of many experiments.

```
import npixloader.analysis_fns as npl_analysis
```

  - `npl_analysis.ephys_all = get_ephys_all()` parses a dataset for all
  experiments containing recordings from
  a particular region
  - `npl_analysis.plt_ephys_all_ordered(ephys_all)` plots a stimulus- or
  choice-aligned raster of the subset of ephys recordings from `get_ephys_all`.
  This can include ordering by time or by relative activity.
  
## Command line tools
- `npixloader/cmd_line_tools` contains simple tools to check the daily behavioral performance of mice (classical conditioning). To use this:
  - in `beh_check.py` replace the shebang line (`#!`) with the path of your conda environment taht nixpoader is run in
  - run `beh_check.sh` in the shell. It takes two arguments: a path to the behavior folder, and an optional flag `-n` that specifies the presence of noise in the lick-channel (alternate algorithm to parse licks).
  - This will automatically generate a figure of licks across each trial-type, which will be saved at the behavioral folder path as a .pdf
