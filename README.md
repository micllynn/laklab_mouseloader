## Main usage
- Import the package:
```
import npixloader as npl
```

  - `npl.ExpObj` provides the main interface to load an experiment.
  Typically, a dataset object is
  loaded first, containing information about all experiments performed for a project.
  It is employed in the following manner:
  - 
  ```
  dset = nl.DSetObj()  # loads dataset .csv, see dset.py for more information about
	                  # where this is located
  exp = nl.ExpObj(dset_obj=dset, dset_ind=5)  # loads expref 5 into exp
  exp.plt_exp()  # visualizes experiment
```
	
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
