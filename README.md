## Main usage
- load_exp.py contains classes to load data from single experiments, including behavior
and simultaneous neuropixels ephys.
  - `ExpObj` provides the main interface to load an experiment. Typically, a dataset object is
  loaded first, containing information about all experiments performed for a project.
  It is employed in the following manner:
  - ```
	from dset import DSetObj
	dset = DSetObj()  # loads dataset .csv, see dset.py for more information about
	                  # where this is located
	exp = ExpObj(dset_obj=dset, dset_ind=5)  # loads expref 5 into exp
	exp.plt_exp()  # visualizes experiment
  ```
- analysis_fns.py contains functions to load and plot entire datasets, comprised of
many experiments.
  - `ephys_all = get_ephys_all()` parses a dataset for all experiments containing recordings from
  a particular region
  - `plt_ephys_all_ordered(ephys_all)` plots a stimulus- or choice-aligned raster of
  the subset of ephys recordings from `get_ephys_all`. This can include ordering by time or by
  relative activity.
