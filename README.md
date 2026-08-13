# mars-data-analysis
Processes output from the Mouse Action Recognition system and BENTO, specifically a .annot and .npz file. Includes functions to output cleaned feature and behavior arrays, identify significant features difference between groups, and compare groups using principle components. Still in rough form. 

Input data:
- for each experiment you need a matching .annot and .npz file which begin with the same 3-digit idea
- put these pairs of files into one folder for mutant and another for control
- these file paths must be inputted into the data extractor scripts (this may be updated in future)

Functionality:
- feature_comparison_main: this is the central script, use the .ipynb to run and experiment. set a variety of filters on timing and context of a behavior, get a visualization of bout amount and timing per group, plot a specific feature and compare between groups, compute and compare AUC between groups to find siginificang groups.
- PCA_plotting: use same filters as in feature_comparison main and plot datapoints for a single behavior in two dimensions with PCA. Compare distributions between behaviors with a few statistics.

Other Notes:
- currently only works with those specific filetypes and directory structure (one folder per experimental group)
- must have behavior and feature names exactly right (look at .annot file to get these)
- automatically trims behavior and feature matrix to start at intruder_enter time which is specified in .annot. The .annot file comes from processing MARS output in BENTO. In BENTO ensure you add an intruder_enter annotation

I created this while working at the Mountou Lab at UTSW Medical Center in the summer of 2026 as part of the SURF program. Thanks to George Mountoufaris PhD.
