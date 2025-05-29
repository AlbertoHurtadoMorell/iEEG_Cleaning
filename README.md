# iEEG Cleaning and Analysis
Preprocessing pipeline including artifact detection to prepare iEEG for analysis. 
Additionally, three analysis methods are given as an example use:
  - Univariate Circular Correlation
  - Multivariate SVM adapted to show the null distribution of the prediction errors
  - Forward Encoding model

In the following paragraphs all functions found both in preprocessing helper functions and analysis helper functions will be briefly explained for context:

  - bipolar_reference_seeg: It accepts an MNE Raw object containing SEEG recordings, groups and numerically orders contacts by electrode, constructs adjacent anode–cathode pairs, applies MNE’s bipolar referencing to produce virtual channels, and returns a new Raw object with the bipolar-referenced data.

  The following 4 functions (padding, remove_small_segments, eegfilt artifact_detection) are based on the logic proposed by Staresina et al 2015:

  - padding: It accepts a one‐dimensional binary array data in which ones mark detected artifact samples and an integer padding_value specifying how many samples to extend before and after each artifact cluster, it then performs connected‐component labeling to identify contiguous artifact segments, sets to one the padding_value samples immediately preceding and following each segment, and returns the modified binary array with these padded artifact regions.

  - remove_small_segments: It takes as inputs a one-dimensional binary array data, in which ones denote artifact samples, and an integer min_seg_length specifying the minimum acceptable length for non-artifact segments; it identifies contiguous runs of zeros shorter than min_seg_length, replaces those short segments with ones to mark them as artifacts, and returns the modified binary array.

  - eegfilt: It accepts as input a two-dimensional array of electrophysiological recordings (channels × time points), the sampling frequency, optional low- and high-cutoff frequencies, an epoch length in frames, an optional filter order, and a reverse-filter flag; it divides the data into equal-length segments, designs a finite impulse response filter to meet the specified passband or stopband criteria, applies zero-phase forward–backward filtering to each segment, and returns the filtered data together with the FIR filter coefficients.

  - The function artifact_detection accepts as inputs a two‐dimensional array of electrophysiological recordings, two z‐score thresholds for amplitude/gradient/envelope outlier detection (std_thres and std_thres2), an integer padding_value specifying how many samples to extend around each detected artifact, and an integer min_seg_length defining the minimum allowable length of non‐artifact segments; it computes per‐channel amplitude, gradient, and high‐pass envelope z‐scores to mark candidate artifact samples, applies the padding and removes spuriously short clean segments, and returns a copy of the data array in which all samples classified as artifacts have been replaced by NaN.

Subsequently, in an effort to increase robustness, custom modifications were applied to this logic to increase robustness and low-level noise detection:

  - The function artifact_detection_tuned implements a robust, median/MAD‐based artifact detection pipeline for multi‐channel electrophysiological recordings by thresholding amplitude, gradient, and high‐frequency envelope, pruning and retaining clusters according to minimum duration and overshoot criteria, extending and filling artifact segments via padding and gap‐filling parameters, and returns a copy of the input data in which all detected artifact samples are replaced by NaN.

  - The function convert_to_mne constructs a new MNE Raw object from a cleaned NumPy array and an existing Raw template by extracting the channel names and sampling frequency, assigning each channel either “seeg” or “stim” type based on its name, creating an MNE Info structure, and returning a RawArray containing the cleaned data.

The interactive rejection workflow begins by calling reject_visual_mne on the pre‐epoched data with a chosen metric (for example “std”), which displays a combined heatmap, scatter and spectrum view to let the user mark bad trials and channels and returns two outputs: the list of rejected trial indices and the list of rejected channel names. These outputs are then passed, together with the original Epochs object and its channel name list, to plot_rejection_with_rejectlog, which constructs an Autoreject RejectLog and produces both an annotated trace plot and a summary visualization of which epochs and channels were flagged as bad.




