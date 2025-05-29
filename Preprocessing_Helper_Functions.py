Python 3.13.0 (v3.13.0:60403a5409f, Oct  7 2024, 00:37:40) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import warnings
import mne
import scipy.stats as sp_stats
from scipy.stats import median_abs_deviation as mad
import scipy.signal as sp_signal
from scipy.signal import firwin, filtfilt, hilbert
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import RectangleSelector, RadioButtons, CheckButtons, Button
 
try:
    from autoreject import RejectLog
except ImportError:
    RejectLog = None
 
 def bipolar_reference_seeg(raw):

#   raw_bipolar = bipolar_reference_seeg(raw)
#    Applies bipolar referencing to SEEG recordings by grouping channels by electrode, ordering contacts numerically, and subtracting each adjacent pair.
#
#    raw           MNE Raw object containing SEEG data with channel names encoding electrode and contact (e.g. “A1”, “A2”, …).
#    raw_bipolar   MNE Raw object containing virtual bipolar channels named “contact_i–contact_{i+1}”, with original references dropped.

    ch_names = raw.ch_names #get channel names
    electrode_groups = {} #create empty dictionary to save electrodes (keys) and contacts (values)
 
    for ch in ch_names: #for each channel
        prefix = ''.join(filter(str.isalpha, ch)) #Extract prefix, so for example from A1 extract A
        if prefix not in electrode_groups: #if it is not already in the dictionary, add it as a key
            electrode_groups[prefix] = []
        electrode_groups[prefix].append(ch) #add the channel in the dictionary as a value to its corresponding key
 
    for prefix in electrode_groups: #in this case we will sort numerically in case something went wrong and they are not ordered
        try:
            electrode_groups[prefix].sort(key=lambda x: int(''.join(filter(str.isdigit, x)))) #isdigit gets the digit from the string, and then sort will sort them out
        except ValueError: #to fix error, because we also have trig channel which does not have a number
            print(f"Skipping non-numeric channel: {electrode_groups[prefix]}")
 
    anodes, cathodes, bipolar_names = [], [], [] #generate empty lists that are required by the mne bipolar refer function
    for prefix, contacts in electrode_groups.items():
        contacts = [c for c in contacts if any(char.isdigit() for char in c)] 
        for i in range(len(contacts) - 1): 
            anodes.append(contacts[i])
            cathodes.append(contacts[i+1])
            bipolar_names.append(f"{contacts[i]}-{contacts[i+1]}") #name of virtual channel created after substraction
 
     
    raw_bipolar = mne.set_bipolar_reference(raw, anodes, cathodes, ch_name=bipolar_names, drop_refs=True) #apply function

    return raw_bipolar

def padding(data, padding_value): # Binary vector where 1 indicates an artifact. padding_value: an integer indicating how many samples before and after a detected artifact should be added.

#   padded = padding(data, padding_value)
#    Extends binary artifact markers by setting to one the specified number of samples immediately before and after each contiguous artifact cluster.
#
#    data            One-dimensional binary array in which “1” denotes an artifact sample.
#    padding_value   Integer number of samples to extend on each side of every detected artifact cluster.
#    padded          One-dimensional binary array with padded artifact regions marked as “1”.

    labeled_array, num_features = label(data) #this gets the clusters of 1s, in labeled_array each cluster of 1s gets a unique label and num_features saves the total number of clusters detected.
    
    if num_features > 0:
        for i in range(1, num_features + 1):
            clust = np.where(labeled_array == i)[0] #np.where gets indices, clust is an array of indices where that cluster of 1s is located.
            
            # Left padding (before the artifact)
            if clust[0] - padding_value >= 0:  # Can pad left
                data[clust[0] - padding_value : clust[0]] = 1
            
            # Right padding (starting at last artifact sample)
            if clust[-1] + padding_value < len(data):  # Can pad right
                data[clust[-1] : clust[-1] + padding_value] = 1
    
    return data

def remove_small_segments(data, min_seg_length):

#   cleaned = remove_small_segments(data, min_seg_length)
#    Identifies non-artifact segments (zeros) shorter than a minimum length and marks them as artifacts.
#
#    data            One-dimensional binary array in which “1” denotes artifact samples.
#    min_seg_length  Integer specifying the minimum length of a zero-run to preserve as non-artifact.
#    cleaned         One-dimensional binary array with runs shorter than min_seg_length set to “1”.

    mks = ~data.astype(bool)  #invert data --> 1s -> 0s (artifacts become non-artifacts) and 0s -> 1s (non-artifact regions become potential candidates for removal).
    # data.astype(bool): Converts 0 → False, 1 → True, ~ (bitwise NOT): Inverts the boolean array
    labeled_array, num_features = label(mks)  #identify connected components of 0s

    if num_features > 0:
        segment_sizes = np.bincount(labeled_array.flat)[1:]  #compute size of each segment.
        small_segments = np.where(segment_sizes < min_seg_length)[0] + 1  #any segment smaller than min_seg_length is converted to 1s.

        for seg in small_segments:
            data[labeled_array == seg] = 1 

    return data

def eegfilt(data, srate, locutoff=None, hicutoff=None, epochframes=0, filtorder=None, revfilt=0): # epochframes: How many frames per segment to filter (default 0 = full trace)

#   [smoothdata, filtwts] = eegfilt(data, srate, locutoff, hicutoff, epochframes, filtorder, revfilt)
#    Designs and applies a zero-phase FIR filter (bandpass, high-pass, or low-pass) to multi-channel data in equal-length segments.
#
#    data            Two-dimensional NumPy array (n_channels × n_times) of raw signals.
#    srate           Sampling frequency in Hz.
#    locutoff        Low-cutoff frequency in Hz (None to omit).
#    hicutoff        High-cutoff frequency in Hz (None to omit).
#    epochframes     Integer segment length in samples (0 to filter entire recording at once).
#    filtorder       Optional integer filter order; if None, computed from srate and cutoff(s).
#    revfilt         Boolean flag (0/1) to invert filter pass/stop behavior.
#    smoothdata      Two-dimensional array of filtered signals.
#    filtwts         One-dimensional array of FIR filter coefficients.

    chans, frames = data.shape
    nyq = srate * 0.5
    minfac = 3 # Filter Order Scaling Factor: This is a scaling factor used to calculate the initial filter order based on the sampling rate and the cutoff frequency.
    # If minfac is too small: The filter will be too short → poor frequency separation
    # If minfac is too large: The filter will be too long → more computation time, needs more padding...
    min_filtorder = 15 # Minimum Filter Order: This is the minimum order of the filter. If the calculated filter order is less than this value, it will be set to this value.
    trans = 0.15 # It controls the width of the transition band in the filter design, it affects how “sharp” the cutoff is.

    if locutoff is not None and hicutoff is not None and locutoff > hicutoff:
        raise ValueError("locutoff must be <= hicutoff.")
    if (locutoff is not None and locutoff < 0) or (hicutoff is not None and hicutoff < 0):
        raise ValueError("Cutoff frequencies must be non-negative.")
    if (locutoff is not None and locutoff >= nyq) or (hicutoff is not None and hicutoff >= nyq):
        raise ValueError("Cutoff frequencies must be less than Nyquist.")

    if filtorder is None:
        if locutoff is not None:
            filtorder = minfac * int(srate / locutoff)
        elif hicutoff is not None:
            filtorder = minfac * int(srate / hicutoff)
        filtorder = max(filtorder, min_filtorder)

    # FIR filters, especially with filtfilt, which applies the filter forward and backward, need a buffer of data around the edges to function properly.
    # Here we manage data segmentation to fix that.

    if epochframes == 0: # If epochframes isn't provided, filter the entire time series as a single chunk.
        epochframes = frames # frames is the total number of timepoints in the signal. If epochframes is 0, it means we want to filter the entire signal as a whole chunk.
    epochs = frames // epochframes # Each epoch must be ≥ 3 × filter order (to avoid edge effects during filtering). This computes how many equal-sized segments we’ll divide the signal into.
    if epochs * epochframes != frames: # Ensures that the signal can be perfectly divided into chunks.
        raise ValueError("epochframes does not evenly divide frames.")
    if filtorder * 3 > epochframes: # FIR filters require enough padding to operate properly, and when using filtfilt, zero-phase filtering, the data is mirrored around the edges.
    # the segment should be at least 3× the filter length to allow proper padding, filtering.
        raise ValueError("epochframes must be at least 3 times the filtorder.")

    # Design a Finite Impulse Response (FIR) filter
    if locutoff is not None and hicutoff is not None:
        bands = [0, locutoff * (1 - trans), locutoff, hicutoff, hicutoff * (1 + trans), nyq] #The bands array defines points in frequency space.
        desired = [0, 0, 1, 1, 0, 0] # The desired array defines the desired amplitude response of the filter at each of those points.
        # trans defines a small slope for the transition band so that the filter is practical to implement.
        cutoff = [bands[2], bands[3]]  # Passband
    elif locutoff is not None:
        bands = [0, locutoff * (1 - trans), locutoff, nyq]
        desired = [0, 0, 1, 1]
        cutoff = [bands[2]]
    elif hicutoff is not None:
        bands = [0, hicutoff, hicutoff * (1 + trans), nyq]
        desired = [1, 1, 0, 0]
        cutoff = [bands[1]]
    else:
        raise ValueError("You must provide a non-zero low or high cutoff frequency.")

    if revfilt:
        desired = [1 - d for d in desired]

    # Normalize cutoff(s)
    normalized_cutoff = [c / nyq for c in cutoff] # firwin expects cutoff frequencies in the range of 0.0 to 1.0 (as a fraction of the Nyquist frequency)
    filtwts = firwin(filtorder + 1, normalized_cutoff, pass_zero=desired[0] == 1) # filtwts array contains the filter kernel

    # Apply filter
    smoothdata = np.zeros_like(data) # empty array of the same shape as the input to store the filtered signal.
    for e in range(epochs):
        for c in range(chans):
            segment = data[c, e * epochframes:(e + 1) * epochframes]
            smoothdata[c, e * epochframes:(e + 1) * epochframes] = filtfilt(filtwts, 1, segment) # filtfilt applies the filter forward and backward to avoid phase distortion. 1 is de denominator of the filter transfer function, which is 1 for FIR filters.

    return smoothdata, filtwts

def artifact_detection(data, std_thres, std_thres2, padding_value, min_seg_length):

#   data_clean = artifact_detection(data, std_thres, std_thres2, padding_value, min_seg_length)
#    Detects artifacts in each channel by thresholding amplitude z-score, gradient z-score, and high-pass envelope z-score; applies padding and removes spuriously short clean segments, replacing artifact samples with NaN.
#
#    data            Two-dimensional NumPy array (n_channels × n_times) of raw signals.
#    std_thres       Z-score threshold for individual metrics (amp, grad, envelope).
#    std_thres2      Secondary, more stringent z-score threshold requiring joint conditions on grad/envelope.
#    padding_value   Integer number of samples to pad around detected artifacts.
#    min_seg_length  Integer minimum length of clean segments to preserve.
#    data_clean      Two-dimensional array of same shape, with artifact samples set to NaN.

    data_clean = np.copy(data)

    for chani in range(data.shape[0]): #Iterates over each EEG channel.
        channel_data = data[chani, :] #Extracts a single-channel time series

        #now it will calculate the z-score, large values indicate outliers

        z_score_amp = (channel_data - np.mean(channel_data)) / np.std(channel_data)
        grad = np.diff(channel_data, append=np.nan) #Computes first derivative
        z_score_grad = (grad - np.nanmean(grad)) / np.nanstd(grad) #Normalizes it using Z-score transformation.

        #Ensure data is 2D for eegfilt function

        if channel_data.ndim == 1:
            channel_data = channel_data[np.newaxis, :]  #Reshape to (1, samples)

        hpf_data, _ = eegfilt(channel_data, 500, None, 249)
        hpf_data = np.abs(hilbert(hpf_data))
        z_score_hpf_d = (hpf_data - np.mean(hpf_data)) / np.std(hpf_data)

        markers = np.zeros_like(channel_data) #create array of 0s
        condition1 = (z_score_amp > std_thres) | (z_score_grad > std_thres) | (z_score_hpf_d > std_thres)
        condition2 = (z_score_amp > std_thres2) & ((z_score_grad > std_thres2) | (z_score_hpf_d > std_thres2))
        markers[condition1 | condition2] = 1 #mark artifact locations as 1s

        new_trace = padding(markers, padding_value)  
        markers = remove_small_segments(new_trace, min_seg_length)  

        data_clean[chani, markers.flatten() == 1] = np.nan #Replaces artifact samples with NaN

    return data_clean

def artifact_detection_tuned(
    data, # 2D array: (n_channels, n_samples)
    amp_thr, # amplitude threshold in MAD‐units
    grad_thr, # gradient threshold in MAD‐units     
    hpf_thr,  # high-frequency envelope threshold in MAD‐units     
    min_art_sec,   # 0.005  → 5 ms minimum artifact duration, minimum artifact run length in seconds
    pad_samps,     # number of samples to pad around each artifact cluster
    fill_gap_sec,  # 0.02   → fill any clean gap ≤20 ms, maximum clean-gap length in seconds to “fill” 
    overshoot_fac=1.3,  # keep VERY large spikes even if short, allow any cluster with a peak > amp_thr × overshoot_fac to survive even if very short
    sfreq=500
):

#   data_clean = artifact_detection_tuned(data, amp_thr, grad_thr, hpf_thr, min_art_sec, pad_samps, fill_gap_sec, overshoot_fac, sfreq)
#    Implements a robust, MAD-based artifact detector that thresholds amplitude, gradient, and high-frequency envelope; prunes or retains clusters by duration and peak magnitude; applies padding and fills short clean gaps; returns data with artifacts set to NaN.
#
#    data            Two-dimensional NumPy array (n_channels × n_times) of raw signals.
#    amp_thr         Amplitude threshold in MAD units.
#    grad_thr        Gradient threshold in MAD units.
#    hpf_thr         High-frequency envelope threshold in MAD units.
#    min_art_sec     Minimum artifact duration in seconds.
#    pad_samps       Number of samples to pad around each artifact cluster.
#    fill_gap_sec    Maximum clean-gap duration in seconds to fill within artifact regions.
#    overshoot_fac   Multiplier allowing very large spikes to survive even if shorter than min_art_sec.
#    sfreq           Sampling frequency in Hz.
#    data_clean      Two-dimensional array of same shape, with artifact samples replaced by NaN.

    nchan, nsamp = data.shape
    out = data.copy()
    min_art = int(min_art_sec  * sfreq)
    fill_gap = int(fill_gap_sec * sfreq)
    
    for ch in range(nchan):
        x = data[ch]

        # Median (med_x) and MAD (mad_x) replace mean/std
        med_x   = np.median(x)
        mad_x   = mad(x, scale='normal')
        z_amp   = (x - med_x) / mad_x
        
        # First derivative via np.diff, then median/MAD normalization. Sharp edges (steep slopes) show up as large z_grad.
        grad    = np.diff(x, append=x[-1])
        med_g   = np.median(grad)
        mad_g   = mad(grad, scale='normal')
        z_grad  = (grad - med_g) / mad_g
        
        hpf, _  = eegfilt(x[np.newaxis,:], sfreq, None, 240)
        env     = np.abs(hilbert(hpf)).flatten()
        med_h   = np.median(env)
        mad_h   = mad(env, scale='normal')
        z_hpf   = (env - med_h) / mad_h
        
        # A sample is flagged if any one of the three metrics crosses its own threshold.
        mask0 = (
              (z_amp  > amp_thr) # Anything where z_amp > amp_thr is a heavy outlier in raw amplitude.
            | (z_grad > grad_thr)
            | (z_hpf  > hpf_thr)
        ).astype(int)
        
        
        labeled, nseg = label(mask0) # label finds each contiguous run (cluster) of 1’s in mask0.
        for seg_id in range(1, nseg+1): # For each cluster, we check: Length, if shorter than min_art samples, and peak amplitude (max_z), also below a “giant‐spike” cutoff (amp_thr × overshoot_fac).
            idx = np.where(labeled==seg_id)[0]
            max_z = max(z_amp[idx].max(),
                        z_grad[idx].max(),
                        z_hpf[idx].max())
            
            if idx.size < min_art and max_z < amp_thr * overshoot_fac:
                mask0[idx] = 0
        
        # In effect, we drop short, small sized clusters, but keep any cluster that’s either long enough or truly huge.
        
        mask1 = padding(mask0, pad_samps) # extend every flagged cluster by pad_samps on both sides ensuring we capture pre and post artifact effects.
        
        mask2 = remove_small_segments(mask1, fill_gap) # This is the opposite of the previous prune, here we fill any clean stretch shorter than fill_gap.
        # It ensures that we don’t leave tiny holes of NaNs inside a broader artifact region, so artifacts appear as solid blocks.
        
        out[ch, mask2==1] = np.nan
    
    return out

def convert_to_mne(data_clean, raw_template):

#   raw_clean = convert_to_mne(data_clean, raw_template)
#    Constructs an MNE RawArray from a cleaned NumPy array using channel names and sampling rate from a template Raw object, assigning “seeg” or “stim” types automatically.
#
#    data_clean      Two-dimensional NumPy array (n_channels × n_times) of cleaned data.
#    raw_template    MNE Raw object providing channel names and sfreq.
#    raw_clean       MNE RawArray containing the cleaned data with corresponding Info.

    ch_names = raw_template.ch_names
    sfreq = raw_template.info['sfreq']
    
    # Assign 'seeg' to all channels except for the trigger channel
    ch_types = ['stim' if 'trig' in ch.lower() else 'seeg' for ch in ch_names]
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw_clean = mne.io.RawArray(data_clean, info)
    
    return raw_clean

try:
    from autoreject import RejectLog
except ImportError:
    RejectLog = None 

def calculate_metric(data_in, metric='var'):

#   [metric_data, found_nans] = calculate_metric(data_in, metric)
#    Computes specified summary statistic across time for each channel and epoch, handling NaNs and SciPy version differences.
#
#    data_in         Three-dimensional NumPy array (n_epochs × n_channels × n_times).
#    metric          String specifying the metric to compute: 'var', 'std', 'mad', '1/var', 'min', 'max', 'maxabs', 'range' (or 'ptp'), or 'kurtosis'.
#    metric_data     Two-dimensional array (n_channels × n_epochs) of computed metric values.
#    found_nans      Boolean flag indicating whether any NaNs were present in data_in.

    found_nans = False

    with np.errstate(invalid='ignore'):

        if np.isnan(data_in).any():
            found_nans = True

    n_epochs, n_channels, n_times = data_in.shape
    # Initialize metric_data with NaNs, so if a calculation fails or is skipped, NaN remains.
    metric_data = np.full((n_channels, n_epochs), np.nan)

    # Suppress RuntimeWarnings locally for nan-aware functions if they still emit them
    # (e.g. "Degrees of freedom <= 0 for slice"" in nanvar/nanstd for all-NaN slices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i_chan in range(n_channels):
            chan_data = data_in[:, i_chan, :] # Shape: (n_epochs, n_times)

            if metric == 'var':
                metric_data[i_chan, :] = np.nanvar(chan_data, axis=1)
            elif metric == 'std':
                metric_data[i_chan, :] = np.nanstd(chan_data, axis=1)
            elif metric == 'mad':
                try:
                    # SciPy >= 1.7.0 has nan_policy for median_abs_deviation
                    metric_data[i_chan, :] = sp_stats.median_abs_deviation(
                        chan_data, axis=1, scale='normal', nan_policy='omit'
                    )
                except TypeError:
                    # Fallback for older SciPy versions
                    warnings.warn(
                        "SciPy version < 1.7.0 detected."
                        "MAD calculation cannot ignore NaNs natively via 'nan_policy'. "
                        "Manually skipping NaNs. Result may be NaN if all data in an epoch is NaN.",
                        RuntimeWarning, stacklevel=2
                    )
                    for i_epoch in range(n_epochs):
                        epoch_data = chan_data[i_epoch, :]
                        if np.all(np.isnan(epoch_data)):
                            metric_data[i_chan, i_epoch] = np.nan
                            continue
                        metric_data[i_chan, i_epoch] = sp_stats.median_abs_deviation(
                            epoch_data[~np.isnan(epoch_data)], scale='normal'
                        )
            elif metric == '1/var':
                variances = np.nanvar(chan_data, axis=1)
                # Add epsilon to variance to prevent division by zero, then compute inverse
                variances[variances == 0] = np.finfo(float).eps
                metric_data[i_chan, :] = 1.0 / variances
            elif metric == 'min':
                metric_data[i_chan, :] = np.nanmin(chan_data, axis=1)
            elif metric == 'max':
                metric_data[i_chan, :] = np.nanmax(chan_data, axis=1)
            elif metric == 'maxabs':
                metric_data[i_chan, :] = np.nanmax(np.abs(chan_data), axis=1)
            elif metric in ['range', 'ptp']: # Peak-to-peak
                max_vals = np.nanmax(chan_data, axis=1)
                min_vals = np.nanmin(chan_data, axis=1)
                metric_data[i_chan, :] = max_vals - min_vals
            elif metric == 'kurtosis':
                try:
                    # SciPy >= 0.19.0 has nan_policy for kurtosis
                    metric_data[i_chan, :] = sp_stats.kurtosis(
                        chan_data, axis=1, nan_policy='omit'
                    )
                except TypeError:
                    # Fallback for older SciPy versions
                    warnings.warn(
                        "Kurtosis calculation cannot ignore NaNs natively via 'nan_policy'."
                        "Manually skipping NaNs. Result may be NaN if all data in an epoch is NaN.",
                        RuntimeWarning, stacklevel=2
                    )
                    for i_epoch in range(n_epochs):
                        epoch_data = chan_data[i_epoch, :]
                        if np.all(np.isnan(epoch_data)):
                            metric_data[i_chan, i_epoch] = np.nan
                            continue
                        metric_data[i_chan, i_epoch] = sp_stats.kurtosis(
                            epoch_data[~np.isnan(epoch_data)]
                        )
            else:
                # Metric not implemented
                warnings.warn(f"Metric '{metric}' not implemented.", UserWarning, stacklevel=2)
                return None, found_nans # Return None for metric_data

    return metric_data, found_nans

def reject_visual_mne(epochs, metric='var', sfreq=None):

#   [rejected_trials, rejected_channels] = reject_visual_mne(epochs, metric, sfreq)
#    Launches an interactive Matplotlib interface for rejecting epochs and channels based on a chosen metric heatmap, scatter summaries, and optional spectrum, returning user-selected rejections.
#
#    epochs               MNE Epochs object containing segmented data.
#    metric               String specifying metric for rejection ('var', 'std', 'mad', 'maxabs', 'ptp', 'kurtosis').
#    sfreq                Sampling frequency in Hz (defaults to epochs.info['sfreq']).
#    rejected_trials      List of integer epoch indices marked as bad by the user.
#    rejected_channels    List of channel names marked as bad by the user.

    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("Input must be an MNE Epochs object.")

    data = epochs.get_data(picks=['eeg', 'meg', 'ecog', 'seeg'])
    if data is None or data.size == 0:
        warnings.warn("No data found for EEG, MEG, ECoG, or SEEG channels. Returning empty selections.", UserWarning)
        return [], []

    n_epochs, n_channels, n_times = data.shape
    ch_names = epochs.copy().pick_types(eeg=True, meg=True, ecog=True, seeg=True).ch_names # Ensure ch_names match data

    if sfreq is None:
        sfreq = epochs.info['sfreq']

    current_metric = metric
    rejected_trials = set()  # Store indices of rejected trials
    rejected_channels_idx = set()  # Store indices of rejected channels
    bad_channel_names = set() # Store names of rejected channels for final output
    calculate_spectrum = True # Flag to control spectrum calculation and display
    _nan_warning_issued = False # Flag to ensure NaN warning is shown only once

    plot_elements = {}  # Dictionary to store matplotlib artists for easy access/update
    selectors = {}      # Dictionary to store RectangleSelector objects

    def _calculate_metric_and_warn(data_to_calc, metric_name):
        nonlocal _nan_warning_issued
        metric_res, found_nans = calculate_metric(data_to_calc, metric_name)
        if found_nans and not _nan_warning_issued:
            warnings.warn(
                "Input data contains NaNs. Metrics will be calculated ignoring NaNs. "
                "Resulting metrics may be NaN if all data points for a specific "
                "channel/epoch calculation are NaN.",
                RuntimeWarning, stacklevel=3 # Adjust stacklevel as needed
            )
            _nan_warning_issued = True # Set flag after the first warning
        return metric_res

    # Initial metric calculation
    metric_data = _calculate_metric_and_warn(data, current_metric)
    if metric_data is None: # Handle invalid initial metric
        warnings.warn(
            f"Initial metric '{current_metric}' is invalid or calculation failed. Defaulting to 'var'.",
            UserWarning
        )
        current_metric = 'var'
        metric_data = _calculate_metric_and_warn(data, current_metric)
        if metric_data is None:
            raise ValueError("Could not calculate the default 'var' metric. Check input data.")

    if np.all(np.isnan(metric_data)):
         warnings.warn(
             "Initial metric calculation resulted in all NaNs. "
             "Check input data or the chosen metric. The visualization might be uninformative.",
             RuntimeWarning, stacklevel=2
         )

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(
        4, 3, width_ratios=[3, 2, 1.5], height_ratios=[2, 2, 0.7, 0.4],
        hspace=0.4, wspace=0.3 # Added wspace for better default layout
    )
    ax_summary = fig.add_subplot(gs[0, 0])          # Heatmap of metric_data
    ax_chan_summary = fig.add_subplot(gs[0, 1], sharey=ax_summary) # Channel means
    ax_trial_summary = fig.add_subplot(gs[1, 0], sharex=ax_summary) # Trial means
    ax_spectrum = fig.add_subplot(gs[1, 1])         # Average spectrum
    ax_controls = fig.add_subplot(gs[0:2, 2])       # Text info and status
    ax_metric_radio = fig.add_subplot(gs[2, 0])     # Radio buttons for metric selection
    ax_spec_check = fig.add_subplot(gs[2, 1])       # Checkbox for spectrum toggle
    ax_quit_button = fig.add_subplot(gs[3, 2])      # Quit button

    # Configure axes appearance
    plt.setp(ax_chan_summary.get_yticklabels(), visible=False)
    plt.setp(ax_trial_summary.get_xticklabels(), visible=False)
    ax_controls.axis('off') # No ticks or frame for the controls panel

    def update_spectrum_plot():

        ax_spectrum.clear()
        if calculate_spectrum:
            good_trials_idx = sorted(list(set(range(n_epochs)) - rejected_trials))
            good_channels_idx = sorted(list(set(range(n_channels)) - rejected_channels_idx))

            if not good_trials_idx or not good_channels_idx:
                ax_spectrum.text(0.5, 0.5, 'No good data selected\nfor spectrum',
                                 ha='center', va='center', transform=ax_spectrum.transAxes)
                ax_spectrum.set_title('Avg Spectrum')
            else:
                # Select only good trials and channels for spectrum calculation
                good_data = data[np.ix_(good_trials_idx, good_channels_idx, np.arange(n_times))]

                with warnings.catch_warnings(): # Suppress warnings from nanmean (e.g. all-NaN slice)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    avg_data_for_spectrum = np.nanmean(good_data, axis=(0, 1)) # Average across good trials and channels

                if np.all(np.isnan(avg_data_for_spectrum)): # Check if avg data is all NaNs
                    warnings.warn(
                        "Average time series for spectrum calculation is all NaNs. Skipping Welch calculation.",
                        RuntimeWarning, stacklevel=2
                    )
                    ax_spectrum.text(0.5, 0.5, 'Cannot compute spectrum\n(average data is all NaN)',
                                     ha='center', va='center', transform=ax_spectrum.transAxes)
                    ax_spectrum.set_title('Avg Spectrum (Error)')
                elif np.any(np.isnan(avg_data_for_spectrum)): # Check for any remaining NaNs (should ideally not happen if good_data is not all NaN)
                     warnings.warn(
                        "Average time series for spectrum calculation contains some NaNs. "
                        "This might affect Welch calculation or indicate an issue.",
                        RuntimeWarning, stacklevel=2
                    )
                     ax_spectrum.text(0.5, 0.5, 'Cannot compute spectrum\n(NaNs in avg data)',
                                     ha='center', va='center', transform=ax_spectrum.transAxes)
                     ax_spectrum.set_title('Avg Spectrum (Error)')
                else:
                    freqs, psd = sp_signal.welch(avg_data_for_spectrum, fs=sfreq, nperseg=min(n_times, 256))
                    ax_spectrum.semilogy(freqs, psd)
                    ax_spectrum.set_xlabel('Frequency (Hz)')
                    ax_spectrum.set_ylabel('PSD (Power/Hz)')
                    ax_spectrum.set_title('Avg Spectrum (Good Data)')
                    ax_spectrum.grid(True, linestyle=':')
                    ax_spectrum.set_xlim([freqs[0], sfreq / 2]) # Show up to Nyquist
        else: # Spectrum calculation is OFF
            ax_spectrum.text(0.5, 0.5, 'Spectrum calculation\ndisabled',
                             ha='center', va='center', transform=ax_spectrum.transAxes)
            ax_spectrum.set_title('Avg Spectrum (Disabled)')
            # Remove ticks and labels when disabled for a cleaner look
            ax_spectrum.tick_params(axis='both', which='both', left=False, bottom=False,
                                    labelleft=False, labelbottom=False)
        fig.canvas.draw_idle()

    def update_plots():

        nonlocal metric_data # Allow modification of metric_data from outer scope
        metric_data = _calculate_metric_and_warn(data, current_metric)
        if metric_data is None:
            warnings.warn(f"Failed to calculate metric '{current_metric}'. Plots will not update.", UserWarning)
            return # Cannot proceed if metric calculation failed

        if np.all(np.isnan(metric_data)):
             # This specific warning is useful even if the general NaN warning was shown previously
             warnings.warn(
                 f"Metric '{current_metric}' resulted in all NaNs for the current data. "
                 "The visualization might be uninformative.",
                 RuntimeWarning, stacklevel=2
             )

        # Define good trials and channels based on current selections
        good_trials_idx = sorted(list(set(range(n_epochs)) - rejected_trials))
        good_channels_idx_for_summary = sorted(list(set(range(n_channels)) - rejected_channels_idx))


        # Clear old plot elements before redrawing
        for key in ['summary_im', 'chan_scatter_good', 'chan_scatter_bad',
                    'trial_scatter_good', 'trial_scatter_bad']:
             if key in plot_elements and plot_elements[key] is not None:
                 # Some elements like images might not have a 'remove' method,
                 # clearing axes is safer. Scatter plots are typically on axes that are cleared.
                 pass # Axes are cleared below

        ax_summary.clear()

        current_cmap = matplotlib.colormaps['viridis'].copy()
        current_cmap.set_bad(color='grey', alpha=0.5) # Color for NaN values in the heatmap
        im = ax_summary.imshow(metric_data, aspect='auto', cmap=current_cmap,
                               origin='lower', interpolation='nearest')
        plot_elements['summary_im'] = im
        ax_summary.set_title(f'Summary Metric ({current_metric})')
        ax_summary.set_xlabel('Trial Number')
        ax_summary.set_ylabel('Channel Number')

        for r_chan_idx in rejected_channels_idx:
            ax_summary.axhline(r_chan_idx, color='white', linestyle='--', alpha=0.7, lw=0.8)
        for r_trial_idx in rejected_trials:
            ax_summary.axvline(r_trial_idx, color='white', linestyle='--', alpha=0.7, lw=0.8)

        ax_chan_summary.clear()
        with warnings.catch_warnings(): # Suppress warnings from nanmean (e.g. all-NaN slice)
            warnings.simplefilter("ignore", category=RuntimeWarning)

            chan_means = np.nanmean(metric_data[:, good_trials_idx], axis=1) if good_trials_idx else np.full(n_channels, np.nan)

        chan_indices_all = np.arange(n_channels)

        good_ch_mask = np.ones(n_channels, dtype=bool)
        if list(rejected_channels_idx): # Check if set is not empty
             good_ch_mask[list(rejected_channels_idx)] = False

        plot_elements['chan_scatter_good'] = ax_chan_summary.scatter(
            chan_means[good_ch_mask], chan_indices_all[good_ch_mask],
            c='blue', marker='.', label='Good Ch'
        )
        if np.any(~good_ch_mask): # If there are any bad channels
            plot_elements['chan_scatter_bad'] = ax_chan_summary.scatter(
                chan_means[~good_ch_mask], chan_indices_all[~good_ch_mask],
                c='red', marker='x', label='Bad Ch'
            )
        ax_chan_summary.set_xlabel(f'Mean {current_metric} (good trials)')
        ax_chan_summary.tick_params(axis='y', which='both', left=False, labelleft=False) # Y-axis shared
        ax_chan_summary.grid(True, axis='x', linestyle=':')
        ax_chan_summary.set_ylim(ax_summary.get_ylim()) # Ensure y-limits match summary plot

        ax_trial_summary.clear()
        with warnings.catch_warnings(): # Suppress warnings from nanmean
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Calculate mean metric per trial, using only good channels
            trial_means = np.nanmean(metric_data[good_channels_idx_for_summary, :], axis=0) if good_channels_idx_for_summary else np.full(n_epochs, np.nan)

        trial_indices_all = np.arange(n_epochs)

        good_tr_mask = np.ones(n_epochs, dtype=bool)
        if list(rejected_trials): # Check if set is not empty
            good_tr_mask[list(rejected_trials)] = False

        plot_elements['trial_scatter_good'] = ax_trial_summary.scatter(
            trial_indices_all[good_tr_mask], trial_means[good_tr_mask],
            c='blue', marker='.', label='Good Tr'
        )
        if np.any(~good_tr_mask): # If there are any bad trials
            plot_elements['trial_scatter_bad'] = ax_trial_summary.scatter(
                trial_indices_all[~good_tr_mask], trial_means[~good_tr_mask],
                c='red', marker='x', label='Bad Tr'
            )
        ax_trial_summary.set_ylabel(f'Mean {current_metric} (good channels)')
        ax_trial_summary.tick_params(axis='x', which='both', bottom=False, labelbottom=False) # X-axis shared
        ax_trial_summary.grid(True, axis='y', linestyle=':')
        ax_trial_summary.set_xlim(ax_summary.get_xlim()) # Ensure x-limits match summary plot

        update_spectrum_plot()

        update_control_text()

        fig.canvas.draw_idle() 

    def update_control_text():

        ax_controls.clear()
        ax_controls.axis('off') 

        rejected_ch_names_list = sorted([ch_names[i] for i in rejected_channels_idx])
        if len(rejected_ch_names_list) > 10:
            ch_text_list = rejected_ch_names_list[:5] + ['...'] + rejected_ch_names_list[-5:]
        else:
            ch_text_list = rejected_ch_names_list
        ch_text = ", ".join(map(str, ch_text_list)) if ch_text_list else "None"

        rejected_trials_list = sorted(list(rejected_trials))
        if len(rejected_trials_list) > 10:
            tr_text_list = [str(t) for t in rejected_trials_list[:5]] + ['...'] + [str(t) for t in rejected_trials_list[-5:]]
        else:
            tr_text_list = [str(t) for t in rejected_trials_list]
        tr_text = ", ".join(tr_text_list) if tr_text_list else "None"

        info_text = (
            f"Metric: {current_metric}\n\n"
            f"Interaction:\n- Click point or drag box on\n  side plots to toggle selection.\n\n"
            f"Rejected Trials: {len(rejected_trials)}/{n_epochs}\n  {tr_text}\n\n"
            f"Rejected Channels: {len(rejected_channels_idx)}/{n_channels}\n  {ch_text}"
        )
        ax_controls.text(0.05, 0.95, info_text, ha='left', va='top', wrap=True, fontsize=9)
        fig.canvas.draw_idle()

    def on_click(event):

        if event.button != 1: return # Process only left-clicks

        toggled = False
        target_chan_idx = -1
        target_trial_idx = -1

        if event.inaxes == ax_chan_summary:
            min_dist_sq = 25 

            scatter_good = plot_elements.get('chan_scatter_good')
            if scatter_good:
                offsets = scatter_good.get_offsets()
                if offsets.size > 0: # Ensure there are points

                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:

                        display_coords = ax_chan_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display)**2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
    
                                target_chan_idx = int(round(valid_offsets[min_idx_local, 1]))

            scatter_bad = plot_elements.get('chan_scatter_bad')
            if scatter_bad:
                offsets = scatter_bad.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_chan_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display)**2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                current_best_dist = distances_sq[min_idx_local]
                                if target_chan_idx == -1 or \
                                   (target_chan_idx != -1 and current_best_dist < np.min(distances_sq)): # Simplified check
                                    target_chan_idx = int(round(valid_offsets[min_idx_local, 1]))


            if 0 <= target_chan_idx < n_channels:
                chan_name = ch_names[target_chan_idx]
                if target_chan_idx in rejected_channels_idx:
                    rejected_channels_idx.remove(target_chan_idx)
                    if chan_name in bad_channel_names: bad_channel_names.remove(chan_name)
                else:
                    rejected_channels_idx.add(target_chan_idx)
                    bad_channel_names.add(chan_name)
                toggled = True

        elif event.inaxes == ax_trial_summary:
            min_dist_sq = 25 
            scatter_good = plot_elements.get('trial_scatter_good')
            if scatter_good:
                offsets = scatter_good.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_trial_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display)**2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                target_trial_idx = int(round(valid_offsets[min_idx_local, 0]))

            scatter_bad = plot_elements.get('trial_scatter_bad')
            if scatter_bad:
                offsets = scatter_bad.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        display_coords = ax_trial_summary.transData.transform(valid_offsets)
                        click_display = (event.x, event.y)
                        distances_sq = np.sum((display_coords - click_display)**2, axis=1)
                        if distances_sq.size > 0:
                            min_idx_local = np.argmin(distances_sq)
                            if distances_sq[min_idx_local] < min_dist_sq:
                                current_best_dist = distances_sq[min_idx_local]
                                if target_trial_idx == -1 or \
                                   (target_trial_idx != -1 and current_best_dist < np.min(distances_sq)):
                                    target_trial_idx = int(round(valid_offsets[min_idx_local, 0]))

            if 0 <= target_trial_idx < n_epochs:
                if target_trial_idx in rejected_trials:
                    rejected_trials.remove(target_trial_idx)
                else:
                    rejected_trials.add(target_trial_idx)
                toggled = True

        if toggled:
            update_plots() 

    def onselect_trials(eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_min_x, selected_max_x = min(x1, x2), max(x1, x2)
        selected_min_y, selected_max_y = min(y1, y2), max(y1, y2)

        toggled_trials_in_selection = set()
        # Check both good and bad trial scatter plots
        for key in ['trial_scatter_good', 'trial_scatter_bad']:
            scatter_plot = plot_elements.get(key)
            if scatter_plot:
                offsets = scatter_plot.get_offsets()
                if offsets.size > 0:
                    # Filter out NaN offsets before checking bounds
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        # Identify points within the selection rectangle
                        mask = ( (valid_offsets[:, 0] >= selected_min_x) &
                                 (valid_offsets[:, 0] <= selected_max_x) &
                                 (valid_offsets[:, 1] >= selected_min_y) &
                                 (valid_offsets[:, 1] <= selected_max_y) )
                        # Add trial indices (x-coordinate) of selected points
                        toggled_trials_in_selection.update(valid_offsets[mask, 0].astype(int))
        if toggled_trials_in_selection:
            for idx in toggled_trials_in_selection:
                if idx in rejected_trials: rejected_trials.remove(idx)
                else: rejected_trials.add(idx)
            update_plots()

    def onselect_channels(eclick, erelease):

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        selected_min_x, selected_max_x = min(x1, x2), max(x1, x2)
        selected_min_y, selected_max_y = min(y1, y2), max(y1, y2)

        toggled_channels_idx_in_selection = set()
        for key in ['chan_scatter_good', 'chan_scatter_bad']:
            scatter_plot = plot_elements.get(key)
            if scatter_plot:
                offsets = scatter_plot.get_offsets()
                if offsets.size > 0:
                    valid_offsets = offsets[~np.isnan(offsets).any(axis=1)]
                    if valid_offsets.size > 0:
                        mask = ( (valid_offsets[:, 0] >= selected_min_x) &
                                 (valid_offsets[:, 0] <= selected_max_x) &
                                 (valid_offsets[:, 1] >= selected_min_y) & # Channel index is y-coord
                                 (valid_offsets[:, 1] <= selected_max_y) )
                        # Add channel indices (y-coordinate)
                        toggled_channels_idx_in_selection.update(valid_offsets[mask, 1].astype(int))

        if toggled_channels_idx_in_selection:
            for idx in toggled_channels_idx_in_selection:
                if 0 <= idx < n_channels: # Ensure valid channel index
                    chan_name = ch_names[idx]
                    if idx in rejected_channels_idx:
                        rejected_channels_idx.remove(idx)
                        if chan_name in bad_channel_names: bad_channel_names.remove(chan_name)
                    else:
                        rejected_channels_idx.add(idx)
                        bad_channel_names.add(chan_name)
            update_plots()

    def on_metric_select(label):
        """Handles selection of a new metric from the radio buttons."""
        nonlocal current_metric # Allow modification of current_metric
        if label != current_metric:
            # Use the helper to recalculate and handle NaN warning state centrally
            temp_metric_data = _calculate_metric_and_warn(data, label)
            if temp_metric_data is not None: # Check if metric calculation was successful
                 current_metric = label
                 update_plots() # Update all plots with the new metric
            else:
                 warnings.warn(f"Cannot switch to invalid or problematic metric: {label}", UserWarning)
                 # Revert radio button to previous metric visually if possible (or just don't update)
                 # For simplicity, we just don't update if the new metric is invalid.
                 # The radio button itself might have already visually changed.

    def on_spectrum_toggle(label):

        nonlocal calculate_spectrum
        new_state = plot_elements['spec_check'].get_status()[0] # Get current status of the first checkbox
        if new_state != calculate_spectrum:
            calculate_spectrum = new_state
            update_spectrum_plot() # Only update the spectrum plot and control text

    def quit_callback(event):

        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)

    valid_metrics = ['var', 'std', 'mad', 'maxabs', 'ptp', 'kurtosis']

    active_metric_idx = valid_metrics.index(current_metric) if current_metric in valid_metrics else valid_metrics.index('var')
    if current_metric not in valid_metrics:
        warnings.warn(f"Initial metric '{current_metric}' is not in valid_metrics. Defaulting to 'var'.", UserWarning)
        current_metric = 'var'

    ax_metric_radio.set_title('Select Metric:', fontsize=12)
    radio = RadioButtons(ax_metric_radio, valid_metrics, active=active_metric_idx)
    for label_widget in radio.labels: label_widget.set_fontsize(12) # Adjust label font size
    radio.on_clicked(on_metric_select)
    plot_elements['metric_radio'] = radio

    spec_check = CheckButtons(ax_spec_check, ['Plot Spectrum'], [calculate_spectrum])
    spec_check.on_clicked(on_spectrum_toggle)
    plot_elements['spec_check'] = spec_check

    quit_button = Button(ax_quit_button, 'Quit & Return Rejected')
    quit_button.on_clicked(quit_callback)
    plot_elements['quit_button'] = quit_button

    selector_props = dict(facecolor='grey', edgecolor='black', alpha=0.3, fill=True)
    selectors['trial'] = RectangleSelector(
        ax_trial_summary, onselect_trials, useblit=True, button=[1], # Left mouse button
        minspanx=5, minspany=5, spancoords='pixels', interactive=True, props=selector_props
    )
    selectors['channel'] = RectangleSelector(
        ax_chan_summary, onselect_channels, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords='pixels', interactive=True, props=selector_props
    )

    def prevent_selector_on_widget_click(event):

        widget_axes = [
            plot_elements['metric_radio'].ax,
            plot_elements['spec_check'].ax,
            plot_elements['quit_button'].ax
        ]
        # For some widgets, we need to check if the event is contained within the widget's drawing area
        if any(ax.contains(event)[0] for ax in widget_axes if ax.contains(event) is not None):
            # If a selector is active, deactivate it
            if 'trial' in selectors and selectors['trial'].active:
                selectors['trial'].set_active(False)
            if 'channel' in selectors and selectors['channel'].active:
                selectors['channel'].set_active(False)


    update_plots() # Initial full draw of all plot elements

    plt.show(block=True) # Blocks execution until the figure is closed

    # --- Return rejected items ---
    final_rejected_trials_indices = sorted(list(rejected_trials))
    final_rejected_channel_names_list = sorted(list(bad_channel_names))

    print(f"Interactive session ended.")
    print(f"Returning {len(final_rejected_trials_indices)} rejected trial indices and "
          f"{len(final_rejected_channel_names_list)} rejected channel names.")
    return final_rejected_trials_indices, final_rejected_channel_names_list

def plot_rejection_with_rejectlog(epochs_original,
                                  rejected_trial_indices,
                                  globally_rejected_channel_names,
                                  log_channel_names):

#   plot_rejection_with_rejectlog(epochs_original, rejected_trial_indices, globally_rejected_channel_names, log_channel_names)
#    Constructs an Autoreject RejectLog from user-selected trial and channel rejections and visualizes both epoch traces with bad segments and a horizontal summary of rejections.
#
#    epochs_original                 MNE Epochs object of the original data.
#    rejected_trial_indices          List of integer indices of rejected epochs.
#    globally_rejected_channel_names List of channel names rejected across all epochs.
#    log_channel_names               List of channel names in the order used for the RejectLog matrix.


    if RejectLog is None:
        print("Autoreject library is not installed. Skipping RejectLog visualization.")
        print("Please install it if you want this feature: pip install autoreject")
        return

    print("\n--- Visualizing with Autoreject RejectLog ---")
    n_total_epochs = len(epochs_original)
    if n_total_epochs == 0:
        print("No epochs in `epochs_original`. Skipping RejectLog visualization.")
        return

    bad_epochs_bool_array = np.zeros(n_total_epochs, dtype=bool)
    if rejected_trial_indices:
        # Ensure indices are valid
        valid_indices = [idx for idx in rejected_trial_indices if 0 <= idx < n_total_epochs]
        if valid_indices:
            bad_epochs_bool_array[valid_indices] = True
        else:
            print("No valid rejected trial indices provided for RejectLog.")

    labels_int_array = np.zeros((n_total_epochs, len(log_channel_names)), dtype=int)

    for epoch_idx in range(n_total_epochs):
        if bad_epochs_bool_array[epoch_idx]: # If this epoch was identified as bad overall
            # Mark specific channels from 'globally_rejected_channel_names' as bad (1) for this epoch
            for ch_name_global_bad in globally_rejected_channel_names:
                if ch_name_global_bad in log_channel_names:
                    # Find the index of this globally bad channel within the log_channel_names list
                    try:
                        ch_idx_in_log = log_channel_names.index(ch_name_global_bad)
                        labels_int_array[epoch_idx, ch_idx_in_log] = 1 # Mark this channel as bad for this epoch
                    except ValueError:
                        # Should not happen if ch_name_global_bad is in log_channel_names
                        pass

    try:
        reject_log_viz = RejectLog(
            bad_epochs=bad_epochs_bool_array,
            labels=labels_int_array,
            ch_names=log_channel_names # Must match the columns of 'labels'
        )
        print("Successfully created RejectLog object.")

        epochs_for_log_traces = epochs_original.copy().pick(log_channel_names, verbose=False)

        if not epochs_for_log_traces.ch_names:
             print(f"Warning: None of the 'log_channel_names' ({log_channel_names}) "
                   f"were found in 'epochs_original'. Skipping RejectLog's plot_epochs.")
        else:
            print("Plotting epoch traces using reject_log_viz.plot_epochs()...")
            fig_traces = reject_log_viz.plot_epochs(
                epochs=epochs_for_log_traces,
                scalings=dict(eeg=60e-6, meg=20e-12, ecog=60e-6, seeg=60e-6), # Generic scalings
                title="Epoch Traces (Bad Selections Highlighted by RejectLog)"
            )


        print("Plotting channel status summary using reject_log_viz.plot()...")
        fig_summary = reject_log_viz.plot(
            orientation='horizontal', # Can be 'vertical'
            show=False # Manage display with plt.show() at the end
        )
        if fig_summary: # reject_log.plot returns a Figure object
            fig_summary.suptitle("Channel-Epoch Rejection Summary (from RejectLog)")
            # fig_summary.show()

        plt.show() # Display all Autoreject plots generated

    except Exception as e:
        print(f"An error occurred during RejectLog processing or plotting: {e}")
        import traceback
        traceback.print_exc()

def plot_mne_native_with_bads(epochs_original,
                              globally_rejected_channel_names,
                              rejected_trial_indices_for_context=None,
                              num_epochs_to_show=5):

#   plot_mne_native_with_bads(epochs_original, globally_rejected_channel_names, rejected_trial_indices_for_context, num_epochs_to_show)
#    Displays a subset of epochs using MNE’s native plotting with channels flagged as bad in info['bads'], prioritizing user-rejected trials if provided.
#
#    epochs_original                  MNE Epochs object containing original data.
#    globally_rejected_channel_names  List of channel names to mark as bad in the plot.
#    rejected_trial_indices_for_context Optional list of epoch indices to prioritize in the display.
#    num_epochs_to_show               Integer specifying the maximum number of epochs to render.

    print("\n--- MNE Native Plotting Demonstration with info['bads'] ---")
    if not globally_rejected_channel_names:
        print("No globally rejected channel names provided. Plotting without highlighting specific bad channels.")
        # Proceed to plot, but info['bads'] will remain unchanged or empty.

    epochs_for_mne_plot = epochs_original.copy() # Work on a copy

    original_bads = list(epochs_for_mne_plot.info['bads']) # Store original bads
    newly_added_bads = []
    for ch_name in globally_rejected_channel_names:
        if ch_name in epochs_for_mne_plot.ch_names:
            if ch_name not in epochs_for_mne_plot.info['bads']:
                epochs_for_mne_plot.info['bads'].append(ch_name)
                newly_added_bads.append(ch_name)
        else:
            print(f"Warning: Channel '{ch_name}' specified in globally_rejected_channel_names "
                  "not found in epoch channel names. Cannot mark as bad.")

    if newly_added_bads:
        print(f"Channels added to info['bads'] for this plot: {newly_added_bads}")
    elif not globally_rejected_channel_names:
        pass # Already handled by the print message above
    else: # globally_rejected_channel_names was provided, but none were new or valid
        print("No new channels were added to info['bads'] for MNE native plotting (either already bad or not found).")

    indices_to_plot = []
    if rejected_trial_indices_for_context:
        # Add some of the rejected trials (if any, and within bounds of total epochs)
        indices_to_plot.extend(
            idx for idx in rejected_trial_indices_for_context if 0 <= idx < len(epochs_original)
        )

    current_epoch_idx = 0
    while len(set(indices_to_plot)) < num_epochs_to_show and current_epoch_idx < len(epochs_original):
        if current_epoch_idx not in indices_to_plot:
            indices_to_plot.append(current_epoch_idx)
        current_epoch_idx += 1

    indices_to_plot = sorted(list(set(idx for idx in indices_to_plot if 0 <= idx < len(epochs_original))))
    indices_to_plot = indices_to_plot[:num_epochs_to_show]

    if not indices_to_plot and len(epochs_original) > 0: # Fallback if list is empty but epochs exist
        indices_to_plot = list(range(min(len(epochs_original), num_epochs_to_show)))

    if not indices_to_plot: # If still no epochs to plot (e.g., epochs_original was empty)
        print("No epochs available or selected for MNE native plotting.")
        epochs_for_mne_plot.info['bads'] = original_bads # Restore original bads
        return

    epochs_subset_for_mne_plot = epochs_for_mne_plot[indices_to_plot]

    print(f"Plotting MNE epochs (indices: {indices_to_plot}) "
          f"with info['bads']: {epochs_subset_for_mne_plot.info['bads']}")

    fig_mne_native = epochs_subset_for_mne_plot.plot(
        bad_color='salmon', # Color for channels listed in info['bads']
        title=f"MNE Native Plot (Epochs: {indices_to_plot}, info['bads'] marked)",
        n_epochs=len(epochs_subset_for_mne_plot), # Number of epochs to display
        show=True, # Show the plot immediately
        block=True # Block execution until plot is closed
    )

    epochs_for_mne_plot.info['bads'] = original_bads
    
SyntaxError: invalid syntax
