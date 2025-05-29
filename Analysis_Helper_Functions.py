Python 3.13.0 (v3.13.0:60403a5409f, Oct  7 2024, 00:37:40) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
import pandas as pd
import mne
import nibabel as nib
from nilearn import plotting
import scipy.stats as sp_stats
from scipy.stats import median_abs_deviation as mad
import scipy.signal as sp_signal
from scipy.signal import firwin, filtfilt, hilbert
from scipy.ndimage import label
from nitime import algorithms as tsa
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import Counter

def circ_corr_fun(Y, X):

#   R2 = circ_corr_fun(Y, X)
#    Computes the trial‐wise linear–circular correlation between multi‐channel time‐resolved signals and a circular predictor.
#
#    Y              NumPy array of shape (n_trials, n_channels, n_timepoints) containing the recorded signals.
#    X              One‐dimensional NumPy array of length n_trials giving the stimulus orientations in radians.
#    R2             NumPy array of shape (n_channels, n_timepoints) containing the coefficient of determination for each channel and timepoint.

    # Y is a matrix with trials x channels x timepoints
    # X is a vector with the orientation of each stimulus in radians
    n_trial, n_chan, n_time = Y.shape
    Yreshape = Y.reshape([-1, n_chan * n_time])
    # linear circular correlation between MEG and stim angle
    _, R2, _ = corr_linear_circular(Yreshape, X)
    R2 = R2.reshape([n_chan, n_time])
    return R2

def get_freq_spec(data, fs, batch_size=100):  # Now uses batch_size parameter properly

#   psd, f, nu = get_freq_spec(data, fs, batch_size)
#    Estimates the multitaper power spectral density for each trial and channel in batches.
#
#    data           NumPy array of shape (n_trials, n_channels, n_timepoints) containing the time series.
#    fs             Sampling frequency in Hz.
#    batch_size     Integer number of trials to process per block to limit memory usage.
#    psd            NumPy array of shape (n_trials, n_channels, n_frequencies) containing the estimated PSD.
#    f              One‐dimensional NumPy array of frequency values corresponding to the PSD.
#    nu             Degrees of freedom associated with the multitaper estimate.

    ntrials = data.shape[0]
    psd_mt_list = []
    
    for i in np.arange(0, ntrials, batch_size):
        data_slice = data[i:i+batch_size, :, :]
        print(f'Processing batch starting at trial {i}')
        f, psd_sl, nu = tsa.multi_taper_psd(
            data_slice, Fs=fs, adaptive=False, jackknife=False
        )
        psd_mt_list.append(psd_sl)
    
    psd = np.concatenate(psd_mt_list, axis=0)
    return psd, f, nu

def norm_freq(psd, f, fband):

#   norm_vals = norm_freq(psd, f, fband)
#    Computes the log‐ratio of high‐frequency-band power to total power for each trial and channel.
#
#    psd            NumPy array of shape (n_trials, n_channels, n_frequencies) containing power spectral densities.
#    f              One‐dimensional NumPy array of length n_frequencies with frequency bins.
#    fband          Tuple (low, high) in Hz defining the high‐frequency band.
#    norm_vals      NumPy array of shape (n_trials, n_channels) with log10‐transformed ratio of band power to total power.

    low, high = fband
    idx_HG = np.logical_and(f >= low, f <= high)
    idx_tot = np.logical_and(f >= 5, f <= 125)

    HG_power = np.sum(psd[:, :, idx_HG], axis=-1)
    total_power = np.sum(psd[:, :, idx_tot], axis=-1)

    # Prevent division errors
    epsilon = 1e-10
    return np.log10((HG_power + epsilon) / (total_power + epsilon))

def compute_null_psd(epochs, angles_rad, freq_band, get_freq_spec, norm_freq, nperm=200, batch_size=100):

#   R_true, R_null = compute_null_psd(epochs, angles_rad, freq_band, get_freq_spec, norm_freq, nperm, batch_size)
#    Calculates the true and permutation‐based null distributions of circular correlation between normalized PSD and stimulus angle.
#
#    epochs         MNE Epochs object containing segmented data of shape (n_trials, n_channels, n_timepoints).
#    angles_rad     One‐dimensional NumPy array of trial‐wise stimulus orientations in radians.
#    freq_band      Tuple (low, high) in Hz defining the frequency band for normalization.
#    get_freq_spec  Function to compute PSD (accepts data array, sampling rate, batch_size).
#    norm_freq      Function to normalize PSD to band ratio.
#    nperm          Integer number of permutations for null distribution.
#    batch_size     Integer number of trials per batch in PSD estimation.
#    R_true         One‐dimensional NumPy array of length n_channels containing the observed correlation per channel.
#    R_null         NumPy array of shape (nperm, n_channels) containing correlations under permuted labels.

    X_time = epochs.get_data()  

    psd_mt, freqs, nu = get_freq_spec(X_time, epochs.info['sfreq'], batch_size=batch_size)

    X_rel = norm_freq(psd_mt, freqs, fband=freq_band)

    X_corr = X_rel[..., np.newaxis]

    R2_true = circ_corr_fun(X_corr, angles_rad)  # returns (channels, 1)
    R_true = R2_true[:, 0]

    nch = len(epochs.ch_names)
    R_null = np.zeros((nperm, nch), dtype=float)
    for i in range(nperm):
        y_perm = np.random.permutation(angles_rad)
        R2p = circ_corr_fun(X_corr, y_perm)
        R_null[i, :] = R2p[:, 0]

    return R_true, R_null

def channel_significance_by_window_psd(epochs_dict,
                                       electrode_df,
                                       time_windows,
                                       freq_band=(2,12),
                                       alpha=0.05,
                                       nperm=200,
                                       batch_size=100):

#   signif_df, merged, region_counts = channel_significance_by_window_psd(epochs_dict, electrode_df, time_windows, freq_band, alpha, nperm, batch_size)
#    Tests each channel’s PSD–angle correlation within multiple latency windows against a permutation null, then annotates significant channels by anatomical region.
#
#    epochs_dict    Dictionary mapping participant identifiers to MNE Epochs objects with metadata field 'T_Angle'.
#    electrode_df   pandas DataFrame with columns 'participant', 'Bipolar_Label', and 'FSLabel' for anatomical mapping.
#    time_windows   Dictionary mapping window label to (tmin, tmax) in seconds.
#    freq_band      Tuple (low, high) in Hz defining the normalization band.
#    alpha          Significance threshold for p‐values.
#    nperm          Number of permutations for null distributions.
#    batch_size     Number of trials per batch in PSD estimation.
#    signif_df      pandas DataFrame of raw significant tests with columns 'participant', 'channel', 'window', 'pval'.
#    merged         pandas DataFrame of significant tests merged with anatomical regions.
#    region_counts  pandas DataFrame counting significant occurrences per region, sorted descending.

    records = []
    for subj, ep in epochs_dict.items():
        angles = np.deg2rad(ep.metadata['T_Angle'].values)
        for label, (tmin, tmax) in time_windows.items():
            ep_win = ep.copy().crop(tmin, tmax)

            R_true, R_null = compute_null_psd(
                ep_win, angles,
                freq_band=freq_band,
                get_freq_spec=get_freq_spec,
                norm_freq=norm_freq,
                nperm=nperm,
                batch_size=batch_size
            )

            pvals = np.mean(R_true[np.newaxis, :] < R_null, axis=0)

            for idx, ch in enumerate(ep.ch_names):
                if pvals[idx] < alpha:
                    records.append({
                        'participant': subj,
                        'channel':     ch,
                        'window':      label,
                        'pval':        float(pvals[idx])
                    })

    signif_df = pd.DataFrame.from_records(records)

    merged = signif_df.merge(
        electrode_df[['participant', 'Bipolar_Label', 'FSLabel']],
        left_on=['participant', 'channel'],
        right_on=['participant', 'Bipolar_Label'],
        how='left'
    )
    merged['region'] = (
        merged['FSLabel']
              .str.replace(r'^Mixed:', '', regex=True)
              .str.split(r'\|')
    ).explode('region')

    bad = merged['region'].str.contains('Unknown|White-Matter|WM', na=False)
    merged = merged.loc[~bad, ['participant', 'channel', 'window', 'pval', 'region']]

    region_counts = (
        merged.groupby('region')
              .size()
              .reset_index(name='count')
              .sort_values('count', ascending=False)
              .reset_index(drop=True)
    )

    return signif_df, merged, region_counts

def plot_period_regions(top_regions_df, period_name, label_names_dict, cmap='Reds'):
#   plot_period_regions(top_regions_df, period_name, label_names_dict, cmap)
#    Visualizes the top anatomical labels as a binary mask on a glass‐brain projection.
#
#    top_regions_df DataFrame containing at least a column 'Count' listing region label names.
#    period_name    String describing the epoch or period being plotted.
#    label_names_dict Dictionary mapping integer label IDs to anatomical label strings.
#    cmap           Matplotlib colormap name for rendering active regions.
#    (No return; displays a glass‐brain figure.)  
    label_ids = []
    for region in top_regions_df['Count']:
        for label_id, label_name in label_names.items():
            if region in label_name:
                label_ids.append(label_id)
    
    
    roi_data = (np.isin(atlas_data, label_ids)).astype(np.uint8)
    
    roi_img = nib.MGHImage(roi_data, img.affine, header=img.header)
    
    fig = plt.figure(figsize=(6, 4))
    plotting.plot_glass_brain(roi_img, 
                            title=f'Active regions: {period_name}',
                            cmap=cmap, 
                            colorbar=True,
                            threshold=0.5, 
                            display_mode='lyrz',
                            figure=fig)
    plt.show()

    
from collections import Counter

def top_regions(df_locs, top_n=5):

#   region_counts = top_regions(df_locs, top_n)
#    Identifies the most frequent regions in a table of significant channel locations.
#
#    df_locs        pandas DataFrame with a 'region' column listing anatomical labels.
#    top_n          Integer number of top regions to return.
#    region_counts  pandas DataFrame with columns 'Region' and 'Count' for the top_n labels.


    filtered = df_locs[df_locs['region'] != 'Unknown']
    
    #filtered['region'] = filtered['region'].str.replace(r'^ctx-(lh|rh)-', '', regex=True)
    
    region_counts = filtered['region'].value_counts().nlargest(top_n)
    return region_counts.reset_index().rename(columns={'index': 'Region', 'region': 'Count'})
SyntaxError: multiple statements found while compiling a single statement
def plot_period_heatmap(top_regions_df, period_name, label_names_dict, atlas_data, img):

#   plot_period_heatmap(top_regions_df, period_name, label_names_dict, atlas_data, img)
#    Renders a heatmap of region counts on a glass‐brain using continuous intensity.
#
#    top_regions_df DataFrame with columns 'count' and region label strings in index or column.
#    period_name    String describing the epoch or period.
#    label_names_dict Dictionary mapping integer label IDs to anatomical label strings.
#    atlas_data     NumPy array of same shape as img data containing integer label IDs.
#    img            nibabel image object providing affine and header.
#    (No return; displays a glass‐brain figure.)

    roi_data = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in top_regions_df.iterrows():
        region_name = row['Count']
        region_count = row['count']

        for label_id, label_name in label_names_dict.items():
            if region_name in label_name:
                roi_data[atlas_data == label_id] = region_count

    roi_img = nib.MGHImage(roi_data, img.affine, header=img.header)

    fig = plt.figure(figsize=(8, 5))
    plotting.plot_glass_brain(
        roi_img,
        #title=f'{period_name} Activity Heatmap (raw counts)',
        cmap='viridis',
        colorbar=True,
        vmin=0,
        vmax=roi_data.max(),
        display_mode='lyrz',
         figure=fig
     )
     plt.show()
 
     
def channel_significance_by_window_psd_preT(epochs_dict, electrode_df, time_windows,freq_band=(2,12),alpha=0.05,nperm=200,batch_size=100):

#   signif_df, merged, region_counts = channel_significance_by_window_psd_preT(epochs_dict, electrode_df, time_windows, freq_band, alpha, nperm, batch_size)
#    Performs PSD–behavior correlation tests using a pre-stimulus metadata field, filtering invalid trials before permutation analysis.
#
#    epochs_dict    Dictionary mapping participant IDs to MNE Epochs objects containing a metadata column 'preT'.
#    electrode_df   pandas DataFrame with anatomical labels for channels.
#    time_windows   Dictionary mapping window label to (tmin, tmax) in seconds.
#    freq_band      Tuple (low, high) in Hz defining the normalization band.
#    alpha          Significance threshold.
#    nperm          Number of permutations.
#    batch_size     Batch size for PSD estimation.
#    signif_df      DataFrame of raw significant tests for preT correlations.
#    merged         DataFrame of significant tests merged with anatomical regions.
#    region_counts  DataFrame counting significant occurrences per region.

    records = []
    for subj, ep in epochs_dict.items():
        for label, (tmin, tmax) in time_windows.items():
            ep_win = ep.copy().crop(tmin, tmax)

            angles = ep_win.metadata['preT'].values
            valid_trials = ~np.isnan(angles)
            angles_valid = np.deg2rad(angles[valid_trials])
            ep_win_valid = ep_win[valid_trials]  
             
            if len(angles_valid) == 0:
                continue  
                 
            R_true, R_null = compute_null_psd(
                ep_win_valid, angles_valid, 
                freq_band=freq_band,
                get_freq_spec=get_freq_spec,
                norm_freq=norm_freq,
                nperm=nperm,
                batch_size=batch_size
            )
 
            pvals = np.mean(R_true[np.newaxis, :] < R_null, axis=0)
 
            for idx, ch in enumerate(ep_win_valid.ch_names): 
                if pvals[idx] < alpha:
                    records.append({
                        'participant': subj,
                        'channel':     ch,
                        'window':      label,
                        'pval':        float(pvals[idx])
                    })

    signif_df = pd.DataFrame.from_records(records)

    merged = signif_df.merge(
        electrode_df[['participant', 'Bipolar_Label', 'FSLabel']],
        left_on=['participant', 'channel'],
        right_on=['participant', 'Bipolar_Label'],
        how='left'
    )
    merged['region'] = (
        merged['FSLabel']
              .str.replace(r'^Mixed:', '', regex=True)
              .str.split(r'\|')
    ).explode('region')

    bad = merged['region'].str.contains('Unknown|White-Matter|WM', na=False)
    merged = merged.loc[~bad, ['participant', 'channel', 'window', 'pval', 'region']]

    region_counts = (
        merged.groupby('region')
              .size()
              .reset_index(name='count')
              .sort_values('count', ascending=False)
              .reset_index(drop=True)
    )

    return signif_df, merged, region_counts

def decode_with_psd_permutation(epochs, angles_deg, freq_band=(70,150),
                                nfold=5, nperm=200, random_state=42):
#   real_error, perm_errors, p_value = decode_with_psd_permutation(epochs, angles_deg, freq_band, nfold, nperm, random_state)
#    Performs circular decoding of stimulus orientation from normalized PSD features with cross‐validation and permutation testing.
#
#    epochs         MNE Epochs object providing the raw time series for PSD extraction.
#    angles_deg     One‐dimensional array of stimulus orientations in degrees.
#    freq_band      Tuple (low, high) in Hz defining the normalization band.
#    nfold          Number of cross‐validation folds.
#    nperm          Number of label permutations for significance testing.
#    random_state   Integer seed for reproducibility of splits and permutations.
#    real_error     Float mean absolute angular error in degrees for true labels.
#    perm_errors    One‐dimensional array of length nperm with errors under permuted labels.
#    p_value        Float proportion of permuted errors less than or equal to the real_error.
    X_raw = epochs.get_data()  # n_trials, n_ch, n_time
    psd_mt, freqs, _ = get_freq_spec(X_raw, epochs.info['sfreq'])
    X_rel = norm_freq(psd_mt, freqs, fband=freq_band)  # (n_trials, n_ch)

    angles_rad = np.deg2rad(angles_deg)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
    
    def one_decoding(X, y_rad):
        errs = []
        for train_ix, test_ix in kf.split(X):
            scaler = StandardScaler().fit(X[train_ix]) # Computes mean & std of each channel on the training set.
            X_train = scaler.transform(X[train_ix]) # Ensures each channel’s feature has zero mean & unit variance—so no single channel dominates.
            X_test  = scaler.transform(X[test_ix])
            
            clf = make_pipeline(AngularRegression(clf=LinearSVR())) # A linear support‐vector regression. Wraps regressor to handle circular targets correctly
            clf.fit(X_train, y_rad[train_ix]) # Learns a weight per channel
            preds_rad = clf.predict(X_test)
            
            err_rad = np.angle(np.exp(1j*(preds_rad - y_rad[test_ix])))

            errs.append(np.mean(np.abs(np.rad2deg(err_rad))))
        return np.mean(errs)
    

    real_error = one_decoding(X_rel, angles_rad)
    

    perm_errors = np.zeros(nperm)
    rng = np.random.RandomState(random_state)
    for i in range(nperm):
        y_perm = rng.permutation(angles_rad)
        perm_errors[i] = one_decoding(X_rel, y_perm)

    p_value = np.mean(perm_errors <= real_error)
    
    return real_error, perm_errors, p_value

def shrinkage_gamma(X, mem_eff=False, feedback=True):

#   gamma = shrinkage_gamma(X, mem_eff, feedback)
#    Computes the optimal shrinkage intensity for the sample covariance of a data matrix according to the Ledoit–Wolf formulation, with an option for a memory‐efficient algorithm.
#
#    X            Two‐dimensional NumPy array of shape (n_features, n_samples) containing the data whose covariance is to be regularized.
#    mem_eff      Boolean flag; if False, uses a full 3D computation of outer products (higher memory cost); if True, computes variances and covariances iteratively to reduce memory usage.
#    feedback     Boolean flag that, when True and mem_eff=True, prints progress updates during the off‐diagonal element loop.
#    gamma        Float in [0,1] representing the shrinkage coefficient to blend the sample covariance toward its diagonal (variance) target.

    num_f, num_n = X.shape
    
    if not mem_eff:

        m = np.mean(X, axis=1, keepdims=True)
        S = np.cov(X, rowvar=True)
        nu = np.trace(S) / num_f

        z = np.zeros((num_f, num_f, num_n))
        for n in range(num_n):
            x_centered = X[:, n:n+1] - m
            z[:, :, n] = x_centered @ x_centered.T

        numerator = (num_n / ((num_n - 1) ** 2)) * np.sum(np.var(z, axis=2))
        denominator = np.sum((S - nu * np.eye(num_f)) ** 2)
        gamma = numerator / denominator
    
    else:

        X = X - np.mean(X, axis=1, keepdims=True)

        sum_var_diag = 0
        diag_s = np.zeros(num_f)
        
        for i_f in range(num_f):
            s = X[i_f, :] ** 2
            diag_s[i_f] = np.sum(s) / (num_n - 1)
            s_centered = s - np.mean(s)
            sum_var_diag += np.sum(s_centered ** 2) / (num_n - 1)
        
        nu = np.mean(diag_s)
        diag_s = diag_s - nu
        sum_s_diag = np.sum(diag_s ** 2)
        
        sum_s = 0
        sum_var = 0
        
        total_elements = (num_f - 1) * num_f // 2
        if feedback:
            print(f"Processing {total_elements} off-diagonal elements...")
        
        counter = 0
        for i_f1 in range(1, num_f):
            for i_f2 in range(i_f1):
                s = X[i_f1, :] * X[i_f2, :]
                sum_s += (np.sum(s) / (num_n - 1)) ** 2
                s_centered = s - np.mean(s)
                sum_var += np.sum(s_centered ** 2) / (num_n - 1)
                
                counter += 1
                if feedback and counter % 1000 == 0:
                    p_done = counter / total_elements
                    print(f"Progress: {p_done*100:.2f}% complete")
        
        sum_s = sum_s * 2 + sum_s_diag
        sum_var = sum_var * 2 + sum_var_diag
        gamma = (num_n / ((num_n - 1) ** 2)) * sum_var / sum_s
    
    return gamma
