
import numpy as np
import scipy.optimize as sci_opt
import re
import os
import h5py

def _multi_lorentz(f, params, dip_number=2):
    np_params = np.array(params)
    f0 = np_params[0:dip_number]
    widths = np_params[dip_number: 2 * dip_number]
    amps = np_params[2 * dip_number: 3 * dip_number]
    intensity = np_params[-1]

    lorentz_line = (amps[:, None] *
                    np.square(widths[:, None]) /
                    (np.square(f - f0[:, None]) + np.square(widths[:, None])))
    lorentz_line = intensity * (1 - np.sum(lorentz_line, axis=0))
    return lorentz_line


def odmr_fit(freqs, odmr_data, dip_number=2, freq_guess=None, amp_guess=None,
             linewid_guess=None, bounds=None, **kwargs):
    """
    Fits an arbitrary number of odmr dips. If freq_guess is not specified it will do an automatic dip search based on
    scipy.signal.find_peaks.
    Order of the data is:
        [freq_1,...,freq_N, width_1,...,width_N ,amp_1,...,amp_N, max_intensity]
    Optional kwargs are:
        - smoothed data: an array of smoothed data, to help improve peak finding
        - index_parameter: value which is printed in the error message when the fit fails
                            (useful when the fit is in a loop)
        - show_guess: return points of frequency guess
        - all the keyword arguments of scipy.signal.find_peaks
        - gtol of scipy.optimize.curve_fit
        - max_nfev of gtol of scipy.optimize.curve_fit

    Returns:
        - optimal parameters
        - covariance matrix
        - an array of the fitted data
        - fail fit flag
        - (if show_guess == True) the guessed frequencies


    If the fitting fails, it returns a NaN
    """
    smoothed_data = kwargs.get('smoothed_data', None)
    maxfev = kwargs.get('maxfev', 200)
    gtol = kwargs.get('gtol', 1e-8)
    index_parameter = kwargs.get('index_parameter', None)
    show_guess = kwargs.get('show_guess', False)

    fail_fit_flag = 0
    if smoothed_data is not None:
        data_4_guess = smoothed_data
    else:
        data_4_guess = odmr_data

    if amp_guess is None:
        amp_guess = 0.1 * np.ones((dip_number,))
    if linewid_guess is None:
        linewid_guess = 1e6 * np.ones((dip_number,))

    fit_func = lambda x, *pars: _multi_lorentz(x, pars, dip_number=dip_number)

    init_guesses = np.concatenate((freq_guess, linewid_guess, amp_guess, [odmr_data.mean()]))

    if bounds is None:
        bounds = (0, np.inf * np.ones((len(init_guesses, ))))
    try:
        opt_pars, cov_mat = sci_opt.curve_fit(fit_func, freqs, odmr_data, p0=init_guesses,
                                              bounds=bounds, maxfev=maxfev,
                                              gtol=gtol)
        fitted_data = _multi_lorentz(freqs, opt_pars, dip_number=dip_number)
    except:
        opt_pars = np.repeat(np.nan, len(init_guesses))
        cov_mat = np.full((len(init_guesses), len(init_guesses)), np.nan)
        fitted_data = np.repeat(np.nan, len(freqs))
        fail_fit_flag = 1
        print("Failed: fit did not converge. Position is: {}".format(index_parameter))

    if show_guess == True:
        return opt_pars, cov_mat, fitted_data, fail_fit_flag, freq_guess
    else:
        return opt_pars, cov_mat, fitted_data, fail_fit_flag


def check_neighbour_pixel(input_matrix, index_tuple, neighbours=1, empty_value=np.nan):
    """
    Takes a (M,N) matrix and check the n nearest neighbours at the index tuple.
    Returns their average value.
    Inputs:
        - the (M,N) numpy array
        - index_tuple: index tuple
        - neighbours: how many nearest neighbours to check
        - empty value: for incomplete scans. Tells the function which value
                       corresponds to an empty pixel
    Returns:
        the average of the n nearest neighbours of index_tuple.
    """
    size_x, size_y = input_matrix.shape
    ii, jj = index_tuple

    xlowbound, xhighbound = (ii - neighbours if (ii - neighbours >= 0) else 0,
                             ii + neighbours + 1 if (ii + neighbours + 1 < size_x) else size_x)
    ylowbound, yhighbound = (jj - neighbours if (jj - neighbours >= 0) else 0,
                             jj + neighbours + 1 if (jj + neighbours + 1 < size_y) else size_y)

    sub_mat = np.array(input_matrix[xlowbound:xhighbound, ylowbound:yhighbound])
    if ~np.isnan(empty_value):
        sub_mat[sub_mat == empty_value] = np.nan
    return np.nanmean(sub_mat)


def arrange_folder_with_names(file_list, patterns, dict_keys=None):
    """
    Legacy function.
    Loads a list of filenames, finds a single number (i.e. a the distance from the surface)
    and saves the strings containing the same number under the same key of a dictionary.
    """

    if dict_keys is None:
        keys = np.arange(0, len(patterns))
        dictionary = dict.fromkeys(keys, None)
    else:
        dictionary = dict.fromkeys(dict_keys, None)

    def match_string(string, pattern):
        # works only for single match (i.e. match number in string)
        matched = re.match(pattern, string)
        if matched is not None:
            matched_array = [matched_group for matched_group in matched.groups()]
            return np.array(matched_array)
        else:
            pass

    dict_keys = list(dictionary.keys())
    for index, pattern in enumerate(patterns):
        values = np.array([match_string(string, pattern)
                           for string in file_list if match_string(string, pattern) is not None])
        dictionary[dict_keys[index]] = values

    return dictionary


def load_magscan(direc):
    '''

    Load data from magscan file with .hdf5 or .txt format. hdf5 takes the absolute file path, .txt takes the folder path
    containing the txt files.
    Output: x_axis, y_axis, X, Y, freqs, fullESR_data, meta, topography_data, retrace_topo

    '''

    if direc.endswith('.hdf5'):
        print('0')
        file = h5py.File(direc, 'r')
        print('1')
        x_axis = np.array(file['magscan x axis'])  # in nm
        print('2')
        y_axis = np.array(file['magscan y axis'])  # in nm
        print('3')
        freqs = np.array(file['Frequencies']) * 1e9  # in Hz
        print('4')
        fulldata_matrix = file['ESR Data']
        print('5')
        retrace = file['ESR Data Retrace']
        print('6')
        meta = file.attrs.get('Tip No.')#.decode('ascii')
        print('7')
        meta += '\n'
        meta += file.attrs.get('Comments')#.decode('ascii')
        meta += '\n'
        meta += file.attrs.get('Sample Details')#.decode('ascii')
        fullESR_data = np.transpose(fulldata_matrix[:, :, file.attrs['Spec Averages'], :, 0], (1, 0, 2))
        topography_data = fulldata_matrix[:, :, 0, 0, 1].T / (-0.0014285714285714286)
        retrace_topo = retrace[:, :, 0, 0, 1].T / (-0.0014285714285714286)
        if x_axis.size == 1:
            x_axis = x_axis.reshape((1,))
        if y_axis.size == 1:
            y_axis = y_axis.reshape((1,))
        X, Y = np.meshgrid(x_axis - x_axis[0], y_axis - y_axis[0])



    else:
        x_filename = "magscan_x_axis.txt"
        y_filename = "magscan_y_axis.txt"
        topography_filename = "magscan_zout.txt"
        retrace_topo_filename = "magscan_retrace_zout.txt"
        meta_filename = "details.txt"
        x_axis, y_axis = np.loadtxt(direc + "\\" + x_filename), np.loadtxt(direc + "\\" + y_filename)
        if x_axis.size == 1:
            x_axis = x_axis.reshape((1,))
        if y_axis.size == 1:
            y_axis = y_axis.reshape((1,))
        X, Y = np.meshgrid(x_axis - x_axis[0], y_axis - y_axis[0])
        topography_data = np.loadtxt(direc + "\\" + topography_filename) / 0.00077206 * (-1)
        retrace_topo = np.loadtxt(direc + "\\" + retrace_topo_filename) / 0.00077206 * (-1)
        file = open(direc + "\\" + meta_filename)
        meta = file.read().replace("\n", " ")
        file.close()
        # meta = np.genfromtxt(direc + "\\" + meta_filename, dtype='str', mising_values = )
        files = os.listdir(direc)
        pattern = "magscan_f(\d+)_(\d*\.\d*)_counts_avg.txt"
        extracted_numbers = arrange_folder_with_names(files, ["magscan_f(\d+)_(\d*\.\d*)_counts_avg.txt"])[0]
        extracted_numbers = extracted_numbers[extracted_numbers[:, 0].astype(int).argsort(), :]
        freqs = extracted_numbers[:, 1].astype(float) * 1e9
        fullESR_data = np.zeros(X.shape + (len(extracted_numbers[:, 0]),))
        for index in range(len(extracted_numbers[:, 0])):
            loaded = np.loadtxt(direc + "\\" + "magscan_f{}_{}_counts_avg.txt"
                                .format(extracted_numbers[index, 0], extracted_numbers[index, 1]))
            if len(loaded.shape) == 1:
                loaded = loaded.reshape((1, loaded.shape[0]))
            # loaded = loaded[ :nan_index_row, :nan_index_col ]
            ### Remove up_to_row from loaded[:up_to_row,:]
            ### if you have the complete data (matrix does not contain NaN)
            fullESR_data[:, :, index] = loaded[:, :]

    return x_axis, y_axis, X, Y, freqs, fullESR_data, meta, topography_data, retrace_topo

def fixed_double_lorentz(x,*params):
    x0, split, a, gam0 , b, gam1, intensity = params
    return intensity*(1-(a * gam0**2 / ( gam0**2 + ( x - x0 )**2) + b * gam1**2 / ( gam1**2 + ( x - (x0+split ))**2)))


def nucl_odmr_fit(freqs, odmr_data, freq_guess=None, split_guess=None, amp1_guess=None,
                  amp2_guess=None, linewid1_guess=None, linewid2_guess=None, bounds=None, **kwargs):
    """
    Fits a single nuclear spin split resonance. Freq_guess is the lower frequency dip.
    Order of the data is:
        [freq, split, amp_1, amp_2, width_1, width_2, max_intensity]
    Optional kwargs are:
        - smoothed data: an array of smoothed data, to help improve peak finding
        - index_parameter: value which is printed in the error message when the fit fails
                            (useful when the fit is in a loop)
        - show_guess: return points of frequency guess
        - all the keyword arguments of scipy.signal.find_peaks
        - gtol of scipy.optimize.curve_fit
        - max_nfev of gtol of scipy.optimize.curve_fit

    Returns:
        - optimal parameters
        - covariance matrix
        - an array of the fitted data
        - fail fit flag
        - (if show_guess == True) the guessed frequencies


    If the fitting fails, it returns a NaN
    """

    smoothed_data = kwargs.get('smoothed_data', None)
    maxfev = kwargs.get('maxfev', 200)
    gtol = kwargs.get('gtol', 1e-8)
    index_parameter = kwargs.get('index_parameter', None)
    show_guess = kwargs.get('show_guess', False)

    fail_fit_flag = 0
    if smoothed_data is not None:
        data_4_guess = smoothed_data
    else:
        data_4_guess = odmr_data

    fit_func = lambda x, *params: fixed_double_lorentz(x, *params)

    init_guesses = [freq_guess, split_guess, amp1_guess, linewid1_guess, amp2_guess, linewid2_guess, odmr_data.mean()]


    try:
        opt_pars, cov_mat = sci_opt.curve_fit(fit_func, freqs, odmr_data, p0=init_guesses,
                                              bounds=bounds, maxfev=maxfev,
                                              gtol=gtol)
        fitted_data = fixed_double_lorentz(freqs, *opt_pars)
    except:
        opt_pars = np.repeat(np.nan, len(init_guesses))
        cov_mat = np.full((len(init_guesses), len(init_guesses)), np.nan)
        fitted_data = np.repeat(np.nan, len(freqs))
        fail_fit_flag = 1
        #print("Failed: fit did not converge. Position is: {}".format(index_parameter))

    if show_guess == True:
        return opt_pars, cov_mat, fitted_data, fail_fit_flag, freq_guess
    else:
        return opt_pars, cov_mat, fitted_data, fail_fit_flag
    

def norm_m1_1(data):
    return (2*(data-np.min(data))/(np.max(data)-np.min(data)))-1

def norm01(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def mean_norm(data):
    return data/np.mean(data)