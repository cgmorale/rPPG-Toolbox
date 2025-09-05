import numpy as np
import pandas as pd
import torch
import os
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def extract_participant_id_from_index(index):
    """Extract participant ID from video index for supervised models."""
    # Handle different naming conventions
    import re
    
    # For UBFC-rPPG: subject1, subject01, etc.
    if 'subject' in index.lower():
        match = re.search(r'subject(\d+)', index.lower())
        if match:
            return f"subject{int(match.group(1))}"  # Normalize to subject1, subject2, etc.
    
    # For PURE: might be different format
    # Add more patterns as needed for other datasets
    
    # Fallback: use the index as-is
    return str(index)


def calculate_metrics_for_participant_supervised(predictions, ground_truth, snr_values, macc_values, config):
    """Calculate all metrics for a single participant in supervised model context"""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    snr_values = np.array(snr_values)
    macc_values = np.array(macc_values)
    
    if len(predictions) == 0:
        return {}
    
    num_samples = len(predictions)
    metrics = {}
    
    for metric in config.TEST.METRICS:
        if metric == "MAE":
            mae = np.mean(np.abs(predictions - ground_truth))
            std_error = np.std(np.abs(predictions - ground_truth)) / np.sqrt(num_samples)
            metrics['MAE'] = mae
            metrics['MAE_SE'] = std_error
            
        elif metric == "RMSE":
            squared_errors = np.square(predictions - ground_truth)
            rmse = np.sqrt(np.mean(squared_errors))
            std_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_samples))
            metrics['RMSE'] = rmse
            metrics['RMSE_SE'] = std_error
            
        elif metric == "MAPE":
            mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100
            std_error = np.std(np.abs((predictions - ground_truth) / ground_truth)) / np.sqrt(num_samples) * 100
            metrics['MAPE'] = mape
            metrics['MAPE_SE'] = std_error
            
        elif metric == "Pearson":
            if num_samples > 2:
                corr_matrix = np.corrcoef(predictions, ground_truth)
                correlation_coef = corr_matrix[0][1]
                std_error = np.sqrt((1 - correlation_coef**2) / (num_samples - 2))
                metrics['Pearson'] = correlation_coef
                metrics['Pearson_SE'] = std_error
            else:
                metrics['Pearson'] = np.nan
                metrics['Pearson_SE'] = np.nan
                
        elif metric == "SNR":
            snr_mean = np.mean(snr_values)
            std_error = np.std(snr_values) / np.sqrt(len(snr_values))
            metrics['SNR'] = snr_mean
            metrics['SNR_SE'] = std_error
            
        elif metric == "MACC":
            macc_mean = np.mean(macc_values)
            std_error = np.std(macc_values) / np.sqrt(len(macc_values))
            metrics['MACC'] = macc_mean
            metrics['MACC_SE'] = std_error
    
    # Add basic stats
    metrics['num_windows'] = num_samples
    metrics['mean_prediction'] = np.mean(predictions)
    metrics['mean_ground_truth'] = np.mean(ground_truth)
    metrics['std_prediction'] = np.std(predictions)
    metrics['std_ground_truth'] = np.std(ground_truth)
    
    return metrics


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.) with per-participant analysis."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    
    # Store per-participant results
    participant_results = {}
    
    print("Calculating metrics with per-participant analysis!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        # Extract participant ID
        participant_id = extract_participant_id_from_index(index)
        
        # Initialize participant results if not exists
        if participant_id not in participant_results:
            participant_results[participant_id] = {
                'predictions_peak': [],
                'ground_truth_peak': [],
                'predictions_fft': [],
                'ground_truth_fft': [],
                'snr_values': [],
                'macc_values': []
            }

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                
                # Store per participant
                participant_results[participant_id]['predictions_peak'].append(pred_hr_peak)
                participant_results[participant_id]['ground_truth_peak'].append(gt_hr_peak)
                participant_results[participant_id]['snr_values'].append(SNR)
                participant_results[participant_id]['macc_values'].append(macc)
                
                # Store globally (original functionality)
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                
                # Store per participant
                participant_results[participant_id]['predictions_fft'].append(pred_hr_fft)
                participant_results[participant_id]['ground_truth_fft'].append(gt_hr_fft)
                participant_results[participant_id]['snr_values'].append(SNR)
                participant_results[participant_id]['macc_values'].append(macc)
                
                # Store globally (original functionality)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    # Calculate per-participant metrics
    participant_metrics = {}
    for participant_id, results in participant_results.items():
        if config.INFERENCE.EVALUATION_METHOD == "peak detection":
            if len(results['predictions_peak']) > 0:
                participant_metrics[participant_id] = calculate_metrics_for_participant_supervised(
                    results['predictions_peak'], 
                    results['ground_truth_peak'],
                    results['snr_values'],
                    results['macc_values'],
                    config
                )
        elif config.INFERENCE.EVALUATION_METHOD == "FFT":
            if len(results['predictions_fft']) > 0:
                participant_metrics[participant_id] = calculate_metrics_for_participant_supervised(
                    results['predictions_fft'], 
                    results['ground_truth_fft'],
                    results['snr_values'],
                    results['macc_values'],
                    config
                )

    # Create results directory
    results_dir = f"/zfsauton/data/straps/PURE/resultsFineFFT/{filename_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save individual window predictions and ground truth (PRIMARY OUTPUT)
    window_data_list = []
    for participant_id, results in participant_results.items():
        if config.INFERENCE.EVALUATION_METHOD == "peak detection":
            predictions_list = results['predictions_peak']
            ground_truth_list = results['ground_truth_peak']
        else:  # FFT
            predictions_list = results['predictions_fft']
            ground_truth_list = results['ground_truth_fft']
        
        snr_values = results['snr_values']
        macc_values = results['macc_values']
        
        # Create a row for each window/prediction
        for i, (pred, gt, snr, macc) in enumerate(zip(predictions_list, ground_truth_list, snr_values, macc_values)):
            window_data_list.append({
                'participant_id': participant_id,
                'window_index': i,
                'predicted_hr': pred,
                'ground_truth_hr': gt,
                'snr': snr,
                'macc': macc,
                'absolute_error': abs(pred - gt),
                'relative_error': abs(pred - gt) / gt * 100 if gt != 0 else 0
            })
    
    # Save window-level data (this is your primary output with individual predictions)
    if window_data_list:
        df_windows = pd.DataFrame(window_data_list)
        window_filename = f"{results_dir}/predictions_per_window.csv"
        df_windows.to_csv(window_filename, index=False)
        print(f"Individual window predictions saved to: {window_filename}")
        print(f"Total windows across all participants: {len(df_windows)}")
        
        # Show sample of the data
        print(f"Sample of window-level data:")
        print(df_windows.head(10).to_string(index=False))
    
    # Save per-participant aggregated metrics (SECONDARY OUTPUT - averaged across windows)
    if participant_metrics:
        df_participants = pd.DataFrame.from_dict(participant_metrics, orient='index')
        df_participants.index.name = 'participant_id'
        
        # Save per-participant metrics (averaged)
        csv_filename = f"{results_dir}/per_participant_averaged_metrics.csv"
        df_participants.to_csv(csv_filename)
        print(f"Per-participant averaged metrics saved to: {csv_filename}")
        
        # Save summary statistics
        summary_stats = df_participants.describe()
        summary_filename = f"{results_dir}/participant_metrics_summary.csv"
        summary_stats.to_csv(summary_filename)
        print(f"Summary statistics saved to: {summary_filename}")
        
        # Save participant-level aggregated statistics from window data
        if window_data_list:
            participant_summary = df_windows.groupby('participant_id').agg({
                'predicted_hr': ['mean', 'std', 'min', 'max', 'count'],
                'ground_truth_hr': ['mean', 'std', 'min', 'max'],
                'absolute_error': ['mean', 'std', 'min', 'max'],
                'relative_error': ['mean', 'std', 'min', 'max'],
                'snr': ['mean', 'std'],
                'macc': ['mean', 'std']
            }).round(4)
            
            # Flatten column names
            participant_summary.columns = ['_'.join(col).strip() for col in participant_summary.columns.values]
            participant_summary_filename = f"{results_dir}/participant_window_statistics.csv"
            participant_summary.to_csv(participant_summary_filename)
            print(f"Participant statistics from window data saved to: {participant_summary_filename}")
        
        # Print some basic statistics
        print(f"\n=== PER-PARTICIPANT ANALYSIS (Averaged Across Windows) ===")
        print(f"Total participants: {len(participant_metrics)}")
        if window_data_list:
            total_windows = len(df_windows)
            avg_windows_per_participant = total_windows / len(participant_metrics)
            print(f"Total windows: {total_windows}")
            print(f"Average windows per participant: {avg_windows_per_participant:.1f}")
        
        if 'MAE' in df_participants.columns:
            print(f"MAE (averaged per participant) - Mean: {df_participants['MAE'].mean():.4f}, Std: {df_participants['MAE'].std():.4f}")
            print(f"MAE (averaged per participant) - Best: {df_participants['MAE'].min():.4f}, Worst: {df_participants['MAE'].max():.4f}")
        if 'RMSE' in df_participants.columns:
            print(f"RMSE (averaged per participant) - Mean: {df_participants['RMSE'].mean():.4f}, Std: {df_participants['RMSE'].std():.4f}")
        if 'Pearson' in df_participants.columns:
            print(f"Pearson (averaged per participant) - Mean: {df_participants['Pearson'].mean():.4f}, Std: {df_participants['Pearson'].std():.4f}")
            
        # Also show window-level statistics
        if window_data_list:
            print(f"\n=== INDIVIDUAL WINDOW ANALYSIS ===")
            print(f"MAE (all windows) - Mean: {df_windows['absolute_error'].mean():.4f}, Std: {df_windows['absolute_error'].std():.4f}")
            print(f"MAE (all windows) - Best window: {df_windows['absolute_error'].min():.4f}, Worst: {df_windows['absolute_error'].max():.4f}")
            print(f"Predicted HR range: {df_windows['predicted_hr'].min():.1f} - {df_windows['predicted_hr'].max():.1f} bpm")
            print(f"Ground truth HR range: {df_windows['ground_truth_hr'].min():.1f} - {df_windows['ground_truth_hr'].max():.1f} bpm")
        
        # Save a consolidated results file
        consolidated_results = {
            'model_path': config.INFERENCE.MODEL_PATH,
            'dataset': config.TEST.DATA.DATASET,
            'evaluation_method': config.INFERENCE.EVALUATION_METHOD,
            'total_participants': len(participant_metrics),
            'total_windows': sum(metrics.get('num_windows', 0) for metrics in participant_metrics.values()),
        }
        
        # Add global metrics to consolidated results
        for metric in config.TEST.METRICS:
            if metric in ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC']:
                if metric in df_participants.columns:
                    consolidated_results[f'{metric}_mean'] = df_participants[metric].mean()
                    consolidated_results[f'{metric}_std'] = df_participants[metric].std()
                    consolidated_results[f'{metric}_median'] = df_participants[metric].median()
                    consolidated_results[f'{metric}_min'] = df_participants[metric].min()
                    consolidated_results[f'{metric}_max'] = df_participants[metric].max()
        
        # Save consolidated results
        consolidated_df = pd.DataFrame([consolidated_results])
        consolidated_filename = f"{results_dir}/consolidated_results.csv"
        consolidated_df.to_csv(consolidated_filename, index=False)
        print(f"Consolidated results saved to: {consolidated_filename}")

    # Original global evaluation with improved error handling
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        
        print(f"\n=== GLOBAL RESULTS ===")
        for metric in config.TEST.METRICS:
            try:
                if metric == "MAE":
                    MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                    standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                    print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
                elif metric == "RMSE":
                    squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                    RMSE_FFT = np.sqrt(np.mean(squared_errors))
                    standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                    print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
                elif metric == "MAPE":
                    MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                    standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                    print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
                elif metric == "Pearson":
                    Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                    correlation_coefficient = Pearson_FFT[0][1]
                    standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                    print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
                elif metric == "SNR":
                    SNR_FFT = np.mean(SNR_all)
                    standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
                elif metric == "MACC":
                    MACC_avg = np.mean(MACC_all)
                    standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                    print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
                elif "AU" in metric:
                    pass
                elif "BA" in metric:  
                    try:
                        compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                            file_name=f'{results_dir}/{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                            file_name=f'{results_dir}/{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
                    except Exception as ba_error:
                        print(f"Warning: Could not generate Bland-Altman plots: {ba_error}")
                else:
                    print(f"Warning: Unknown metric '{metric}' encountered. Skipping...")
                    continue
            except Exception as metric_error:
                print(f"Error processing metric '{metric}': {metric_error}")
                continue
                
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        
        print(f"\n=== GLOBAL RESULTS ===")
        for metric in config.TEST.METRICS:
            try:
                if metric == "MAE":
                    MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                    standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                    print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
                elif metric == "RMSE":
                    squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                    RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                    standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                    print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
                elif metric == "MAPE":
                    MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                    standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                    print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
                elif metric == "Pearson":
                    Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                    correlation_coefficient = Pearson_PEAK[0][1]
                    standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                    print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
                elif metric == "SNR":
                    SNR_PEAK = np.mean(SNR_all)
                    standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                    print("PEAK SNR (PEAK Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
                elif metric == "MACC":
                    MACC_avg = np.mean(MACC_all)
                    standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                    print("PEAK MACC (PEAK Label): {0} +/- {1}".format(MACC_avg, standard_error))
                elif "AU" in metric:
                    pass
                elif "BA" in metric:
                    try:
                        compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                        compare.scatter_plot(
                            x_label='GT PPG HR [bpm]',
                            y_label='rPPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                            file_name=f'{results_dir}/{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                        compare.difference_plot(
                            x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                            y_label='Average of rPPG HR and GT PPG HR [bpm]',
                            show_legend=True, figure_size=(5, 5),
                            the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                            file_name=f'{results_dir}/{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
                    except Exception as ba_error:
                        print(f"Warning: Could not generate Bland-Altman plots: {ba_error}")
                else:
                    print(f"Warning: Unknown metric '{metric}' encountered. Skipping...")
                    continue
            except Exception as metric_error:
                print(f"Error processing metric '{metric}': {metric_error}")
                continue
    else:
        raise ValueError("Inference evaluation method name wrong!")