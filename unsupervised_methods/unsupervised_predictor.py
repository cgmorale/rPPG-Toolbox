"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
import pandas as pd
import os
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def extract_participant_id(test_batch, batch_idx, idx):
    """
    Extract participant ID from the test batch.
    This function tries multiple methods to get the participant ID.
    """
    # Method 1: Try to get from batch metadata (if available)
    try:
        if len(test_batch) > 2:
            # Check if there's participant info in the batch
            if hasattr(test_batch[2], '__getitem__') and len(test_batch[2]) > idx:
                participant_info = test_batch[2][idx]
                if hasattr(participant_info, 'item'):
                    participant_id = str(participant_info.item())
                elif isinstance(participant_info, str):
                    participant_id = participant_info
                elif hasattr(participant_info, '__str__'):
                    participant_id = str(participant_info)
                else:
                    raise ValueError("Cannot convert participant info to string")
                    
                # Extract subject number if it's in the format like 'subject1'
                import re
                match = re.search(r'subject(\d+)', participant_id)
                if match:
                    return f"subject{match.group(1)}"
                else:
                    return participant_id
    except:
        pass
    
    # Method 2: Try to get from file paths (if available)
    try:
        if len(test_batch) > 3 and hasattr(test_batch[3], '__getitem__'):
            file_path = str(test_batch[3][idx])
            import re
            match = re.search(r'subject(\d+)', file_path)
            if match:
                return f"subject{match.group(1)}"
    except:
        pass
    
    # Method 3: Fallback to generated ID based on UBFC-rPPG naming convention
    # This assumes sequential loading of subjects
    return f"subject{batch_idx + idx + 1}"

def calculate_metrics_for_participant(predictions, ground_truth, snr_values, macc_values, config):
    """Calculate all metrics for a single participant"""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    snr_values = np.array(snr_values)
    macc_values = np.array(macc_values)
    
    if len(predictions) == 0:
        return {}
    
    num_samples = len(predictions)
    metrics = {}
    
    for metric in config.UNSUPERVISED.METRICS:
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

def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset with per-participant statistics."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    print("Configured metrics:", config.UNSUPERVISED.METRICS)
    
    # Store results per participant
    participant_results = {}
    
    # Global aggregation (keep original functionality)
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []
    MACC_all = []
    
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    
    for batch_idx, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        
        for idx in range(batch_size):
            # Extract participant ID using the helper function
            participant_id = extract_participant_id(test_batch, batch_idx, idx)
            
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
            
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            
            # Apply unsupervised method
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            elif method_name == "OMIT":
                BVP = OMIT(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            # Process windows for this participant
            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    
                    # Store per participant
                    participant_results[participant_id]['predictions_peak'].append(pre_hr)
                    participant_results[participant_id]['ground_truth_peak'].append(gt_hr)
                    participant_results[participant_id]['snr_values'].append(SNR)
                    participant_results[participant_id]['macc_values'].append(macc)
                    
                    # Store globally (original functionality)
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                    
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    
                    # Store per participant
                    participant_results[participant_id]['predictions_fft'].append(pre_fft_hr)
                    participant_results[participant_id]['ground_truth_fft'].append(gt_fft_hr)
                    participant_results[participant_id]['snr_values'].append(SNR)
                    participant_results[participant_id]['macc_values'].append(macc)
                    
                    # Store globally (original functionality)
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                else:
                    raise ValueError("Inference evaluation method name wrong!")

    print("Used Unsupervised Method: " + method_name)

    # Filename ID to be used in any results files
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    # Calculate per-participant metrics
    participant_metrics = {}
    for participant_id, results in participant_results.items():
        if config.INFERENCE.EVALUATION_METHOD == "peak detection":
            if len(results['predictions_peak']) > 0:
                participant_metrics[participant_id] = calculate_metrics_for_participant(
                    results['predictions_peak'], 
                    results['ground_truth_peak'],
                    results['snr_values'],
                    results['macc_values'],
                    config
                )
        elif config.INFERENCE.EVALUATION_METHOD == "FFT":
            if len(results['predictions_fft']) > 0:
                participant_metrics[participant_id] = calculate_metrics_for_participant(
                    results['predictions_fft'], 
                    results['ground_truth_fft'],
                    results['snr_values'],
                    results['macc_values'],
                    config
                )

    # Save per-participant results to CSV
    if participant_metrics:
        df_participants = pd.DataFrame.from_dict(participant_metrics, orient='index')
        df_participants.index.name = 'participant_id'
        
        # Create results directory if it doesn't exist
        results_dir = f"/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/{filename_id}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save per-participant metrics
        csv_filename = f"{results_dir}/per_participant_metrics.csv"
        df_participants.to_csv(csv_filename)
        print(f"Per-participant metrics saved to: {csv_filename}")
        
        # Save summary statistics
        summary_stats = df_participants.describe()
        summary_filename = f"{results_dir}/participant_metrics_summary.csv"
        summary_stats.to_csv(summary_filename)
        print(f"Summary statistics saved to: {summary_filename}")
        
        # Save raw predictions and ground truth per participant
        raw_data_list = []
        for participant_id, results in participant_results.items():
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                predictions = results['predictions_peak']
                ground_truth = results['ground_truth_peak']
            else:  # FFT
                predictions = results['predictions_fft']
                ground_truth = results['ground_truth_fft']
            
            snr_values = results['snr_values']
            macc_values = results['macc_values']
            
            # Create a row for each window/prediction
            for i, (pred, gt, snr, macc) in enumerate(zip(predictions, ground_truth, snr_values, macc_values)):
                raw_data_list.append({
                    'participant_id': participant_id,
                    'window_index': i,
                    'predicted_hr': pred,
                    'ground_truth_hr': gt,
                    'snr': snr,
                    'macc': macc,
                    'absolute_error': abs(pred - gt),
                    'relative_error': abs(pred - gt) / gt * 100 if gt != 0 else 0
                })
        
        # Save raw data to CSV
        if raw_data_list:
            df_raw = pd.DataFrame(raw_data_list)
            raw_filename = f"{results_dir}/raw_predictions_per_window.csv"
            df_raw.to_csv(raw_filename, index=False)
            print(f"Raw predictions per window saved to: {raw_filename}")
            
            # Save participant-level aggregated raw data
            participant_summary = df_raw.groupby('participant_id').agg({
                'predicted_hr': ['mean', 'std', 'min', 'max', 'count'],
                'ground_truth_hr': ['mean', 'std', 'min', 'max'],
                'absolute_error': ['mean', 'std', 'min', 'max'],
                'relative_error': ['mean', 'std', 'min', 'max'],
                'snr': ['mean', 'std'],
                'macc': ['mean', 'std']
            }).round(4)
            
            # Flatten column names
            participant_summary.columns = ['_'.join(col).strip() for col in participant_summary.columns.values]
            participant_summary_filename = f"{results_dir}/participant_summary_stats.csv"
            participant_summary.to_csv(participant_summary_filename)
            print(f"Participant summary statistics saved to: {participant_summary_filename}")
        
        # Print some basic statistics
        print(f"\n=== PER-PARTICIPANT ANALYSIS ===")
        print(f"Total participants: {len(participant_metrics)}")
        if 'MAE' in df_participants.columns:
            print(f"MAE - Mean: {df_participants['MAE'].mean():.4f}, Std: {df_participants['MAE'].std():.4f}")
            print(f"MAE - Best participant: {df_participants['MAE'].min():.4f}, Worst: {df_participants['MAE'].max():.4f}")
        if 'RMSE' in df_participants.columns:
            print(f"RMSE - Mean: {df_participants['RMSE'].mean():.4f}, Std: {df_participants['RMSE'].std():.4f}")
        if 'Pearson' in df_participants.columns:
            print(f"Pearson - Mean: {df_participants['Pearson'].mean():.4f}, Std: {df_participants['Pearson'].std():.4f}")
        
        # Save a consolidated results file
        consolidated_results = {
            'method': method_name,
            'dataset': config.UNSUPERVISED.DATA.DATASET,
            'evaluation_method': config.INFERENCE.EVALUATION_METHOD,
            'total_participants': len(participant_metrics),
            'total_windows': sum(metrics.get('num_windows', 0) for metrics in participant_metrics.values()),
        }
        
        # Add global metrics to consolidated results
        for metric in config.UNSUPERVISED.METRICS:
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

    # Original global evaluation (keep existing functionality) - WITH ERROR HANDLING
    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        
        print("\n=== GLOBAL RESULTS ===")
        for metric in config.UNSUPERVISED.METRICS:
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
                    SNR_FFT = np.mean(SNR_all)
                    standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
                elif metric == "MACC":
                    MACC_avg = np.mean(MACC_all)
                    standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                    print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
                elif "BA" in metric or metric == "BA":
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
                
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        
        print("\n=== GLOBAL RESULTS ===")
        for metric in config.UNSUPERVISED.METRICS:
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
                    SNR_PEAK = np.mean(SNR_all)
                    standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
                elif metric == "MACC":
                    MACC_avg = np.mean(MACC_all)
                    standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                    print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
                elif "BA" in metric or metric == "BA":
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
    else:
        raise ValueError("Inference evaluation method name wrong!")
    
    # Return both global and per-participant results
    return {
        'participant_metrics': participant_metrics,
        'global_predictions': predict_hr_peak_all if config.INFERENCE.EVALUATION_METHOD == "peak detection" else predict_hr_fft_all,
        'global_ground_truth': gt_hr_peak_all if config.INFERENCE.EVALUATION_METHOD == "peak detection" else gt_hr_fft_all
    }