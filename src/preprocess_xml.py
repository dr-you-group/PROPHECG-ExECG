"""Convert GE CardioSoft/CASE exercise ECG XML files into median beats and a master table.

For every XML file named '<PseudoID>#<Name>#<YYYY_MM_DD>#<hh_mm_ss>#.XML' this writes
  <out-dir>/trend/<file>_trend.csv           trend data (one row per trend entry)
  <out-dir>/median/<TestID>.csv              12-lead median beats (one file per median)
  <out-dir>/median/<TestID>_Master.csv       per-ExECG master table: stage of each median beat and the blood
                                             pressure measured closest to it, relative to stage 1
  <out-dir>/DukeTreadmillScore.csv           Duke treadmill score components
  <out-dir>/ProcessedFile.csv                list of processed files
  <out-dir>/whole_validation_master.csv      all master tables combined (input of scripts/Downstream/predict.py)
  <out-dir>/error.txt                        files that could not be processed

usage:
  python preprocess_xml.py --xml-dirs /path/Case2 /path/Case3 --out-dir /path/processed
"""
import argparse
import os
import xml.etree.ElementTree as ET
from ast import literal_eval

import pandas as pd
from tqdm import tqdm


def none_to_empty(value):
    return None if value is None else value.text


def extract_trend(root, xml_file, output_directory):
    """Save the trend data and return the blood-pressure measurements (EntryTime, SystolicBP, DiastolicBP, PhaseName)."""
    data_list = []
    for entry in root.findall(".//TrendEntry"):
        pInfo = entry.find(".//PhaseTime")
        sInfo = entry.find(".//StageTime")
        phase = none_to_empty(entry.find(".//PhaseName"))
        stage_number = none_to_empty(entry.find(".//StageNumber"))
        # recovery segments are labelled '#<n>', exercise stages 'STAGE <n>'
        PhaseName = "#" + stage_number if phase == "RECOVERY" else "STAGE " + stage_number if phase == "EXERCISE" else phase
        strip_time_sec = int(entry.find(".//Minute").text) * 60 + int(entry.find(".//Second").text)
        data_list.append({
            "Idx": entry.attrib["Idx"],
            "Time": f'{none_to_empty(entry.find(".//Minute"))}{none_to_empty(entry.find(".//Second"))}',
            "Mets": none_to_empty(entry.find(".//Mets")),
            "VECount": none_to_empty(entry.find(".//VECount")),
            "PaceCount": none_to_empty(entry.find(".//PaceCount")),
            "Artifact": none_to_empty(entry.find(".//Artifact")),
            "Speed": none_to_empty(entry.find(".//Speed")),
            "Grade": none_to_empty(entry.find(".//Grade")),
            "SystolicBP": none_to_empty(entry.find(".//SystolicBP")),
            "DiastolicBP": none_to_empty(entry.find(".//DiastolicBP")),
            "PhaseTime": f'{none_to_empty(pInfo.find(".//Minute"))}:{none_to_empty(pInfo.find(".//Second"))}',
            "PhaseName": PhaseName,
            "StageTime": f'{none_to_empty(sInfo.find(".//Minute"))}:{none_to_empty(sInfo.find(".//Second"))}',
            "StageNumber": stage_number,
            "EntryTime": strip_time_sec,
        })
    trend = pd.DataFrame(data_list)
    trend = trend.fillna(method='ffill')
    file_name = os.path.basename(xml_file).replace('.XML', '_trend.csv')
    trend.to_csv(os.path.join(output_directory, file_name), index=False)
    return trend[trend["SystolicBP"].notna()][["EntryTime", "SystolicBP", "DiastolicBP", "PhaseName"]]


def st_element(ExerciseMeasurements, status):
    data = {}
    for element in ExerciseMeasurements.findall(status):
        lead = element.attrib.get("lead", None)
        st_amplitude = element.find(".//STAmplitude").text if element.find(".//STAmplitude") is not None else None
        data[lead] = float(st_amplitude)
    return pd.DataFrame(list(data.items()), columns=['lead', 'STAmplitude'])


def maximal_net_ST_segment_deviation(ExerciseMeasurements):
    """Largest absolute change in ST amplitude from rest to the maximal-ST measurement across leads."""
    restST = st_element(ExerciseMeasurements, ".//RestingStats//RestST//Measurements")
    maxST = st_element(ExerciseMeasurements, ".//MaxSTStats//MaxST//Measurements")
    subset = pd.merge(restST, maxST, on='lead', how='left')
    subset["STD"] = abs(subset["STAmplitude_y"] - subset["STAmplitude_x"])
    return subset["STD"].abs().max()


def duke_treadmill_score(root, file):
    """Duke treadmill score = exercise minutes - 5 x ST deviation - 4 x angina index."""
    ExerciseMeasurements = root.find('.//ExerciseMeasurements')
    duration = int(ExerciseMeasurements.find(".//ExercisePhaseTime//Minute").text)
    reason = root.find(".//Interpretation").find(".//ReasonForTermination")
    interpretation = reason.text if reason is not None else 'default value'
    angio_index = 2 if isinstance(interpretation, str) and 'chest' in interpretation else 0
    std = maximal_net_ST_segment_deviation(ExerciseMeasurements)
    return {"Trend_fname": file,
            "Duration": duration,
            "STD": std,
            "Angina": interpretation,
            "DukeScore": duration - (5 * std) - 4 * angio_index}


def extract_median(save_dir, root, pt_id, acq_date, bp):
    """Save each median beat and the master table of the ExECG (stage and blood pressure of every median beat)."""
    rate = root.find(".//MedianData//SampleRate").text
    resolution = root.find(".//MedianData//Resolution").text
    stage_time_df = bp.sort_values('EntryTime').reset_index(drop=True)

    master = []
    for median in root.findall(".//Median"):
        time = median.find(".//Time//Minute").text + str(median.find(".//Time//Second").text.rjust(2, '0'))
        fname = pt_id + '_' + acq_date + '_' + time
        data = {}
        for waveform in median.findall(".//WaveformData"):
            data[waveform.attrib.get("lead")] = literal_eval(waveform.text)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(save_dir, fname + ".csv"), index=False)

        # stage of the median beat = phase of the last blood-pressure entry at or before the beat
        max_range = len(stage_time_df)
        for i in range(1, max_range):
            if (int(time) >= stage_time_df['EntryTime'].loc[i - 1]) & (int(time) < stage_time_df['EntryTime'].loc[i]):
                stage = stage_time_df['PhaseName'].loc[i - 1]
                break
            elif i == max_range - 1:
                stage = stage_time_df['PhaseName'].loc[max_range - 1]
                break
        master.append({'PseudoID': pt_id, 'AcqDate': acq_date, 'TestID': fname, 'shape': df.shape, 'stage': stage,
                       'SampleBase': rate, 'Gain': resolution, "Time": time})

    # blood pressure: the measurement of the same stage closest in time to the median beat
    keys = ['PseudoID', 'AcqDate', 'TestID', 'shape', 'stage', 'SampleBase', 'Gain']
    master_df = pd.merge(pd.DataFrame(master), bp, left_on="stage", right_on="PhaseName", how="left")
    master_df["Time"] = master_df["Time"].astype("int")
    master_df["EntryTime"] = master_df["EntryTime"].fillna(0).astype("int")
    master_df["gap"] = abs(master_df["Time"] - master_df["EntryTime"])
    master_df = master_df.sort_values('gap').groupby(keys).first().reset_index()
    master_df = master_df.sort_values("Time")[keys + ['SystolicBP', 'DiastolicBP', 'Time']].reset_index(drop=True)
    master_df[['SystolicBP', 'DiastolicBP']] = master_df[['SystolicBP', 'DiastolicBP']].fillna(0).astype("int")
    # blood-pressure change relative to stage 1
    master_df["Baseline_Sys"] = master_df[master_df["stage"] == "STAGE 1"]["SystolicBP"].iloc[0]
    master_df["Baseline_Dias"] = master_df[master_df["stage"] == "STAGE 1"]["DiastolicBP"].iloc[0]
    master_df["relative3_sys"] = master_df["SystolicBP"] - master_df["Baseline_Sys"]
    master_df["relative3_dia"] = master_df["DiastolicBP"] - master_df["Baseline_Dias"]
    master_df.to_csv(os.path.join(save_dir, fname + "_Master.csv"), index=False)


def create_whole_master(folder_path):
    median_dir = os.path.join(folder_path, "median")
    masters = [f for f in os.listdir(median_dir) if "_Master.csv" in f]
    whole_master = pd.concat([pd.read_csv(os.path.join(median_dir, m)) for m in tqdm(masters)], axis=0, ignore_index=True)
    whole_master.to_csv(os.path.join(folder_path, "whole_validation_master.csv"))


def process_files(file_list, save_path):
    save_path_trend = os.path.join(save_path, "trend")
    save_path_median = os.path.join(save_path, "median")
    os.makedirs(save_path_trend, exist_ok=True)
    os.makedirs(save_path_median, exist_ok=True)

    duke_score_list, result_list = [], []
    for xml_file in tqdm(file_list):
        pt_id, name, acq_date = os.path.basename(xml_file).split("#")[:3]
        acq_date = ''.join(acq_date.split('_'))

        root = ET.parse(xml_file).getroot()
        age = root.find(".//PatientInfo//Age").text
        gender = root.find(".//PatientInfo//Gender").text
        result_list.append({"PseudoID": pt_id, "Name": name, "AcqDate": acq_date, "AcqAge": age, "Gender": gender,
                            "fname": xml_file})
        try:
            bp = extract_trend(root=root, xml_file=xml_file, output_directory=save_path_trend)
            duke_score_list.append(duke_treadmill_score(root=root, file=xml_file))
            extract_median(root=root, pt_id=pt_id, acq_date=acq_date, save_dir=save_path_median, bp=bp)
        except Exception as e:
            with open(os.path.join(save_path, 'error.txt'), 'a') as f:
                f.write(f'{xml_file} : {e} \n')

    pd.DataFrame(duke_score_list).to_csv(os.path.join(save_path, "DukeTreadmillScore.csv"), index=False)
    pd.DataFrame(result_list).to_csv(os.path.join(save_path, "ProcessedFile.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml-dirs', nargs='+', required=True, help='folders containing the exercise ECG XML files')
    ap.add_argument('--out-dir', required=True, help='output folder')
    a = ap.parse_args()

    file_list = [os.path.join(d, f) for d in a.xml_dirs if os.path.exists(d) for f in os.listdir(d)]
    os.makedirs(a.out_dir, exist_ok=True)
    process_files(file_list, a.out_dir)

    if os.path.isfile(os.path.join(a.out_dir, "whole_validation_master.csv")):
        print("whole_validation_master.csv already exists; not overwritten.")
    else:
        create_whole_master(a.out_dir)


if __name__ == '__main__':
    main()
