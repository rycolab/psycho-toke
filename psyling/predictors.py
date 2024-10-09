import argparse
import pandas as pd
from collections import defaultdict
from stimuli import Stimulus


def load_surprisals(filename: str, skip_eos: bool = True) -> dict:
    """ 
    Load character-level surprisals from a file from a file (white-space separated values).
    The file should have the following format:
    stimulus_id char surprisal
    where stimulus_id is an integer, char is a character (possibly a whitespace), and surprisal is a float.

    Parameters
    ----------
    filename : str
        The path to the file.
    skip_eos : bool
        Whether to skip end-of-sentence tokens.

    Returns
    -------
    dict
        A dictionary mapping stimulus ids to lists of tuples (char, surprisal).
    """
    surprisals = defaultdict(list)
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                parts = [parts[0], ' ', parts[1]]
            elif len(parts) != 3:
                raise ValueError(f"Line '{line}' has {len(parts)} parts, expected 3")
            stimulus_id, char, surprisal = parts
            if char == '<EOS>' and skip_eos:
                continue
            surprisals[int(stimulus_id)].append((char, float(surprisal)))
    return dict(surprisals)



def main(args):
    # Load stimuli
    with open(args.stimuli_file, 'r') as f:
        stimuli = [Stimulus(line) for line in f]
    
    # Load ROI-level measurements
    measurements = pd.read_csv(args.measurements_file)

    # Load character-level surprisals
    with open(args.surprisals_file, 'r') as f:
        surprisals = load_surprisals(args.surprisals_file)

    predictors = []

    for stimulus_id, stimulus in enumerate(stimuli):
        
        # Get regions of interest
        for roi_type in ['leading_whitespace', 'trailing_whitespace']:
            regions = stimulus.get_regions(roi_type)

            # Get focal areas
            for focal_area_type in ['dynamic-7', 'dynamic-8', 'fixed', 'whole', 'spanning-3', 'spanning-4', 'spanning-5', 'spanning-6', 'spanning-7', 'spanning-whole']:
                if focal_area_type.startswith('dynamic'):
                    span = int(focal_area_type.split('-')[1])
                    focal_areas = stimulus.get_focal_areas(regions, focal_area_type='dynamic', word_identification_span=span)
                elif focal_area_type.startswith('spanning'):
                    span = focal_area_type.split('-')[1]
                    if span == 'whole':
                        span = -1
                    focal_areas = stimulus.get_focal_areas(regions, focal_area_type='spanning', focal_area_size=int(span))
                else:
                    focal_areas = stimulus.get_focal_areas(regions, focal_area_type=focal_area_type)

                # Get focal area surprisals
                for region_id, focal_area in enumerate(focal_areas):
                    chars_and_surprisals = surprisals[stimulus_id][focal_area.start:focal_area.end]
                    chars = [char for char, _ in chars_and_surprisals]
                    surprisal = sum([surprisal for _, surprisal in chars_and_surprisals])
                    assert ''.join(chars) == stimulus.string[focal_area.start:focal_area.end], f"Expected '{''.join(chars)}' to be equal to '{stimulus.string[focal_area.start:focal_area.end]}' (stimulus_id={stimulus_id}, region_id={region_id}, roi_type={roi_type}, focal_area_type={focal_area_type})"

                    predictors.append({
                        'StimulusID': stimulus_id + 1,
                        'RegionID': region_id,
                        'RegionType': roi_type,
                        'FocalAreaType': focal_area_type,
                        'FocalArea': ''.join(chars),
                        'Region': stimulus.string[regions[region_id].start:regions[region_id].end],
                        'Surprisal': surprisal
                    })
            
    predictors = pd.DataFrame(predictors)


    predictor_names = {
        'leading_whitespace': 'Leading',
        'trailing_whitespace': 'Trailing',
        'dynamic-7': 'Dynamic7',
        'dynamic-8': 'Dynamic8',
        'fixed': 'Fixed',
        'whole': 'Whole',
        'spanning-3': 'Spanning3',
        'spanning-4': 'Spanning4',
        'spanning-5': 'Spanning5',
        'spanning-6': 'Spanning6',
        'spanning-7': 'Spanning7',
        'spanning-whole': 'SpanningWhole'
    }

    # Replace each StimulusID in measurements with indices starting from 1, to match the indices in predictors
    stimulus_ids = measurements['StimulusID'].unique()
    stimulus_id_to_index = {stimulus_id: i + 1 for i, stimulus_id in enumerate(stimulus_ids)}
    measurements['StimulusID'] = measurements['StimulusID'].apply(lambda x: stimulus_id_to_index[x])

    for i, row in measurements.iterrows():
        stimulus_id = row['StimulusID']
        region_id = row['ContextLength']
        region = row['Region']

        # Ugly fixes for the Provo dataset
        if 'provo' in args.measurements_file:
            if (stimulus_id == 3 and region_id >= 45) or (stimulus_id == 13 and region_id >= 19):
                region_id -= 1
            if stimulus_id == 18 and region_id == 2 and region == 'evolution':
                region_id = 50
            if stimulus_id == 36 and region_id >= 24:
                region_id += 1

        # Find the matching row in the predictors dataframe
        matching_rows = predictors[(predictors['StimulusID'] == stimulus_id) & (predictors['RegionID'] == region_id)]
        if len(matching_rows) != 40:
            raise ValueError(f"Expected to find exactly 40 matching rows (4 ROI types x 10 FA types), but found {len(matching_rows)}")

        # Assert that the regions match
        region_measurements = region[:-1] if region[-1] == '.' else region
        region_predictors = matching_rows['Region'].iloc[0].strip()
        region_predictors = region_predictors[:-1] if region_predictors[-1] == '.' else region_predictors
        # with lots of ugly special cases
        if 'meco' in args.measurements_file:
            region_measurements = region_measurements.replace('–', '-')
        if 'celer' in args.measurements_file:
            region_measurements = region_measurements.replace('"', '')
            region_measurements = region_measurements[:-1] if region_measurements[-1] == '.' else region_measurements
            region_predictors = region_predictors.replace('"', '')
            region_predictors = region_predictors[:-1] if region_predictors[-1] == '.' else region_predictors

        if region_measurements != region_predictors:
            print(f"Expected '{region_measurements}' to be equal to '{region_predictors}'")
            print(row)
            print(matching_rows)
            exit()

        # Focal area surprisals
        for j, matching_row in matching_rows.iterrows():
            predictor_name = f"Surprisal{predictor_names[matching_row['RegionType']]}{predictor_names[matching_row['FocalAreaType']]}"
            measurements.at[i, predictor_name] = matching_row['Surprisal']  
                
        # Predictors for spillover effects
        measurements.at[i, "RegionLengthPrev"] = 0
        measurements.at[i, "RegionLengthPrevPrev"] = 0
        measurements.at[i, "WordFreqZipfPrev"] = 0
        measurements.at[i, "WordFreqZipfPrevPrev"] = 0
        measurements.at[i, "SurprisalLeadingPrev"] = 0
        measurements.at[i, "SurprisalLeadingPrevPrev"] = 0
        measurements.at[i, "SurprisalTrailingPrev"] = 0
        measurements.at[i, "SurprisalTrailingPrevPrev"] = 0
        measurements.at[i, "SurprisalLeadingAndTrailingPrev"] = 0
        measurements.at[i, "SurprisalLeadingAndTrailingPrevPrev"] = 0
        measurements.at[i, "SurprisalNoWhitespacePrev"] = 0
        measurements.at[i, "SurprisalNoWhitespacePrevPrev"] = 0

        prev_df = measurements[
            (measurements['StimulusID'] == stimulus_id) & 
            (measurements['ContextLength'] == region_id - 1)
        ]
        if len(prev_df) == 1 or ('provo' in args.measurements_file and len(prev_df) == 2 and prev_df['Region'].iloc[0] == 'evolution' and prev_df['Region'].iloc[1] == 'the'):
            measurements.at[i, "RegionLengthPrev"] = prev_df['RegionLength'].iloc[0]
            measurements.at[i, "WordFreqZipfPrev"] = prev_df['WordFreqZipf'].iloc[0]
   
            # Surprisal at t - 1
            for roi_type in ['leading_whitespace', 'trailing_whitespace']:
                matching_rows_prev = predictors[
                    (predictors['StimulusID'] == stimulus_id) & 
                    (predictors['RegionID'] == region_id - 1) &
                    (predictors['RegionType'] == roi_type) &
                    (predictors['FocalAreaType'] == 'whole')
                ]
                if len(matching_rows_prev) != 1:
                    raise ValueError(f"Expected to find exactly 1 matching row, but found {len(matching_rows_prev)}")
                if roi_type == 'leading_whitespace':
                    measurements.at[i, "SurprisalLeadingPrev"] = matching_rows_prev['Surprisal'].iloc[0]
                elif roi_type == 'trailing_whitespace':
                    measurements.at[i, "SurprisalTrailingPrev"] = matching_rows_prev['Surprisal'].iloc[0]
                elif roi_type == 'leading_and_trailing_whitespace':
                    measurements.at[i, "SurprisalLeadingAndTrailingPrev"] = matching_rows_prev['Surprisal'].iloc[0]
                elif roi_type == 'no_whitespace':
                    measurements.at[i, "SurprisalNoWhitespacePrev"] = matching_rows_prev['Surprisal'].iloc[0]

        elif len(prev_df) > 1:
            print("More than one match")
            print(prev_df)
            exit()
            
        prev_prev_df = measurements[
            (measurements['StimulusID'] == stimulus_id) & 
            (measurements['ContextLength'] == region_id - 2)
        ]
        if len(prev_prev_df) == 1 or ('provo' in args.measurements_file and len(prev_prev_df) == 2 and prev_prev_df['Region'].iloc[0] == 'evolution' and prev_prev_df['Region'].iloc[1] == 'the'):
            measurements.at[i, "RegionLengthPrevPrev"] = prev_prev_df['RegionLength'].iloc[0]
            measurements.at[i, "WordFreqZipfPrevPrev"] = prev_prev_df['WordFreqZipf'].iloc[0]
   
            # Surprisal at t - 2
            for roi_type in ['leading_whitespace', 'trailing_whitespace']:
                matching_rows_prev_prev = predictors[
                    (predictors['StimulusID'] == stimulus_id) & 
                    (predictors['RegionID'] == region_id - 2) &
                    (predictors['RegionType'] == roi_type) &
                    (predictors['FocalAreaType'] == 'whole')
                ]
                if len(matching_rows_prev_prev) != 1:
                    raise ValueError(f"Expected to find exactly 1 matching row, but found {len(matching_rows_prev_prev)}")
                if roi_type == 'leading_whitespace':
                    measurements.at[i, "SurprisalLeadingPrevPrev"] = matching_rows_prev_prev['Surprisal'].iloc[0]
                elif roi_type == 'trailing_whitespace':
                    measurements.at[i, "SurprisalTrailingPrevPrev"] = matching_rows_prev_prev['Surprisal'].iloc[0]
                elif roi_type == 'leading_and_trailing_whitespace':
                    measurements.at[i, "SurprisalLeadingAndTrailingPrevPrev"] = matching_rows_prev_prev['Surprisal'].iloc[0]
                elif roi_type == 'no_whitespace':
                    measurements.at[i, "SurprisalNoWhitespacePrevPrev"] = matching_rows_prev_prev['Surprisal'].iloc[0]
                    
        elif len(prev_prev_df) > 1:
            print("More than one match")
            print(prev_prev_df)
            exit()

    measurements.to_csv(args.output_file, index=False)
    print(f"Predictors saved to {args.output_file}")


if __name__ == '__main__':
    """
    Get predictors for each region of interest and focal area.
    """
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--stimuli_file', type=str, help='The text file containing stimuli', default='data/stimuli/mecoL1.txt')
    # parser.add_argument('--measurements_file', type=str, help='The csv file containing measurements', default='data/measurements/mecoL1_clean.csv')
    # parser.add_argument('--surprisals_file', type=str, help='The txt file containing character-level surprisals', default='data/surprisals/mecoL1_surprisals_K5.gpt2.txt')
    # parser.add_argument('--output_file', type=str, help='The output csv file', default='analysis/mecoL1_gpt2_k5.csv')
    # parser.add_argument('--n_stimuli', type=int, default=1, help='The number of stimuli to process (for debugging).')
    args = parser.parse_args()


    for corpus in ['mecoL1', 'celer', 'provo', 'ucl']:
        args.stimuli_file = f'data/stimuli/{corpus}.txt'
        args.measurements_file = f'data/measurements/{corpus}_clean.csv'
        args.surprisals_file = f'data/surprisals/{corpus}_surprisals_K5.gpt2.txt'
        args.output_file = f'analysis/data/{corpus}_gpt2_k5.csv'
        main(args)

