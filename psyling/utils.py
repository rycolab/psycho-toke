import os
import numpy as np
import pandas as pd

from typing import List
from string import punctuation

from wordfreq import word_frequency, zipf_frequency

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def extract_dundee_stimuli(input_dir: str, output_file: str) -> List[str]:
    """
    Extract stimuli from the Dundee corpus and save them to a text file.

    Parameters
    ----------
    input_dir : str
        The path to the directory containing the Dundee corpus .dat files.
    output_file : str
        The path to save the stimuli.

    Returns
    -------
    list of str
        A list of stimuli.
    """
    stimuli = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".dat"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_csv(
                file_path, 
                usecols=[0], 
                delim_whitespace=True, 
                names=['roi'], 
                header=None, 
                encoding='utf-8',
                encoding_errors='replace',
                dtype={'roi': str},
                na_values=[''],
                na_filter=False
            )
            stimuli.append(' '.join(df['roi']))

    # Save the texts to a file, one text per line
    with open(output_file, 'w') as f:
        for stimulus in stimuli:
            f.write(stimulus + '\n')
    
    return stimuli


def extract_meco_stimuli(dataset_path: str, output_path: str) -> List[str]:
    """
    Extract stimuli from the MECO dataset and save them to a text file.
    Run this after cleaning the original dataset using `clean_meco`. 
    
    Parameters
    ----------
    dataset_path : str
        The path to the cleaned MECO dataset (as output by `clean_meco`).
    output_path : str
        The path to save the stimuli.

    Returns
    -------
    list of str
        A list of stimuli.
    """
    df = pd.read_csv(dataset_path)
    df_sorted = df.sort_values(by=['trialid', 'ianum'])
    df_grouped = df_sorted.groupby(['trialid', 'ianum']).first().reset_index()
    df_grouped = df_grouped.groupby('trialid').agg({'ia': lambda x: ' '.join(x)})
    stimuli = df_grouped.ia.to_list()
    with open(output_path, 'w') as f:
        for s in stimuli:
            s = s.replace("’", "'").replace('“', '"').replace('”', '"').replace('–', '-')
            f.write(s + '\n')
    return stimuli


def log_freq(x):
    return np.log(x + 1e-10)

def clean_provo(dataset_path: str, output_path: str, discard_punctuation_and_clitics: bool = False) -> pd.DataFrame:
    """
    Clean the Provo dataset and save it to a csv file.

    Parameters
    ----------
    dataset_path : str
        The path to the Provo dataset.
    output_path : str
        The path to save the cleaned dataset.
    discard_punctuation_and_clitics : bool, optional
        Whether to discard punctuation and clitics. The default is False.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """

    provo = pd.read_csv(dataset_path, dtype={'Word': str, 'Word_Cleaned': str})

    # Columns to keep
    columns_stimulus_info = [
        'Text_ID', 'Word_Number', 'Word', 'Word_Cleaned', 'Word_Length', 'Word_Content_Or_Function', 'Word_POS', 
    ]
    columns_rt = [
        'IA_FIRST_FIXATION_DURATION', 'IA_DWELL_TIME', 'IA_SKIP',
    ]

    provo = provo[columns_stimulus_info + columns_rt].dropna()

    # Punctuation and clitics
    provo["PunctOrClitic"] = provo.Word.apply(contains_punctuation_or_clitic)
    if discard_punctuation_and_clitics:
        provo = provo[provo.PunctOrClitic == 0]

    # Aggregate RTs over participants
    provo = provo.groupby(columns_stimulus_info, as_index=False).mean()
    provo = provo.sort_values(by=['Text_ID', 'Word_Number'])
    
    # Get WordFreq frequencies
    vocab = provo['Word'].unique()
    vocab_clean = dict(zip(provo['Word'], provo['Word_Cleaned']))
    for word in vocab:
        freq = word_frequency(word, 'en')
        if freq == 0:
            freq = word_frequency(vocab_clean[word], 'en')
        provo.loc[provo['Word'] == word, 'WordFreq'] = freq

    provo['WordFreqLog'] = provo['WordFreq'].apply(log_freq)
    
    # Get WordFreq Zipf frequencies
    for word in vocab:
        freq = zipf_frequency(word, 'en')
        if freq == 0:
            freq = zipf_frequency(vocab_clean[word], 'en')
        provo.loc[provo['Word'] == word, 'WordFreqZipf'] = freq


    # Cosmetics
    provo = provo.rename(columns={
        'Word': 'Region', 'Word_Cleaned': 'RegionCleaned', 'Text_ID': 'StimulusID',
        'Word_Number': 'ContextLength', 'Word_Length': 'RegionLength', 
        'Word_Content_Or_Function': 'RegionContentOrFunction', 'Word_POS': 'RegionPOS', 
        'IA_FIRST_FIXATION_DURATION': 'FirstFixationDuration', 'IA_DWELL_TIME': 'DwellTime', 'IA_SKIP': 'SkipRate'
    })
    provo['ContextLength'] = (provo['ContextLength'] - 1).astype(int)
    provo['RegionLength'] = provo['RegionLength'].astype(int)
    provo['Region'] = provo['Region'].astype(str)
    provo['RegionCleaned'] = provo['RegionCleaned'].astype(str)
    provo = provo[[
        'StimulusID', 'ContextLength', 'Region', 'RegionCleaned', 'RegionLength', 'WordFreq', 'WordFreqLog', 'WordFreqZipf', 
        'PunctOrClitic', 'RegionContentOrFunction', 'RegionPOS', 'FirstFixationDuration', 'DwellTime', 'SkipRate'
    ]]

    # Save the cleaned dataset
    provo.to_csv(output_path, index=False)

    return provo


def clean_ucl(dataset_path: str, output_path: str, discard_punctuation_and_clitics: bool = False) -> pd.DataFrame:
    """
    Clean the UCL dataset and save it to a csv file.
    
    Parameters
    ----------
    dataset_path : str
        The path to the UCL dataset.
    output_path : str
        The path to save the cleaned dataset.
    discard_punctuation_and_clitics : bool, optional
        Whether to discard punctuation and clitics. The default is False.

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """

    ucl = pd.read_csv(dataset_path)

    # Columns to keep
    columns = ['sent_id', 'context_length', 'word', 'length', 'RTfirstfix', 'RTfirstpass', 'RTrightbound', 'RTgopast']
    ucl = ucl[columns]

    # Punctuation and clitics
    ucl["PunctOrClitic"] = ucl.word.apply(contains_punctuation_or_clitic)
    if discard_punctuation_and_clitics:
        ucl = ucl[ucl.PunctOrClitic == 0]

    # Get WordFreq frequencies
    vocab = ucl['word'].unique()
    for word in vocab:
        freq = word_frequency(word, 'en')
        ucl.loc[ucl['word'] == word, 'WordFreq'] = freq

    # Get WordFreq Zipf frequencies
    for word in vocab:
        freq = zipf_frequency(word, 'en')
        ucl.loc[ucl['word'] == word, 'WordFreqZipf'] = freq

    ucl['WordFreqLog'] = ucl['WordFreq'].apply(log_freq)

    # Cosmetics
    ucl = ucl.rename(columns={
        'sent_id': 'StimulusID', 'word': 'Region', 
        'length': 'RegionLength', 'context_length': 'ContextLength',
        'RTfirstfix': 'FirstFixationDuration', 'RTfirstpass': 'FirstPassDuration',
        'RTrightbound': 'RightBoundedDuration', 'RTgopast': 'GoPastDuration'
    })
    ucl = ucl[[
        'StimulusID', 'ContextLength', 'Region', 'RegionLength', 'WordFreq', 'WordFreqLog', 'WordFreqZipf', 
        'PunctOrClitic', 'FirstFixationDuration', 'FirstPassDuration', 'RightBoundedDuration', 'GoPastDuration'
    ]]

    ucl = ucl.sort_values(by=['StimulusID', 'ContextLength'])

    # Save the cleaned dataset
    ucl.to_csv(output_path, index=False)

    return ucl


def clean_meco(dataset_path: str, output_path: str, discard_punctuation_and_clitics: bool = False) -> pd.DataFrame:
    """
    Clean the MECO dataset and save it to a csv file.

    Parameters
    ----------
    dataset_path : str
        The path to the MECO dataset.
    output_path : str
        The path to save the cleaned dataset.
    discard_punctuation_and_clitics : bool, optional
        Whether to discard punctuation and clitics. The default is False.
    
    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """

    meco = pd.read_csv(dataset_path)
    
    # Filter out non-English data
    meco = meco[meco['lang'] == 'en']
    
    # Columns to keep
    columns_stimulus_info = [
        "trialid", "ianum", "ia", "freq"
    ]
    columns_measurements = [
        "total_rt", "gaze_rt", "firstfix_rt"
    ]
    meco = meco[columns_stimulus_info + columns_measurements]

    # Aggregate measurements over participants
    meco = meco.groupby(columns_stimulus_info, as_index=False).mean()

    # Context and region length
    meco['ianum'] = (meco['ianum'] - 1).astype(int)
    meco['RegionLength'] = meco['ia'].apply(lambda x: len(x))

    meco['ia'] = meco['ia'].str.replace("’", "'").str.replace('“', '"').str.replace('”', '"')

    # Punctuation and clitics
    meco["PunctOrClitic"] = meco.ia.apply(contains_punctuation_or_clitic)
    if discard_punctuation_and_clitics:
        meco = meco[meco.PunctOrClitic == 0]

    # Get WordFreq frequencies
    vocab = meco['ia'].unique()
    for word in vocab:
        freq = word_frequency(word, 'en')
        meco.loc[meco['ia'] == word, 'WordFreq'] = freq

    meco['WordFreqLog'] = np.log(meco['WordFreq'])

    # Get WordFreq Zipf frequencies
    for word in vocab:
        freq = zipf_frequency(word, 'en')
        meco.loc[meco['ia'] == word, 'WordFreqZipf'] = freq

    # Cosmetics
    meco = meco.rename(columns={
        'trialid': 'StimulusID',
        'ianum': 'ContextLength',
        'ia': 'Region',
        'skip': 'SkipRate',
        'nfix': 'NumFixations',
        'total_rt': 'TotalDuration',
        'gaze_rt': 'GazeDuration',
        'firstfix_rt': 'FirstFixationDuration',
        'freq': 'Freq'
    })
    meco = meco[[
        'StimulusID', 'ContextLength', 'Region', 'RegionLength', 'Freq', 'WordFreq', 'WordFreqLog', 'WordFreqZipf',
        'PunctOrClitic', 'TotalDuration', 'GazeDuration', 'FirstFixationDuration'
    ]]

    # Save the cleaned dataset
    meco.to_csv(output_path, index=False)

    return meco


def clean_celer(dataset_path: str, output_path: str, metadata_path: str, discard_punctuation_and_clitics: bool = False) -> pd.DataFrame:
    """
    Clean the Celer dataset and save it to a csv file.

    Parameters
    ----------
    dataset_path : str
        The path to the Celer dataset.
    output_path : str
        The path to save the cleaned dataset.
    discard_punctuation_and_clitics : bool, optional
        Whether to discard punctuation and clitics. The default is False.
    
    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
        
    celer = pd.read_csv(dataset_path, sep='\t')
    celer = celer[(celer['dataset_version'] == 2) & (celer['shared_text'] == 1)]

    # Remove participants that don't have English as their L1
    celer_metadata_df = pd.read_csv(metadata_path, sep='\t')
    filtered_celer_metadata = celer_metadata_df[celer_metadata_df['L1'] == 'English']

    celer = celer[celer['list'].isin(filtered_celer_metadata['List'].tolist())]
    
    # Columns to keep
    columns_stimulus_info = [
        'sentenceid', 'IA_ID', 'IA_LABEL', 'WORD_NORM', 'WORD_LEN'
    ]
    columns_measurements = [
        'IA_DWELL_TIME', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_LANDING_POSITION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_SKIP'
    ]
    celer = celer[columns_stimulus_info + columns_measurements]
    for col in columns_measurements:
        celer[col] = celer[col].replace(".", np.nan).astype(float)

    # Aggregate RTs over participants
    celer = celer.groupby(columns_stimulus_info, as_index=False).mean()
    celer = celer.sort_values(by=['sentenceid', 'IA_ID'])
    
    # Context length
    celer['IA_ID'] = (celer['IA_ID'] - 1).astype(int)

    # Punctuation and clitics
    celer["PunctOrClitic"] = celer["IA_LABEL"].apply(contains_punctuation_or_clitic)
    if discard_punctuation_and_clitics:
        celer = celer[celer.PunctOrClitic == 0]


    # Get WordFreq frequencies
    vocab = celer['IA_LABEL'].unique()
    vocab_clean = dict(zip(celer['IA_LABEL'], celer['WORD_NORM']))
    for word in vocab:
        freq = word_frequency(word, 'en')
        if freq == 0:
            freq = word_frequency(vocab_clean[word], 'en')
        celer.loc[celer['IA_LABEL'] == word, 'WordFreq'] = freq

    celer['WordFreqLog'] = np.log(celer['WordFreq'])

    # Get WordFreq Zipf frequencies
    for word in vocab:
        freq = zipf_frequency(word, 'en')
        if freq == 0:
            freq = zipf_frequency(vocab_clean[word], 'en')
        celer.loc[celer['IA_LABEL'] == word, 'WordFreqZipf'] = freq


    # Cosmetics
    celer = celer.rename(columns={
        'sentenceid': 'StimulusID',
        'IA_ID': 'ContextLength',
        'IA_LABEL': 'Region',
        'WORD_NORM': 'RegionCleaned',
        'WORD_LEN': 'RegionLength',
        'IA_DWELL_TIME': 'DwellTime',
        'IA_FIRST_FIXATION_DURATION': 'FirstFixationDuration',
        'IA_FIRST_RUN_DWELL_TIME': 'FirstPassDwellTime',
        'IA_FIRST_RUN_LANDING_POSITION': 'FirstPassLandingPosition',
        'IA_SKIP': 'SkipRate',
    })
    celer = celer[[
        'StimulusID', 'ContextLength', 'Region', 'RegionCleaned', 'RegionLength', 'WordFreq', 'WordFreqLog', 'WordFreqZipf',
        'PunctOrClitic', 'DwellTime', 'FirstFixationDuration', 'FirstPassDwellTime', 'FirstPassLandingPosition', 'SkipRate'
    ]]

    # Save the cleaned dataset
    celer.to_csv(output_path, index=False)

    return celer


def extract_celer_stimuli(dataset_path: str, metadata_path: str, output_path: str) -> List[str]:
    """
    Extract stimuli from the Celer dataset and save them to a text file.
    Run this after cleaning the original dataset using `clean_celer`. 
    
    Parameters
    ----------
    dataset_path : str
        The path to the cleaned Celer dataset (as output by `clean_celer`).
    metadata_path : str
        The path to the Celer metadata file.
    output_path : str
        The path to save the stimuli.

    Returns
    -------
    list of str
        A list of stimuli.
    """
    df = pd.read_csv(dataset_path, sep='\t')
    df = df[(df['dataset_version'] == 2) & (df['shared_text'] == 1)]

    # Remove participants that don't have English as their L1
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    filtered_metadata = metadata_df[metadata_df['L1'] == 'English']
    df = df[df['list'].isin(filtered_metadata['List'].tolist())]

    # Get unique stimuli
    df_sorted = df.sort_values(by=['sentenceid'])
    df_grouped = df_sorted.groupby(['sentenceid', 'sentence']).first().reset_index()
    stimuli = df_grouped.sentence.to_list()

    with open(output_path, 'w') as f:
        for s in stimuli:
            s.replace("\'", "'")
            f.write(s + '\n')
    return stimuli


def contains_punctuation_or_clitic(s: str) -> bool:
    """
    Check if a string contains punctuation or clitics.

    Parameters
    ----------
    s : str
        The string to check.
    
    Returns
    -------
    bool
        True if the string contains punctuation or clitics, False otherwise.
    """
    if "." in s:
        return True
    elif "," in s:
        return True
    elif "'" in s:
        return True
    else:
        return False


if __name__ == "__main__":

    provo_dataset_path = "data/measurements/provo.csv"
    provo_output_path = "data/measurements/provo_clean.csv"
    if not os.path.exists(provo_output_path):
        clean_provo(provo_dataset_path, provo_output_path)

    ucl_dataset_path = "data/measurements/ucl.csv"
    ucl_output_path = "data/measurements/ucl_clean.csv"
    if not os.path.exists(ucl_output_path):
        clean_ucl(ucl_dataset_path, ucl_output_path)

    meco_dataset_path = "data/measurements/mecoL1.csv"
    meco_output_path = "data/measurements/mecoL1_clean.csv"
    if not os.path.exists(meco_output_path):
        clean_meco(meco_dataset_path, meco_output_path)

    meco_output_stimuli_path = "data/stimuli/mecoL1.txt"
    if not os.path.exists(meco_output_stimuli_path):
        extract_meco_stimuli(meco_dataset_path, meco_output_stimuli_path)

    celer_dataset_path = "data/measurements/celer.tsv"
    celer_metadata_path = "data/measurements/celer_metadata.tsv"
    celer_output_path = "data/measurements/celer_clean.csv"
    if not os.path.exists(meco_output_path):
        clean_celer(celer_dataset_path, celer_output_path, celer_metadata_path)
    
    celer_output_stimuli_path = "data/stimuli/celer.txt"
    if not os.path.exists(celer_output_stimuli_path):
        extract_celer_stimuli(celer_dataset_path, celer_metadata_path, celer_output_stimuli_path)

