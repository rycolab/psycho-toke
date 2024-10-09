import argparse
import math

from typing import List


class RegionOfInterest(object):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class FocalArea(object):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class Stimulus(object):
    def __init__(self, string: str):
        # remove leading and trailing whitespace 
        # and replace multiple whitespace characters with a single space
        self.string = ' '.join(string.strip().split())

    def __str__(self):
        return self.string
    
    def get_regions(self, mode: str):
        """
        Extract regions of interest from the stimulus string.

        Parameters
        ----------
        mode : str
            The mode of extraction. Can be 'leading_whitespace', 'trailing_whitespace', 'leading_and_trailing_whitespace', or 'no_whitespace'.

        Returns
        -------
        list of RegionOfInterest
            A list of regions of interest.

        """
        if mode not in ['leading_whitespace', 'trailing_whitespace', 'leading_and_trailing_whitespace', 'no_whitespace']:
            raise ValueError('Invalid mode. Must be one of "leading_whitespace", "trailing_whitespace", or "leading_and_trailing_whitespace".')

        if self.string == '':
            return []
        
        regions = []
        current_region_start = 0
        for i, char in enumerate(self.string):
            if char.isspace():
                if mode == 'leading_whitespace':
                    regions.append(RegionOfInterest(current_region_start, i))
                    current_region_start = i
                elif mode == 'trailing_whitespace':
                    regions.append(RegionOfInterest(current_region_start, i + 1))
                    current_region_start = i + 1
                elif mode == 'leading_and_trailing_whitespace':
                    regions.append(RegionOfInterest(current_region_start, i + 1))
                    current_region_start = i
                elif mode == 'no_whitespace':
                    regions.append(RegionOfInterest(current_region_start, i))
                    current_region_start = i + 1
                    
        regions.append(RegionOfInterest(current_region_start, len(self.string)))
        return regions
    
    def get_focal_areas(self, regions: List[RegionOfInterest], focal_area_type='dynamic', word_identification_span=7, focal_area_size=3):
        """
        Extract focal areas from a sequence of ROIs.

        Parameters
        ----------
        regions : list of RegionOfInterest
            A list of regions of interest.
        focal_area_type : str, optional
            The type of focal area extraction. Can be 'dynamic' or 'fixed'. The default is 'dynamic'.
        word_identification_span : int, optional
            The size of word identification span for dynamic focal area extraction. The default is 7 (characters).
        focal_area_size : int, optional
            The focal area size for fixed focal area extraction. The default is 3 (characters).

        Returns
        -------
        list of FocalArea
            A list of focal areas.
        """
        if focal_area_type not in ['dynamic', 'fixed', 'whole', 'spanning']:
            raise ValueError('Invalid focal area type. Must be one of "dynamic", "fixed", "whole", or "spanning".')
        
        if not regions:
            return []

        if focal_area_type == 'dynamic':
            # for first region, default to the entire region
            focal_areas = [FocalArea(regions[0].start, regions[0].end)]
            for region, next_region in zip(regions, regions[1:]):
                viewing_location = region.start + math.ceil((region.end - region.start) / 2) - 1
                end_index = viewing_location + word_identification_span + 1 - (region.end - next_region.start)
                if end_index < next_region.start:
                    end_index = next_region.start
                if end_index > next_region.end:
                    end_index = next_region.end
                focal_areas.append(FocalArea(next_region.start, end_index))
            return focal_areas
        elif focal_area_type == 'fixed':
            return [FocalArea(region.start, region.start + focal_area_size) for region in regions]
        elif focal_area_type == 'whole':
            return [FocalArea(region.start, region.end) for region in regions]
        elif focal_area_type == 'spanning':
            assert focal_area_size == -1 or focal_area_size > 0, 'Invalid focal area size. Must be -1 to span the whole next word, or greater than 0.'
            focal_areas = []
            # if focal_area_size is -1, then the focal area spills over the entirety of the next ROI
            if focal_area_size == -1:
                for region, next_region in zip(regions, regions[1:]):
                    focal_areas.append(FocalArea(region.start, next_region.end))
                # the last ROI is a special case, as there is no next ROI
                focal_areas.append(FocalArea(regions[-1].start, regions[-1].end))
            # otherwise, the focal area size spills over the next ROI by the specified size
            else:
                for region, next_region in zip(regions, regions[1:]):
                    focal_areas.append(FocalArea(region.start, next_region.start + focal_area_size))
                focal_areas.append(FocalArea(regions[-1].start, regions[-1].end))

            return focal_areas
        

if __name__ == '__main__':
    """
    Demo: Extract regions of interest and focal areas from a text file.
    """
    parser = argparse.ArgumentParser(description='Extract regions of interest and focal areas from a text file')
    parser.add_argument('--input_file', type=str, help='The input text file', default='data/stimuli/ucl.txt')
    parser.add_argument('--n_stimuli', type=int, default=1, help='The number of stimuli to process.')
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        dataset = [Stimulus(line) for line in f][:args.n_stimuli]

    for i, stimulus in enumerate(dataset):
        print(f'Stimulus {i + 1}: {stimulus}')
        for mode in ['leading_whitespace', 'trailing_whitespace']:
            print(f'Regions of interest (mode={mode}):')
            regions = stimulus.get_regions(mode)
            print([stimulus.string[region.start:region.end] for region in regions])
            print(f'Focal areas (mode={mode}):')
            for focal_area_type in ['dynamic', 'fixed', 'whole', 'spanning']:
                print(f'  Focal area type: {focal_area_type}')
                focal_areas = stimulus.get_focal_areas(regions, focal_area_type=focal_area_type)
                print([stimulus.string[focal_area.start:focal_area.end] for focal_area in focal_areas])
        print()
