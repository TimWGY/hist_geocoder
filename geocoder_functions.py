##########################################################################################
# V20230517

from __future__ import annotations

import os
os.system('pip install unidecode number-parser==0.3.0 thefuzz python-Levenshtein jellyfish geopandas rtree pyproj mapclassify')

import pandas as pd
pd.set_option('chained_assignment', None)

street_index_df = pd.read_csv('/content/hist_geocoder/street_data/mnbk_streets_index_v20230515.csv')
street_index_df['street_name_stem'] = street_index_df['street_name_stem'].fillna('')
ybs1_id_to_stname_mapping = street_index_df.set_index('ybs1_id')['stname'].to_dict()
street_range_df = pd.read_csv('/content/hist_geocoder/street_data/mnbk_streets_range_v20230515.csv')
street_range_df['start_end_coordinates'] = street_range_df['start_end_coordinates'].apply(eval)
                                                                                          
##########################################################################################

import geopandas as gpd
import numpy as np
import re

from unidecode import unidecode
def deaccent_lowercase_remove_most_special(x):
    return re.sub(r'\s+',' ',re.sub('[^a-z0-9 \-\&\/]','',unidecode(x).lower())).strip() if isinstance(x,str) else np.nan

def deaccent_lowercase_remove_all_special(x):
    return re.sub(r'\s+',' ',re.sub('[^a-z0-9 ]','',unidecode(x).lower())).strip() if isinstance(x,str) else np.nan
  
from number_parser import parse as parse_number_text
parse_number_text_memory = {}
def parse_number_text_with_memory(input_string):
    if input_string in parse_number_text_memory:
        return parse_number_text_memory[input_string]
    else:
        output_string = parse_number_text(input_string)
        parse_number_text_memory[input_string] = output_string
        return output_string

from collections import OrderedDict
street_name_type_variation_dict = OrderedDict()
street_name_type_variation_dict['st'] = ['street', 'stree', 'sreet', 'treet', 'stret', 'stre', 'stee', 'seet', 'reet', 'st']
street_name_type_variation_dict['dr'] = ['drive', 'driv', 'drie', 'drv', 'dri', 'dr', 'dv', 'de']
street_name_type_variation_dict['cir'] = ['circle', 'circl', 'cicle', 'circ', 'cir', 'crl', 'cl', 'cr']
street_name_type_variation_dict['av'] = ['avenue', 'avenu', 'avnue', 'aven', 'avn', 'ave', 'av']
street_name_type_variation_dict['ct'] = ['court', 'cour', 'cort', 'crt', 'ctr', 'cot', 'ct']
street_name_type_variation_dict['blvd'] = ['boulevard', 'blvd', 'bvd', 'bld']
street_name_type_variation_dict['rd'] = ['road', 'rod', 'rad', 'rd']
street_name_type_variation_dict['aly'] = ['alley', 'ally', 'aley', 'alle', 'aly', 'al', 'ay']
street_name_type_variation_dict['pl'] = ['place', 'plac', 'plce', 'plc', 'pce', 'pl']
street_name_type_variation_dict['park'] = ['park', 'prak', 'pak', 'prk', 'pk']
street_name_type_variation_dict['dvwy'] = ['driveway', 'dvwy', 'dy']
street_name_type_variation_dict['pkwy'] = ['parkway', 'parkw', 'prkwy', 'pkwy', 'pwy', 'pkw', 'py']
street_name_type_variation_dict['hwy'] = ['highway', 'hgwy', 'hwy']
street_name_type_variation_dict['expwy'] = ['expressway', 'expwy', 'express', 'expre']
street_name_type_variation_dict['tnpk'] = ['turnpike']
street_name_type_variation_dict['appr'] = ['approach', 'approa', 'apprch', 'aprch', 'appr', 'apr']
street_name_type_variation_dict['ter'] = ['terrace', 'terr', 'ter', 'trce', 'trc', 'tr']
street_name_type_variation_dict['plz'] = ['plaza', 'plza', 'plaz', 'plz', 'pz']
street_name_type_variation_dict['ln'] = ['lane', 'lan', 'lne', 'ln', 'la']
street_name_type_variation_dict['brg'] = ['bridge', 'brdg', 'brg', 'bge', 'br']
street_name_type_variation_dict['hl'] = ['hill', 'hil', 'hll', 'hl']
street_name_type_variation_dict['hts'] = ['heights', 'height', 'heights', 'heghts', 'heigt', 'hts', 'ht']
street_name_type_variation_dict['slip'] = ['slip', 'slep', 'slp']
street_name_type_variation_dict['row'] = ['row', 'rw']
street_name_type_variation_dict['sq'] = ['square', 'sqr', 'sq']
street_name_type_variation_dict['wy'] = ['way','wy']
street_name_type_variation_dict['ext'] = ['extension','ext','ex']
street_name_type_variation_dict['walk'] = ['walk']
street_name_type_variation_dict['cir'] = ['circle','cir','c']
street_name_type_variation_dict['pr'] = ['pier']
street_name_type_variation_dict['wf'] = ['wharf']
street_name_type_variation_dict['pth'] = ['path']

variation_to_standard_street_name_type_mapping = {}
for k,v in street_name_type_variation_dict.items():
    for x in v:
        variation_to_standard_street_name_type_mapping[x]=k

def extract_street_name_direction(input_string: str) -> tuple[str, str|None, str|None]:
    """
    """
    for direction in ['west','east','south','north']: # detect suffix first, handle cases such as "e mosholu parkway n"
        output_string, num_of_sub_made = re.subn(fr'\s{direction[0]}({direction[1:]})?$', '', input_string)
        if num_of_sub_made > 0:
            return output_string, direction, 'suffix'
    for direction in ['west','east','south','north']:
        output_string, num_of_sub_made = re.subn(fr'^{direction[0]}({direction[1:]})?\s', '', input_string)
        if num_of_sub_made > 0:
            return output_string, direction, 'prefix'
    return input_string, None, None

def extract_street_name_type(input_string: str, variation_to_standard_street_name_type_mapping: dict[str, str]) -> tuple[str, str|None, str|None]:
    """
    """
    street_name_type = None
    first_component, output_string = input_string.split(maxsplit=1) if ' ' in input_string else (input_string, '')
    if first_component in ['avenue','ave','av']: # only "ave" are used as suffix in street names (Ave A, Ave B, ...)
        return output_string, 'av', 'prefix'
    output_string, last_component = input_string.rsplit(maxsplit=1) if ' ' in input_string else ('', input_string)
    street_name_type = variation_to_standard_street_name_type_mapping.get(last_component, None)
    if street_name_type:
        return output_string, street_name_type, 'suffix'
    else:
        return input_string, None, None

def custom_regex_standandization(x):
    x = re.sub('[^a-z0-9 ]','',x).strip()

    x = re.sub(r'^de (\w+)',r'de\1',x)
    x = re.sub(r'^la (\w+)',r'la\1',x)
    x = re.sub(r'^mac (\w+)',r'mac\1',x)
    x = re.sub(r'^mc (\w+)',r'mc\1',x)
    x = re.sub(r'^van (\w+)',r'van\1',x)

    x = re.sub(r'^ft (\w+)',r'fort \1',x)
    x = re.sub(r'^st (\w+)',r'saint \1',x)
    x = re.sub(r'^rev (\w+)',r'reverend \1',x)

    x = re.sub(r' service (road|rd)( \w+)?$',r'',x)
    x = re.sub(r' sr( \w+)?$',r'',x)
    
    x = re.sub(r' footbridge$',r'',x)
    return x

def parse_street_name(street_name):
    street_name = custom_regex_standandization(street_name)
    street_name, street_name_direction, street_name_direction_affix_type = extract_street_name_direction(street_name)
    street_name_stem, street_name_type, street_name_type_affix_type = extract_street_name_type(street_name, variation_to_standard_street_name_type_mapping)
    if street_name_stem == '': 
        # handle case such as "west street" or "avenue w"
        street_name_stem = street_name_direction[0] if (street_name_type == 'av' and street_name_type_affix_type == 'prefix' and street_name_direction_affix_type == 'suffix') else street_name_direction
        street_name_direction, street_name_direction_affix_type = None, None
    if street_name_stem is not None:
        street_name_stem = re.sub(r'(\d+)(st|nd|rd|th)',r'\1',street_name_stem)
        street_name_stem = parse_number_text_with_memory(street_name_stem)
    return street_name_direction, street_name_direction_affix_type, street_name_stem, street_name_type, street_name_type_affix_type

def reconstruct_street_name(street_name_direction, street_name_direction_affix_type, street_name_stem, street_name_type, street_name_type_affix_type):
    output = street_name_stem if isinstance(street_name_stem,str) else ''
    if street_name_type_affix_type == 'prefix':
        output =  f'{street_name_type} {output}'
    elif street_name_type_affix_type == 'suffix':
        output =  f'{output} {street_name_type}'
    if street_name_direction_affix_type == 'prefix':
        output =  f'{street_name_direction[0]} {output}'
    elif street_name_direction_affix_type == 'suffix':
        output =  f'{output} {street_name_direction[0]}'
    return output.strip()

##########################################################################################

from thefuzz import fuzz
import jellyfish
nysiis = jellyfish.nysiis # 'JALYF'
def get_nysiss_code(x):
    x = ' '.join([(w if w.isnumeric() else nysiis(w)) for w in x.split()]) if isinstance(x,str) else np.nan
    x = np.nan if not isinstance(x,str) or (len(x)==1 and not x.isnumeric()) else x
    return x

def propose_ybs1_id(year, borough, stname, street_index_df, mininum_match_score = 65, maximum_gap_for_cut_off = 10, verbose = False):
    if not isinstance(stname,str) or stname.strip()=='':
        return []
    street_name_direction, street_name_direction_affix_type, street_name_stem, street_name_type, street_name_type_affix_type = parse_street_name(stname)
    street_name_stem_nysiis = get_nysiss_code(street_name_stem)
    index_df = street_index_df.query('year == @year & borough == @borough').copy()
    if street_name_type is not None:
        index_df = index_df.query('street_name_type == @street_name_type')
    if len(index_df)==0:
        return []
    if not verbose and stname in index_df['stname'].tolist():
        return index_df.query('stname == @stname')['ybs1_id'].tolist()
    
    index_df['street_name_direction'+'_matched'] = index_df['street_name_direction'].apply(lambda x: x==street_name_direction and street_name_direction is not None)
    index_df['street_name_direction_affix_type'+'_matched'] = index_df['street_name_direction_affix_type'].apply(lambda x: x==street_name_direction_affix_type and street_name_direction_affix_type is not None)
    index_df['street_name_stem'+'_matched'] = index_df['street_name_stem'].apply(lambda x: x==street_name_stem and street_name_stem is not None)
    index_df['street_name_type'+'_matched'] = index_df['street_name_type'].apply(lambda x: x==street_name_type and street_name_type is not None)
    index_df['street_name_type_affix_type'+'_matched'] = index_df['street_name_type_affix_type'].apply(lambda x: x==street_name_type_affix_type and street_name_type_affix_type is not None)
    index_df['street_name_stem_nysiis'+'_matched'] = index_df['street_name_stem_nysiis'].apply(lambda x: x==street_name_stem_nysiis and street_name_stem_nysiis is not None)    
    index_df['street_name_stem_ratio'] = index_df['street_name_stem'].apply(lambda x: round(fuzz.ratio(x, street_name_stem)/100, 2) if isinstance(street_name_stem, str) and not street_name_stem.isnumeric() else (1 if x==street_name_stem else 0))
    index_df['match_score'] = index_df['street_name_stem_ratio'] * 100 + \
                            index_df['street_name_stem_nysiis_matched'] * 20 + \
                            index_df['street_name_direction_matched'] * 15 + \
                            index_df['street_name_type_affix_type_matched'] * 10 + \
                            index_df['street_name_direction_affix_type_matched'] * 5
    index_df['match_score'] = (index_df['match_score']/1.5).astype(int)
    index_df = index_df.sort_values('match_score', ascending=False)
    index_df = index_df.query('match_score >= @mininum_match_score')
    index_df['match_rank'] = index_df['match_score'].rank(method='dense', ascending=False).astype(int)
    if len(index_df) > 1:
        cut_off_rank = (index_df.drop_duplicates('match_rank').set_index('match_rank')['match_score'].diff()<=-maximum_gap_for_cut_off).idxmax()
        if cut_off_rank > 1:
            index_df = index_df[index_df['match_rank']<cut_off_rank]
    index_df = index_df.reset_index(drop=True)[['ybs1_id','year','borough','stname','match_score','match_rank']]
    if verbose:
        return index_df
    return index_df.query('match_rank == 1')['ybs1_id'].tolist()

##########################################################################################

from pyproj import Transformer
projection_transformer = Transformer.from_crs(2263, 4326)

def get_vector_direction(vector, rounding = 1):
  """np_arctan2_in_degree (-180 to 180 reference angle is the positive direction of x axis in cartesian space)""" 
  x, y = vector
  return np.round(np.arctan2(y, x) * 180 / np.pi, rounding)

def add_degree_to_azimuth(current, change):
    assert(abs(change)<180)
    output = current+change
    if output<-180:
        output = 180-(-output-180)
    elif output>180:
        output = -180+(output-180)
    if output == -180:
        output = -output
    return output

def get_unit_vector(angle):
    rad = np.deg2rad(angle)
    return np.array([np.cos(rad), np.sin(rad)])

def get_hnumber_street_side(hnumber, odd_on):
    is_odd = hnumber%2==1
    street_side = 'left' if (odd_on=='left' and is_odd) or (odd_on=='right' and not is_odd) else 'right'
    street_side = 'left' if odd_on == 'all_on_left' else 'right' if odd_on == 'all_on_right' else street_side
    return street_side

def get_hnumber_coordinates_from_street_details(hnumber, start_end_hnumbers, start_end_coordinates, segment_direction, street_side, offset_from_road_center):
    low, high = start_end_hnumbers
    if (high - low)==0:
        street_center_position = np.mean(start_end_coordinates,axis=0).tolist()
    else:
        f_pt_proportion = (hnumber - low)/(high - low)
        t_pt_proportion = 1 - f_pt_proportion
        street_center_position = np.average(np.array(start_end_coordinates), weights = (f_pt_proportion, t_pt_proportion), axis=0).tolist()
    offset_direction = add_degree_to_azimuth(segment_direction,-90) if street_side == 'left' else add_degree_to_azimuth(segment_direction,90)
    target_position = street_center_position + offset_from_road_center * get_unit_vector(offset_direction)
    return target_position

def convert_nys_to_wgs(x, y):
    lon, lat = np.round(projection_transformer.transform(x,y)[::-1],6)
    return lon, lat

def get_hnumber_coordinate(hnumber, ybs1_id, street_range_df):
    
    this_street_range_df = street_range_df.query('ybs1_id == @ybs1_id').copy()
    if len(this_street_range_df )==0:
        return None

    most_common_odd_on = this_street_range_df['odd_on'].str.replace('all_on_','').value_counts().index.tolist()[0]

    street_side = get_hnumber_street_side(hnumber, most_common_odd_on)

    row = this_street_range_df.query(f'{street_side}_low <= @hnumber & @hnumber <= {street_side}_high').head(1).reset_index(drop=True).T.to_dict()
    if len(row)==0:
        return None
    else:
        row = row[0]

    this_segment_direction = row['segment_direction']
    intersection_offset = 36
    block_edge_start_end_coordinates = (row['start_end_coordinates'][0] + intersection_offset*get_unit_vector(this_segment_direction), row['start_end_coordinates'][1] - intersection_offset*get_unit_vector(this_segment_direction))

    xy_coordinate = get_hnumber_coordinates_from_street_details(hnumber, (row[f'{street_side}_low'], row[f'{street_side}_high']), block_edge_start_end_coordinates, this_segment_direction, street_side, row['offset_from_road_center'])

    lon, lat = convert_nys_to_wgs(* xy_coordinate)

    return (lon, lat)

##########################################################################################

def get_borough_code(borough_name):
    if borough_name in range(1,5+1):
        return borough_name
    borough_name = borough_name.lower().strip()
    borough_name_to_code_mapping = {'manhattan':1,'bronx':2,'brooklyn':3,'queens':4,'staten island':5,
                                    'mn':1,       'bx':2,   'bk':3,      'qn':4,    'si':5}
    borough_code = borough_name_to_code_mapping.get(borough_name, None)
    if borough_code is None:
        raise Exception('Borough not found. Please choose one from manhattan, bronx, brooklyn, queens, staten island.')
    else:
        return borough_code

def get_closest_years(input_year, difference_threshold = 40):
    output_year_list = sorted([1850,1880,1910,1940], key=lambda x: abs(x-input_year))
    output_year_list = [x for x in output_year_list if abs(x-input_year)<=difference_threshold]
    return output_year_list

def standardize_house_number_part_within_address(x):
    x = re.sub(r'\s+',' ',x)
    x = re.sub(r'^(\d+)( |-)1/2',r'\1', x)
    x = re.sub(r'^(\d+)[a|b|r]?(?: )?[\-|\/|\&](?: )?(\d+)[a|b|r]? ',r'\1-\2 ', x)
    x = re.sub(r'^(\d+)[a|b|r]? ',r'\1 ', x)
    return x

def parse_hnumber_and_street_name(x):
    hnumber_part = re.findall(r'^([\d\-]+) ',x)
    hnumber_part = hnumber_part[0] if len(hnumber_part)>0 else ''
    street_name_part = x[len(hnumber_part):].strip()
    return hnumber_part, street_name_part

def extract_house_number_from_house_number_part(hnumber_part, choice_for_range = 'first'):
    if hnumber_part == '':
        return None
    if '-' in hnumber_part:
        hnumber, second_hnumber = [int(x) for x in hnumber_part.split('-')] 
        if choice_for_range == 'first':
            hnumber = hnumber
        elif choice_for_range == 'average':
            hnumber = (hnumber + second_hnumber)//2
        else:
            raise Exception('Invalid choice for handling range. Please choose from "first" or "average" (house number).')
    else:
        hnumber = int(hnumber_part)
    return hnumber

def geocode(input_address, input_borough, input_year, coordinate_only = False):
    # basic clean address
    input_address = deaccent_lowercase_remove_most_special(input_address)
    # parse address
    house_number_part, street_name_part = parse_hnumber_and_street_name(standardize_house_number_part_within_address(input_address))
    # get house number
    hnumber = extract_house_number_from_house_number_part(house_number_part)
    # get street name
    street_name = reconstruct_street_name(* parse_street_name(street_name_part))
    # identify borough
    borough_code = get_borough_code(input_borough)
    # identify year
    ybs1_id_list = []
    street_matched = False
    for matched_year in get_closest_years(input_year):
        ybs1_id_list = propose_ybs1_id(matched_year, borough_code, street_name, street_index_df)
        street_matched = len(ybs1_id_list)>0
        if street_matched:
            if abs(matched_year - input_year)>10:
                print('Please note that the best street data we can find for this address is more than 10 years apart from the year you provided. Use caution when interpreting results.')
            break    
    results = []
    if street_matched:
        # try all matched street
        for ybs1_id in ybs1_id_list:
            coordinate = get_hnumber_coordinate(hnumber, ybs1_id, street_range_df)
            if coordinate is not None:
                results.append((matched_year, borough_code, ybs1_id_to_stname_mapping[ybs1_id], hnumber, coordinate))
    if coordinate_only:
        results = [res[-1] for res in results]
    return results

##########################################################################################

from IPython.display import clear_output
clear_output()
print("\nThe geocoder is ready to use. Try running this line of code below:\n\ngeocode('23-25 catharine street', 'manhattan', 1910)\n\n * The input fields are: address, borough, year\n * The output fields are: year_matched, borough_code, standardized_street_name, standardized_house_number, lonlat_coordinates\n\nYou can also limit the output to coordinates only. For example:\n\ngeocode('23-25 catharine street', 'manhattan', 1910, coordinate_only=True)\n\nThis historical geocoder is developed as part of the Mapping Historical New York project (https://mappinghny.com) at Columbia University.\nFor more questions about the geocoder, please contact Tim Wu at gw2415@columbia.edu.\n\n")
