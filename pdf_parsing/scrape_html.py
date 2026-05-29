import glob
import os
import os.path
import re
from copy import deepcopy
from uuid import uuid4

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import numpy as np

strip_n_re = re.compile(r'\n')

def find_class(element):
    if element.has_key('class'):
        result = element['class']
    else:
        result = []

    if result != []:
        return result
    elif element.name != '[document]':
        return find_class(element.parent)
    else:
        return ''

def find_in_style(element,attribute):
    re_str = "(?:" + re.escape(attribute) + "\:)([^;]+)"
    if attribute == 'font-style':
        if element.name.lower() == 'u':
            return 'underlined'
    if attribute == 'font-weight':
        if element.name.lower() == 'b':
            return 'bold'

    if element.has_key('style'):
        result = re.findall(re_str,element['style'])
    else:
        result = []


    if result != []:
        return result[0].strip()
    elif element.name != '[document]':
        return find_in_style(element.parent,attribute)
    else:
        return ''

def get_rules(attribute,soup,rules=None):
    if rules is None:
        rules ={}
    split_rules = re.split(r'((?:\.|\s)\w{1,3}[0-9]{1,2}\s?\{)', str(soup.style.string))
    if attribute == 'font-size':
        attribute_re = re.compile(r"(?:font:\s(?:italic bold|bold|italic)?\s?)([^']+)",re.IGNORECASE)
    elif attribute == 'font-weight':
        attribute_re = re.compile(r"(?:font:\s)(?:italic\s)?(bold)")
    elif attribute == 'font-style':
        attribute_re = re.compile(r"(?:font:\s)(italic)")
    else:
        attribute_re = re.compile("(?:" + re.escape(attribute) + "\:)([^;]+)")

    for i in range(1, len(split_rules), 2):
        name = split_rules[i][1:-1].strip()
        tag = split_rules[i+1].strip()[:-1]
        results = attribute_re.findall(tag)
        if len(results) > 0:
            result = results[0]
            result = result.strip()
            rules[name] = result
    return rules

def find_in_style_and_add_rules(element,attribute,rules):
    result = find_in_style(element, attribute)
    if result == '' and element.has_key('class'):
        try:
            result = next(rules[c] for c in element['class'] if c in rules.keys())
        except StopIteration:
            pass
    if result == '':
        # if it is a string
        if element.name in rules.keys():
            result = rules[element.name]
        else:
            # if it is a tuple
            try:
                result = next(rules[c] for c in element.name if c in rules.keys())
            except StopIteration:
                pass
    return result


def find_in_style_and_add_rules_hyrarchical(element, attribute, rules):
    """ find style attributes in the HTML and look in the parents if it
        is not found. Contrary to the other function, this function also
        looks for the rules in the parents
    """
    re_str = "(?:" + re.escape(attribute) + "\:)([^;]+)"
    if attribute == 'font-style':
        if element.name.lower() == 'u':
            return 'underlined'
    if attribute == 'font-weight':
        if element.name.lower() == 'b':
            return 'bold'

    if element.has_key('style'):
        result_list = re.findall(re_str, element['style'])
    else:
        result_list = []

    if result_list != []:
        return result_list[0].strip()

    result = ''
    if element.has_key('class'):
        try:
            result = next(rules[c] for c in element['class'] if c in rules.keys())
        except StopIteration:
            pass
    if result == '':
        if element.name in rules.keys():
            result = rules[element.name]
        elif isinstance(element.name,list):
            try:
                result = next(rules[c] for c in element.name if c in rules.keys())
            except StopIteration:
                pass
    if result == '':
        result = find_in_style_and_add_rules(element.parent, attribute, rules)
    return result

def children_to_dict(element, parent_ids=None, parent_names=None, font_size_rules=None, font_weight_rules=None,
                     font_style_rules=None):
    if parent_ids is None:
        parent_ids = []
    if parent_names is None:
        parent_names = []
    element_id = str(uuid4())
    name = element.name
    all_parent_ids = [element_id] + parent_ids
    all_parent_names = [name] + parent_names
    is_single = len([e for e in element.children]) == 1
    text = ''
    if is_single:
        child = next(element.children)
        if isinstance(child, NavigableString):
            text = str(child)
    text = strip_n_re.sub('',text).strip()
    if text == '':
        text_list = element.find_all(text=True, recursive=False)
        text = ' '.join(text_list)
    text = strip_n_re.sub('',text).strip()
    child_dicts = []
    for child in element.children:
        if isinstance(child,NavigableString):
            continue
        child_dict = children_to_dict(child,all_parent_ids,all_parent_names,
                                      font_size_rules,font_weight_rules,font_style_rules)
        if child_dict['text']!='' or child_dict['children']!= []:
            child_dicts.append(child_dict)
    element_class = find_class(element)
    if text != '':
        font_size = find_in_style_and_add_rules_hyrarchical(element, 'font-size', font_size_rules)
        font_weight = find_in_style_and_add_rules_hyrarchical(element, 'font-weight', font_weight_rules)
        font_style = find_in_style_and_add_rules_hyrarchical(element, 'font-style', font_style_rules)
    else:
        font_size = ''
        font_weight = ''
        font_style = ''

    out_dict = {'text':text,
                'class':', '.join(element_class),
                'fontSize':font_size,
                'fontStyle':font_style,
                'fontWeight':font_weight,
                'name':name,
                'id':element_id,
                'parentList': parent_ids,
                'parentNames': parent_names,
                'children':child_dicts
                }
    return out_dict

def flatten(in_dicts):
    out_dicts = []
    for in_dict in in_dicts:
        out_dict = deepcopy(in_dict)
        del out_dict['children']
        out_dicts.append(out_dict)
        out_dicts = out_dicts + flatten(in_dict['children'])
    return out_dicts


def scrape_html(in_files):
    if not isinstance(in_files,list):
        in_files = [in_files]
    flat_dfs = []
    for in_file in in_files:
        with open(in_file,encoding="utf8") as f:
            contents = f.read()

        soup = BeautifulSoup(contents, 'html.parser')
        body = soup.find('body')
        font_size_rules = {'h0':'40pt',
                           'h1':'32pt',
                           'h2':'26pt',
                           'h3':'22pt',
                           'h4':'20pt',
                           'p': '11pt',
                           'u': '12pt'}
        font_style_rules = {}
        font_weight_rules = {}
        font_size_rules = get_rules('font-size',soup,font_size_rules)
        font_weight_rules = get_rules('font-weight',soup,font_weight_rules)
        font_style_rules = get_rules('font-style',soup,font_style_rules)

        file_root, _ = os.path.splitext(os.path.basename(in_file))

        all_dict = children_to_dict(body,
                                    font_size_rules=font_size_rules,
                                    font_weight_rules=font_weight_rules,
                                    font_style_rules=font_style_rules
                                    )
        flat_dfs.append(pd.DataFrame(flatten([all_dict])))
    flat_df = pd.concat(flat_dfs)
    return flat_df
