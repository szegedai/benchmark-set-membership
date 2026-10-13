###=============================================================================
### Imports
###=============================================================================

import string
from collections import defaultdict
from itertools import product

###=============================================================================
### Features
###=============================================================================

def get_feature_ids(feature, template):
    feature_ids = dict()
    for fname, fval in feature.items():
        feature_ids[fname] = fval['id']
    if 'features' in template:
        for fname, fval in template['features'].items():
            feature_ids[fname] = fval['id']
    return feature_ids

def get_feature_grid(features):
    grid = product(*features.values())
    named_grid = [dict(zip(features.keys(), f)) for f in grid]
    return named_grid


def get_feature_names(features, templates=None):
    feature_names = list(features.keys())
    # assume that every template has the same features
    if templates is not None and 'features' in templates[0]:
        feature_names += list(templates[0]['features'].keys())
    return feature_names


def get_feature_id_mapping(features, templates=None):
    mapping = defaultdict(dict)
    for fn, values in features.items():
        for v in values:
            fid = v['id']
            fval = v['to_str']
            mapping[fn][fid] = fval
    if 'features' in templates[0]:
        for template in templates:
            for fn, v in template['features'].items():
                fid = v['id']
                fval = v['to_str']
                mapping[fn][fid] = fval

    return mapping

###=============================================================================
### Prompts: Common
###=============================================================================

class FormatDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def apply_feature_quotation(x, feature):
    q = feature['type']
    if q:
        if isinstance(x, str):
            x = q + x + q
        elif isinstance(x, list):
            x = [q + e + q for e in x]
    return x

###=============================================================================
### Prompts: Natural
###=============================================================================

def create_prompt_natural(template, feature, elements, element):
    prompt = template['prompt']

    # set: element
    element = apply_feature_quotation(element, feature['quotation'])
    prompt = prompt.format_map(FormatDict(element=element))

    # set: elements
    elements = apply_feature_quotation(elements, feature['quotation'])
    elements = apply_feature_listing(elements, feature['listing'])
    prompt = prompt.format_map(FormatDict(elements=elements))

    # feature: element
    f_element = get_feature_element(feature['element'])
    prompt = prompt.format_map(FormatDict(f_element=f_element))

    if '{f_element_pl}' in prompt:
        f_element_pl = get_feature_element(feature['element'], use_plural=True)
        prompt = prompt.format_map(FormatDict(f_element_pl=f_element_pl))

    # feature: contain
    if template['use_infinitive']:
        f_contain = get_feature_contain(feature['contain'])
        prompt = prompt.format_map(FormatDict(f_contain=f_contain))
    else:
        f_contain_aux, f_contain = get_feature_contain(
            feature['contain'],
            use_infinitive=False
        )
        prompt = prompt.format_map(FormatDict(
            f_contain_aux=f_contain_aux,
            f_contain=f_contain
        ))

    if '{f_contain_3rd}' in prompt:
        f_contain_3rd = get_feature_contain(
            feature['contain'],
            use_third_person=True
        )
        prompt = prompt.format_map(FormatDict(f_contain_3rd=f_contain_3rd))

    # feature: set
    f_set = get_feature_set(feature['set'])
    prompt = prompt.format_map(FormatDict(f_set=f_set))

    # post-process
    prompt = fix_prompt(prompt)

    return prompt


def get_feature_element(feature, use_plural=False):
    synonym = feature['synonym']
    if use_plural:
        synonym += 's'
    else:
        synonym = feature['article'] + ' ' + synonym
    return synonym


def get_feature_contain(feature, use_infinitive=True, use_third_person=False):
    if use_infinitive:
        synonym = feature['infinitive']
        if use_third_person:
            synonym = synonym + 's'
        return synonym
    else:
        auxilary, synonym = feature['question']
        return auxilary, synonym


def get_feature_set(feature):
    return feature['synonym']


def apply_feature_listing(x, feature):
    if feature['is_bulletpoint']:
        bulletpoint = feature['type']
        delimiter = '\n' + bulletpoint + ' '
        return delimiter.join(['', *x]) + '\n'
    else:
        begin, end = feature['type']
        delimiter = ', '
        return begin + delimiter.join(x) + end


def fix_prompt(prompt):
    prompt = prompt[0].upper() + prompt[1:]
    prompt = prompt.replace('\n\n', '\n')
    prompt = prompt.replace(' \n', '\n')
    prompt = prompt.replace('\n ', '\n')
    return prompt

###=============================================================================
### Prompts: Code
###=============================================================================

def create_prompt_coding_simple(template, feature, elements, element):
    prompt = template['prompt']

    # feature: init
    f_init = get_feature_init(feature['init'])
    prompt = prompt.format_map(FormatDict(f_init=f_init))

    # set: elements
    if feature['init']['type'] != 'constructor-split':
        elements = apply_feature_quotation(elements, feature['quotation'])
        elements = ', '.join(elements)
    if feature['init']['type'] == 'constructor-split':
        elements = ' '.join(elements)
        elements = apply_feature_quotation(elements, feature['quotation'])
    prompt = prompt.format_map(FormatDict(elements=elements))

    # feature: result
    f_result_question, f_result_code = get_feature_result(feature['result'])
    prompt = prompt.format_map(FormatDict(
        f_result_question=f_result_question,
        f_result_code=f_result_code
    ))

    # feature: logic
    f_logic, f_logic_operation = get_feature_logic(
        feature['logic'],
        feature['arg_order']['order'],
        template
    )
    prompt = prompt.format_map(FormatDict(
        f_logic=f_logic,
        f_logic_operation=f_logic_operation
    ))

    # feature: variable
    f_variable = get_feature_variable(feature['variable'])
    prompt = prompt.format_map(FormatDict(f_variable=f_variable))

    # set: element
    element = apply_feature_quotation(element, feature['quotation'])
    prompt = prompt.format_map(FormatDict(element=element))

    # feature: argument order
    if feature['arg_order']['order'] == 'set_first':
        f_arg1, f_arg2 = f_variable, element
    else:
        f_arg1, f_arg2 = element, f_variable
    prompt = prompt.format_map(FormatDict(f_arg1=f_arg1))
    prompt = prompt.format_map(FormatDict(f_arg2=f_arg2))

    # feature: tab
    prompt = apply_feature_tab(feature['tab'], prompt)

    prompt = remove_tabs(prompt)

    return prompt


def create_prompt_coding_algorithmic(template, feature, elements, element):
    prompt = template['prompt']

    # feature: init
    f_init = get_feature_init(feature['init'])
    prompt = prompt.format_map(FormatDict(f_init=f_init))

    # set: elements
    if feature['init']['type'] != 'constructor-split':
        elements = apply_feature_quotation(elements, feature['quotation'])
        elements = ', '.join(elements)
    if feature['init']['type'] == 'constructor-split':
        elements = ' '.join(elements)
        elements = apply_feature_quotation(elements, feature['quotation'])
    prompt = prompt.format_map(FormatDict(elements=elements))

    # feature: result
    f_result_question, f_result_code = get_feature_result(feature['result'])
    if feature['logic']['type'] == 'main' and feature['result']['type'] == 'print':
        f_result_code = 'result = {f_logic_operation}\nprint(result)'
    prompt = prompt.format_map(FormatDict(
        f_result_question=f_result_question,
        f_result_code=f_result_code
    ))

    # feature: logic
    f_logic, f_logic_operation = get_feature_logic(
        feature['logic'],
        feature['arg_order']['order'],
        template
    )
    prompt = prompt.format_map(FormatDict(
        f_logic=f_logic,
        f_logic_operation=f_logic_operation
    ))

    # feature: variable
    f_variable = get_feature_variable(feature['variable'])
    prompt = prompt.format_map(FormatDict(f_variable=f_variable))

    # set: element
    element = apply_feature_quotation(element, feature['quotation'])
    prompt = prompt.format_map(FormatDict(element=element))

    # feature: argument order
    if feature['arg_order']['order'] == 'set_first':
        f_arg1, f_arg2 = f_variable, element
    else:
        f_arg1, f_arg2 = element, f_variable
    prompt = prompt.format_map(FormatDict(f_arg1=f_arg1))
    prompt = prompt.format_map(FormatDict(f_arg2=f_arg2))

    # feature: tab
    prompt = apply_feature_tab(feature['tab'], prompt)

    prompt = remove_tabs(prompt)

    return prompt


def get_feature_init(feature):
    return feature['code']


def get_feature_result(feature):
    return feature['question'], feature['code']


def get_feature_variable(feature):
    return feature['name']


def get_feature_logic(feature, order, template):
    if feature['type'] == 'main':
        return '', feature['operation']

    function = '\n'

    # imports
    if feature['imports']:
        function += feature['imports']
        function += '\n\n'

    # header
    function += 'def contains'
    if order == 'set_first':
        function += '(' + feature['arg_set'] + ', ' + feature['arg_item'] + ')'
    else:
        function += '(' + feature['arg_item'] + ', ' + feature['arg_set'] + ')'
    if feature['returns']:
        function += ' -> ' + feature['returns']
    function += ':'
    function += '\n'

    # docstring
    if feature['docstring']:
        docstring = tabulate_code(template['docstring'])
        if order == 'element_first':
            split_ds = docstring.split('\n')
            i = split_ds.index(TAB + 'Args:')
            split_ds[i+1], split_ds[i+2] = split_ds[i+2], split_ds[i+1]
            docstring = '\n'.join(split_ds)
        function += docstring
        function += '\n'

    # code
    function += tabulate_code(feature['code'])
    function += '\n'

    return function, feature['operation']


def apply_feature_tab(feature, prompt):
    if isinstance(feature['type'], int):
        tab = ''.join([' ' * feature['type']])
        return prompt.replace(TAB, tab)
    else:
        return prompt.replace(TAB, '\t')

###-----------------------------------------------------------------------------
### Utilities
###-----------------------------------------------------------------------------

TAB = '    '


def tabulate_code(code, tab=TAB):
    rows = code.split('\n')
    rows = [r if r == '\n' else tab + r for r in rows]
    return '\n'.join(rows)


def remove_tabs(prompt):
    rows = prompt.split('\n')
    for i in range(len(rows)):
        if rows[i] == '\t' or rows[i] == '  ' or rows[i] == '    ':
            rows[i] = ''
    return '\n'.join(rows)

###=============================================================================
### Answers
###=============================================================================

def create_set_of_words(s):
    punctuation_table = str.maketrans('', '', string.punctuation)

    prepared_s = s.translate(punctuation_table)
    prepared_s = prepared_s.lower()
    prepared_s = prepared_s.split()

    return set(prepared_s)
