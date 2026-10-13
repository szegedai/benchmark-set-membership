###=============================================================================
### Imports
###=============================================================================

from itertools import permutations

###=============================================================================
### Implementation
###=============================================================================

def generate_setinclusion_permutations(elements, intruder=None):
    all_sets = list()

    # included
    for e in elements:
        for elems in permutations(elements):
            e = str(e)
            elems = list(map(str, elems))
            all_sets.append({
                'elements': elems,
                'element': e,
                'groundtruth': True,
                'has_intruder': False
            })

    # not included
    for pemutation in permutations(elements):
        e = str(pemutation[0])
        elems = list(map(str, pemutation[1:]))
        all_sets.append({
            'elements': elems,
            'element': e,
            'groundtruth': False,
            'has_intruder': False
        })

    # intruder word
    if intruder is not None:
        for pemutation in permutations(elements):
            elems = list(map(str, pemutation[1:]))
            all_sets.append({
                'elements': elems,
                'element': str(intruder),
                'groundtruth': False,
                'has_intruder': str(intruder)
            })

    return all_sets
