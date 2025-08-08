import numpy as np
import pytest
import numpy as np

from phllm.extract.chunkers import altered_n_select

# ---------- Test Cases ----------
CASES = {}

oner1 = ['AAAAA', 'BBB', 'CCCC']
oner2 = ['AA', 'B', 'CCCCC']
case_one = {
    'strain1': oner1, 
    'strain2': oner2
}
answer_one = np.array([
    oner1, 
    oner2
], dtype=object)

twor1 = ['AAAAA', None, 'CCCC']
twor2 = ['AA', 'B', '']
case_two = {
    'strain1': twor1, 
    'strain2': twor2
}
answer_two = np.array([
    ['AAAAA', '', 'CCCC'], 
    twor2
], dtype=object)

case_three = { # testing subchunking: no overlap with context window 5
    'strain1': ['AAAAA', 'B'*10, 'CCCC'], 
    'strain2': ['AA', 'B', 'C'*15]
}
answer_three = np.array([
    ['AAAAA', 'B'*5, 'B'*5, 'CCCC'],
    ['AA', 'B'] + ['C'*5 for i in range(3)]
], dtype=object)

case_four = { # testing padding
    'strain1': ['AAAAA', 'B'*3, 'CCCC'], 
    'strain2': ['AA', 'B', 'C'*4, 'D'*2], 
    'strain3': ['AA', 'B', 'C'*5, 'D'*5, 'E'*5, 'F'*3]
}
answer_four = np.array([
    ['AAAAA', 'B'*3, 'CCCC'],
    ['AA', 'B', 'C'*4, 'D'*2],
    ['AA', 'B', 'C'*5, 'D'*5, 'E'*5, 'F'*3]
], dtype=object)

case_five = { # testing padding and subchunking: no overlap with context window 5
    'strain1': ['AAAAA', 'B'*30, 'CCCC'], 
    'strain2': ['AA', 'B', 'C'*40, 'D'*20], 
    'strain3': ['AA', 'B', 'C'*50, 'D'*35, 'E'*15, 'F'*35]
}
answer_five = np.array([
    ['AAAAA'] + ['B'*5 for i in range(6)] + ['CCCC'],
    ['AA', 'B'] + ['C'*5 for i in range(8)] + ['D'*5 for i in range(4)],
    ['AA', 'B'] 
    + ['C'*5 for _ in range(10)] 
    + ['D'*5 for _ in range(7)] 
    + ['E'*5 for _ in range(3)] 
    + ['F'*5 for _ in range(7)]
], dtype=object)

CASES = {
    1: {
        'PROB': case_one,
        'ANS': answer_one
    },
    2: {
        'PROB': case_two,
        'ANS': answer_two
    },
    3: {
        'PROB': case_three,
        'ANS': answer_three
    },
    4: {
        'PROB': case_four,
        'ANS': answer_four
    },
    5: {
        'PROB': case_five,
        'ANS': answer_five
    }
}

# ---------- Helper functions to test ----------
def get_case(num, return_val):
    """Gets the specified case from the case dictionary above. `problem_or_answer` takes in a string designating either 'prob' or 'ans' to be returned."""
    CONF = return_val.upper()
    assert 'PROB' == CONF or 'ANS' == CONF, f"Unsupported return value, input either 'prob' or 'ans' currently {return_val}"
    assert num in CASES, f"Unsupported case number, choose from {list(CASES.keys())}, currently {num}"
    
    return CASES[num].get(return_val)

# ---------- Tests ----------
@pytest.mark.parametrize("case_name, case", CASES.items())
def test_altered_n_select(case_name, case):
    prob = case['PROB']
    expected = case['ANS']

    out, pads_per_val, pad_starts = altered_n_select(
        d=prob,
        n=5,
        overlap_proportion=0.0,
        rand_score=0.0,
        rt_array=True
    )

    assert np.array_equal(out, expected), (
        f"Case {case_name} failed:\nExpected:\n{expected}\nGot:\n{out}"
    )

# Run tests
if __name__ == "__main__":
    for case_name, case in CASES.items():
        test_altered_n_select(case_name, case)