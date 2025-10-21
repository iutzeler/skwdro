import pytest
import numpy as np
from skwdro.linear_models import LogisticRegression

def test_classes_nb():
    model = LogisticRegression()
    with pytest.raises(ValueError) as e:
        model.fit(np.random.randn(3, 2), [0, 0, 0])
    assert 'Found' in str(e.value)

def test_classes_type():
    model = LogisticRegression()
    with pytest.raises(ValueError) as e:
        model.fit(np.array([['x', 'y']]*3), [0, 1, 0])
    assert 'dtype' in str(e.value)
