import numpy as np
import scedar.eda as eda
import pytest


class TestMType(object):
    """docstring for TestMType"""

    def test_is_valid_sfid(self):
        assert eda.mtype.is_valid_sfid('1')
        assert eda.mtype.is_valid_sfid(1)
        assert not eda.mtype.is_valid_sfid(np.array([1])[0])
        assert not eda.mtype.is_valid_sfid([])
        assert not eda.mtype.is_valid_sfid([1])
        assert not eda.mtype.is_valid_sfid(None)
        assert not eda.mtype.is_valid_sfid((1,))

    def test_check_is_valid_sfids(self):
        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids(np.arange(5))

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids([True, False])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids(None)

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids([[1], [2]])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids(['1', 2, 3])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids(['1', '1', '3'])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids([0, 0, 1])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_sfids(['1', 2, '3'])

        eda.mtype.check_is_valid_sfids([1, 2])
        eda.mtype.check_is_valid_sfids(['1', '2'])
        eda.mtype.check_is_valid_sfids([1, 2, 3])

    def test_is_valid_lab(self):
        assert eda.mtype.is_valid_lab('1')
        assert eda.mtype.is_valid_lab(1)
        assert not eda.mtype.is_valid_lab(np.array([1])[0])
        assert not eda.mtype.is_valid_lab([])
        assert not eda.mtype.is_valid_lab([1])
        assert not eda.mtype.is_valid_lab(None)
        assert not eda.mtype.is_valid_lab((1,))

    def test_check_is_valid_labs(self):
        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_labs(np.arange(5))

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_labs([True, False])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_labs(None)

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_labs([[1], [2]])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_labs(['1', 2, 1])

        with pytest.raises(Exception) as excinfo:
            eda.mtype.check_is_valid_labs(['1', 2, '3'])

        eda.mtype.check_is_valid_labs([1, 2])
        eda.mtype.check_is_valid_labs([1, 1, 3])
        eda.mtype.check_is_valid_labs(['1', '2', '3'])
