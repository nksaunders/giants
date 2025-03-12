from giants.target import Target

def test_target_init():
    target = Target(97766057)
    assert target.has_target_info
    assert target.ticid == 97766057
    assert round(target.ra) == 252
    assert round(target.dec) == -32
    assert target.available_sectors[0] == 12

def test_custom_target_info():
    target_info = {
        'ra':252.047938250953,
        'dec':-32.4336794894763,
        'rstar':2.06072,
        'mstar':None,
        'teff':6380,
        'logg':3.91383,
        'vmag':12.661
    }

    target = Target(97766057, target_info=target_info)
    assert target.has_target_info
    assert target.ticid == 97766057
    assert round(target.ra) == 252
    assert round(target.dec) == -32
    assert target.available_sectors[0] == 12