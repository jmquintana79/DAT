
# test main library import
def test_import_lib():
    try:
        import DAT
        is_done = True
    except:
        is_done = False
    assert is_done

# test eda module import
def test_import_eda():
    try:
        from DAT import EDA
        is_done = True
    except:
        is_done = False
    assert is_done


# test plot module import
def test_import_plot():
    try:
        from DAT import PLOTS
        is_done = True
    except:
        is_done = False
    assert is_done
