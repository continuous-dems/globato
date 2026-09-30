from globato.streams.readers.bag import identify_unregistered_vertical

# CRS of NOS BAGs as rasterio reports them (H11951_MB_8m_MLLW_4of5,
# D00228_MB_VR_MLLW and H11951_MB_1m_MLLW_1of5).
NAMED_ONLY_MLLW_DEPTH = (
    'COMPD_CS["NAD83 / UTM zone 10N + MLLW depth",PROJCS["NAD83 / UTM zone '
    '10N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",637813'
    '7,298.257222101004,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["E'
    'PSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree '
    '(supplier to define representation)",0.0174532925199433,AUTHORITY["EPSG","9122"]'
    '],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitud'
    'e_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.99'
    '96],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre"'
    ',1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORI'
    'TY["EPSG","26910"]],VERT_CS["MLLW depth",VERT_DATUM["MLLW '
    'depth",2005],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Up",UP]]]'
)

NAMED_ONLY_MEAN_LOWER_LOW_WATER = (
    'COMPD_CS["NAD83 / UTM zone 11N + Mean Lower Low Water",PROJCS["NAD83 / UTM zone '
    '11N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",637813'
    '7,298.257222101004,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["E'
    'PSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree '
    '(supplier to define representation)",0.0174532925199433,AUTHORITY["EPSG","9122"]'
    '],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitud'
    'e_of_origin",0],PARAMETER["central_meridian",-117],PARAMETER["scale_factor",0.99'
    '96],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre"'
    ',1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORI'
    'TY["EPSG","26911"]],VERT_CS["Mean Lower Low Water",VERT_DATUM["Mean Lower Low '
    'Water",2000],UNIT["metre",1],AXIS["Depth",DOWN]]]'
)

DECLARED_EPSG_5866 = (
    'COMPD_CS["NAD83 / UTM zone 10N + MLLW depth",PROJCS["NAD83 / UTM zone '
    '10N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",637813'
    '7,298.257222101004,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["E'
    'PSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree '
    '(supplier to define representation)",0.0174532925199433,AUTHORITY["EPSG","9122"]'
    '],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitud'
    'e_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.99'
    '96],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre"'
    ',1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORI'
    'TY["EPSG","26910"]],VERT_CS["MLLW depth",VERT_DATUM["Mean Lower Low Water",2005,'
    'AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",'
    'DOWN],AUTHORITY["EPSG","5866"]]]'
)


def test_vertical_datum_named_but_not_registered():
    assert (
        identify_unregistered_vertical(NAMED_ONLY_MLLW_DEPTH)
        == "EPSG:26910+vdatum:mllw"
    )
    assert (
        identify_unregistered_vertical(NAMED_ONLY_MEAN_LOWER_LOW_WATER)
        == "EPSG:26911+vdatum:mllw"
    )


def test_registered_vertical_datum_is_left_alone():
    assert identify_unregistered_vertical(DECLARED_EPSG_5866) is None


def test_not_compound_or_unparseable():
    assert identify_unregistered_vertical("EPSG:26911") is None
    assert identify_unregistered_vertical("not a crs") is None
