import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': 'total_precipitation',
        'year': '2009',
        'month': '06',
        'day': '05',
        'time': [
            '08:00', '09:00', '10:00',
            '11:00', '12:00', '13:00',
            '14:00', '15:00', '16:00',
            '17:00', '18:00', '19:00',
            '20:00',
        ],
        'area': [
            33, 119, 29,
            123,
        ],
    },
    'download.grib')