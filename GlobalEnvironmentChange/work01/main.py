import ReadFile
import pandas as pd

[data1, data2] = ReadFile.read()
start_year = 1959
end_year = 2020
year = list(range(start_year, end_year + 1))

_year = start_year

CO2 = []
_value = 0
for i in range(len(data1)):
    if _year == end_year + 1:
        break
    elif int(data1.loc[i, 'year']) == _year:
        _value += float(data1.loc[i, 'de-season alized'])
    elif int(data1.loc[i, 'year']) > _year:
        _year += 1
        CO2.append(_value / 12)
        _value = float(data1.loc[i, 'de-season alized'])
    else:
        continue

temp = [float(i) for i in data2.loc[79: 141, 'Lowess(5)']]

co2_temp = pd.DataFrame(index=year)
co2_temp['co2'] = CO2
co2_temp['temp'] = temp

print(co2_temp.corr('pearson'))
