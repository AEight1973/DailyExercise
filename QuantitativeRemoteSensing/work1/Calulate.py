import ReadFile
import pandas as pd

wave1 = [620, 670]
wave2 = [841, 876]

leaf, avhrr, modis = ReadFile.read()

leaf_wave= dict()
for i in range(len(leaf)):
    leaf_wave[leaf.iloc[i, 0]] = leaf.iloc[i, 1]

modis_wave1, modis_wave2, avhrr_wave1, avhrr_wave2 = 0, 0, 0, 0

count1, count2 = 0, 0
for i in range(len(avhrr)):

    if wave1[0] <= avhrr.iloc[i, 0] <= wave1[1]:
        avhrr_wave1 += avhrr.iloc[i, 1]*leaf_wave[avhrr.iloc[i, 0]]
        count1 += 1
    elif wave2[0] <= avhrr.iloc[i, 0] <= wave2[1]:
        avhrr_wave2 += avhrr.iloc[i, 2] * leaf_wave[avhrr.iloc[i, 0]]
        count2 += 1
avhrr_wave1 /= count1
avhrr_wave2 /= count2
avhrr_ndvi = (avhrr_wave2 - avhrr_wave1) / (avhrr_wave2 + avhrr_wave1)

count1, count2 = 0, 0
for i in range(len(modis)):
    if wave1[0] <= modis.iloc[i, 0] <= wave1[1]:
        modis_wave1 += modis.iloc[i, 1]*leaf_wave[modis.iloc[i, 0]]
        count1 += 1
    elif wave2[0] <= modis.iloc[i, 0] <= wave2[1]:
        modis_wave2 += modis.iloc[i, 2] * leaf_wave[modis.iloc[i, 0]]
        count2 += 1
modis_wave1 /= count1
modis_wave2 /= count2
modis_ndvi = (modis_wave2 - modis_wave1) / (modis_wave2 + modis_wave1)
