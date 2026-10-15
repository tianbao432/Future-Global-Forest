Forest Type Data Preprocessing
This repository prepares 0.5° forest-type fractions and annual forest land-cover fractions (1700–2100) for LPJ-GUESS.
1. ESACC_forest_type_fractions_0p5deg.py
Extracts the relative proportions of needleleaf (ND), broadleaf (BD), and mixed (MX) forests for each 0.5° grid cell from the ESA CCI 2020 land-cover map.
Output: forest_type_fractions_0p5deg_2020.txt, used as input to the next step.
2. Tiabnbao_ST_PFT.py
Matches the 2020 forest-type fractions with ForestST_agegroups_1700_2100_15.txt by geographic coordinates to produce annual forest-type fractions for 1700–2100.
The allocation follows:
FOREST = sum of the 15 ForestST fractions
NATURAL = 1 - FOREST
Type fraction = FOREST × fixed 2020 type weight
ND + BD + MX + UNK = FOREST
Grid cells with all ND/BD/MX fractions missing or zero are assigned to the unknown type (UNK), without nearest-neighbor imputation.
Output: PFT/Tianbao_ST_PFT_1700_2100.txt
Columns: lon, lat, year, FOREST, NATURAL, ND, BD, MX, UNK.
Usage
Install the required Python packages (numpy, pandas, and rasterio), configure input/output paths as needed, and run:
python ESACC_forest_type_fractions_0p5deg.py
python Tiabnbao_ST_PFT.py
Note: Forest-type composition is held fixed at its 2020 proportions. Annual variation reflects changes in the total ForestST forest fraction, not reconstructed changes in forest-type composition.
