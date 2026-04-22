# Odhad neznámého stavu pomocí faktorových grafů
**Zdrojové kódy k diplomové práci**

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023+-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![GTSAM](https://img.shields.io/badge/GTSAM-4.2+-red.svg)](https://gtsam.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tento repozitář obsahuje softwarovou implementaci algoritmů vytvořených v rámci diplomové práce zaměřené na odhad stavu dynamických systémů pomocí **faktorových grafů (Factor Graphs)** a jejich přímé porovnání s tradičními i pokročilými metodami Kalmanovy filtrace a vyhlazování (**EKF, IEKF, UKF, ERTSS**).

## 📌 O projektu

Cílem práce a přiloženého softwaru je demonstrovat výhody dávkové optimalizace a iterativní relinearizace v nelineárních úlohách s omezenou pozorovatelností. Algoritmy jsou implementovány ve dvou prostředích a testovány na třech scénářích s rostoucí složitostí:

1. **Lineární systém (LS)** – Verifikace matematické ekvivalence faktorových grafů a optimálního vyhlazovače RTSS.
2. **Systém s mírnou nelinearitou (NS)** – Sledování skrytého parametru a demonstrace řešení problému nepřesné počáteční podmínky.
3. **Terénní navigace (TAN - Terrain Aided Navigation)** – Silně nekonvexní problém map-matchingu nad digitálním modelem elevace (DEM), testovaný při degradaci senzoru rychlosti. Demonstruje nasazení faktorového grafu v režimu klouzavého okna (Sliding Window) a analýzu statistické konzistence (ANEES).

### Zahrnuté implementace:
* **MATLAB**: Vlastní implementace řešiče faktorových grafů pro řídké matice (Batch i Sliding Window režim), sada implementovaných Kalmanových filtrů.
* **C++**: Průmyslová implementace dávkové optimalizace pomocí open-source knihovny **GTSAM**.

---

## Struktura repozitáře

```text
DP_factor_graphs/
│
├── MATLAB/                 # Skripty a třídy pro prostředí MATLAB
│   ├── main_LS.m           # Experiment 1: Lineární systém
│   ├── main_NS.m           # Experiment 2: Mírně nelineární systém
│   ├── main_MAPA.m         # Experiment 3: Terénní navigace
│   ├── FactorGraphSolver.m # Třída vlastního řešiče FG
│   └── TrajectoryFilters.m # Třída s filtry (EKF, UKF, IEKF, ERTSS)
│
├── CPP_GTSAM/              # Zdrojové kódy v C++ (GTSAM)
│   ├── CMakeLists.txt      # Konfigurační soubor pro sestavení
│   ├── GNSS_X.csv, GNSS_Y.csv        # Referenční trajektorie
│   ├── main_mc.cpp         # Zdrojový kód pro Monte Carlo simulace 
│   ├── hB.csv                        # Data měření
│   ├── mapX.csv, mapY.csv, mapZ.csv  # Digitální model terénu (DEM)
│   └── MapUtils.h          # Nástroje pro práci s mapou a výpočet gradientů
    
