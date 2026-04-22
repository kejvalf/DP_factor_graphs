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
3. **Terénní navigace (TAN - Terrain Aided Navigation)** – Silně nekonvexní problém map-matchingu nad digitálním modelem elevace, testovaný při degradaci senzoru rychlosti. Demonstruje nasazení faktorového grafu v režimu klouzavého okna (Sliding Window) a analýzu statistické konzistence (ANEES).

### Zahrnuté implementace:
* **MATLAB**: Vlastní implementace řešiče faktorových grafů pro řídké matice (Batch i Sliding Window režim), sada implementovaných Kalmanových filtrů.
* **C++**: Průmyslová implementace dávkové optimalizace pomocí open-source knihovny **GTSAM**.

---

## Struktura repozitáře

```text
DP_factor_graphs/
│
├── MATLAB/                 # Skripty a třídy pro prostředí MATLAB
│   ├── FG_estimation_LS.m           # Skript pro lineární systém
│   ├── FG_estimation_NS.m           # Skript pro mírně nelineární systém
│   ├── FG_estimation_window.m       # Skript pro window odhad FG - TAN
│   ├── FG_estimation_batch.m        # Skript pro batch odhad FG - TAN
│   ├── FactorGraphSolver.m          # Třída vlastního řešiče FG
│   └── TrajectoryFilters.m          # Třída s filtry (EKF, UKF, IEKF, ERTSS)
│
├── CPP_GTSAM/                        # Zdrojové kódy v C++ (GTSAM)
│   ├── CMakeLists.txt                # Konfigurační soubor pro sestavení
│   ├── GNSS_X.csv, GNSS_Y.csv        # Referenční trajektorie
│   ├── main.cpp                      # Zdrojový kód pro nelineární systém
│   ├── main_LS.cpp                   # Zdrojový kód pro lineární systém
│   ├── main_MC.cpp                   # Zdrojový kód pro Monte Carlo simulaci
│   ├── hB.csv                        # Data měření výškoměru
│   ├── mapX.csv, mapY.csv, mapZ.csv  # Digitální model terénu (DEM)
│   └── MapUtils.h                    # Nástroje pro práci s mapou a výpočet gradientů
    
