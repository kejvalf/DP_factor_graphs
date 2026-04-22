#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

// === Třída pro práci s Mapou (stejná jako minule) ===
class MapInterpolant
{
public:
    std::vector<double> xv, yv;
    std::vector<std::vector<double>> z_map;
    double dx, dy;

    MapInterpolant(const std::vector<double> &x_grid, const std::vector<double> &y_grid, const std::vector<std::vector<double>> &z_data)
        : xv(x_grid), yv(y_grid), z_map(z_data)
    {
        if (xv.size() > 1)
            dx = xv[1] - xv[0];
        else
            dx = 1.0;
        if (yv.size() > 1)
            dy = yv[1] - yv[0];
        else
            dy = 1.0;
    }

    double getZ(double x, double y) const
    {
        // Ošetření hranic
        if (x <= xv.front())
            x = xv.front();
        if (x >= xv.back())
            x = xv.back() - 1e-6;
        if (y <= yv.front())
            y = yv.front();
        if (y >= yv.back())
            y = yv.back() - 1e-6;

        int ix = static_cast<int>((x - xv[0]) / dx);
        int iy = static_cast<int>((y - yv[0]) / dy);

        // Bezpečnostní pojistka indexů
        ix = std::max(0, std::min(ix, (int)xv.size() - 2));
        iy = std::max(0, std::min(iy, (int)yv.size() - 2));

        double x_loc = (x - xv[ix]) / dx;
        double y_loc = (y - yv[iy]) / dy;

        // Pozor: Přístup k z_map[iy][ix] předpokládá strukturu [Řádek(Y)][Sloupec(X)]
        double z00 = z_map[iy][ix];
        double z10 = z_map[iy][ix + 1];
        double z01 = z_map[iy + 1][ix];
        double z11 = z_map[iy + 1][ix + 1];

        return (1 - y_loc) * ((1 - x_loc) * z00 + x_loc * z10) +
               y_loc * ((1 - x_loc) * z01 + x_loc * z11);
    }

    void getGradients(double x, double y, double &dZdx, double &dZdy) const
    {
        double eps = 1e-2;
        dZdx = (getZ(x + eps, y) - getZ(x - eps, y)) / (2 * eps);
        dZdy = (getZ(x, y + eps) - getZ(x, y - eps)) / (2 * eps);
    }
};

// === Funkce pro čtení CSV ===
inline std::vector<std::vector<double>> readCSV(const std::string &filename)
{
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;

    if (!file.is_open())
    {
        throw std::runtime_error("Nelze otevřít soubor: " + filename);
    }

    while (std::getline(file, line))
    {
        // 1. Odstranění "neviditelného" znaku návratu vozíku (\r),
        // který vzniká ve Windows a dělá problémy v Linux/Mac/GTSAM.
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        // 2. Detekce oddělovače (pokud řádek obsahuje středník, použijeme ho, jinak čárku)
        char delimiter = (line.find(';') != std::string::npos) ? ';' : ',';

        std::vector<double> row;
        std::stringstream ss(line);
        std::string val_str;

        while (std::getline(ss, val_str, delimiter))
        {
            // Odstranění případných mezer kolem čísla
            val_str.erase(0, val_str.find_first_not_of(" \t"));
            val_str.erase(val_str.find_last_not_of(" \t") + 1);

            if (!val_str.empty())
            {
                try
                {
                    // std::stod načítá double s plnou přesností (včetně formátu 1.48e+06)
                    // Je nezávislý na mezerách, ale vyžaduje tečku jako desetinný oddělovač.
                    row.push_back(std::stod(val_str));
                }
                catch (const std::invalid_argument &)
                {
                    // Ignorujeme buňky, které nejsou čísla (např. hlavičky)
                }
                catch (const std::out_of_range &)
                {
                    // Číslo je příliš velké pro double (nepravděpodobné u GNSS)
                }
            }
        }

        if (!row.empty())
            // std::cout << "Type of variable is : " << typeid(row).name() << std::endl;
            data.push_back(row);
    }
    return data;
}

// === Vylepšená funkce flatten ===
// Nyní používá reserve(), aby se vyhnula zbytečnému kopírování paměti
inline std::vector<double> flatten(const std::vector<std::vector<double>> &mat)
{
    if (mat.empty())
        return {};

    // Spočítáme celkový počet prvků pro alokaci paměti najednou
    size_t total_size = 0;
    for (const auto &row : mat)
        total_size += row.size();

    std::vector<double> vec;
    vec.reserve(total_size); // Klíčové pro výkon a paměť

    for (const auto &row : mat)
    {
        vec.insert(vec.end(), row.begin(), row.end());
    }
    return vec;
}