# MAE_Green_Function
VASP PDOS Analyzer: Orbital-Resolved MAE — 1st AND 2nd Order SOC
=================================================================
Implements the full Green-function perturbation theory from:

  "Green-Function Formalism for Spin–Orbit Coupling Energy and
   Magnetocrystalline Anisotropy: Second-Order Susceptibility Theory
   and Explicit First-Order (Degenerate) SOC"   H. Sabri (2026)

Sections implemented
---------------------
Sec. 3   Dyson expansion → identifies E^(1) and E^(2)
Sec. 4   Second-order susceptibility χ^{σσ'}_{mm'} (orbital-pair picture)
Sec. 5   First-order SOC energy via G0·Vso trace / orbital-moment route
Sec. 6   Degenerate first-order treatment for (dxz, dyz) doublet
Sec. 7   Complex-orbital PDOS ρ_{±1}(E), orbital-moment extraction
Sec. 8   Combined K ≈ K^(1)_active + K^(2)_rest

VASP requirements
-----------------
  LORBIT = 11   (lm-decomposed PDOS, real cubic harmonics)
  ISPIN  = 2    (collinear spin-polarised)
  For first-order MAE: two separate runs with SAXIS = 0 0 1 and SAXIS = 1 0 0
                       (or equivalent), so OUTCAR orbital moments can be compared.

Column layout expected in DOSCAR (LORBIT=11, ncols=19 per ion):
  0  Energy
  1-2   s   (up, dn)
  3-4   py  (up, dn)
  5-6   pz  (up, dn)
  7-8   px  (up, dn)
  9-10  dxy (up, dn)
  11-12 dyz (up, dn)
  13-14 dz2 (up, dn)
  15-16 dxz (up, dn)
  17-18 dx2-y2 (up, dn)
"""
