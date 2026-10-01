#!/usr/bin/env python3
"""
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
  ISMEAR = -5   (tetrahedron method)
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz
import os
import warnings

# ---------------------------------------------------------------------------
# Convenience: L·S matrix elements in real cubic-harmonic d basis
# (from Appendix A of the paper)
# Convention:  basis order = [dxy, dyz, dz2, dxz, dx2-y2]
#              m-quantum numbers: -2, -1, 0, +1, +2
#
# For magnetisation along z:  <m,σ|Lz Sz|m',σ'> = m δ_{mm'} δ_{σσ'} (σ = ±1/2)
# For spin-flip (l± s∓) the non-zero elements are stored below.
# ---------------------------------------------------------------------------

# Map real-harmonic labels → magnetic quantum number m
ORBITAL_M = {'dxy': -2, 'dyz': -1, 'dz2': 0, 'dxz': +1, 'dx2-y2': +2}
D_ORBITALS = ['dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']

# ---------------------------------------------------------------------------


class VaspMAEAnalyzer:
    """
    Full 1st- and 2nd-order SOC/MAE analyzer from VASP DOSCAR (and OUTCAR).
    """

    # ------------------------------------------------------------------
    # Construction / I/O
    # ------------------------------------------------------------------

    def __init__(self, doscar_file='DOSCAR'):
        self.doscar_file = doscar_file
        self.efermi = None
        self.pdos_data    = {}   # from SOC DOSCAR (LSORBIT=T) — used for K^(1) diagnostics
        self.pdos_G0_data = {}   # from scalar-relativistic DOSCAR (LSORBIT=F) — used for K^(2)
        self.efermi_G0    = None # E_F from the scalar-relativistic run
        self.nions = None
        self.orbital_moments = {}
        self.spin_moments    = {}

    # ------------------------------------------------------------------

    def read_doscar(self):
        """Parse DOSCAR; store lm-projected DOS for every ion."""
        print("=" * 70)
        print("Reading DOSCAR …")

        with open(self.doscar_file, 'r') as fh:
            lines = fh.readlines()

        self.nions  = int(lines[0].split()[0])
        fermi_line  = lines[5].split()
        self.efermi = float(fermi_line[3])
        nedos       = int(fermi_line[2])

        print(f"  Fermi energy : {self.efermi:.4f} eV")
        print(f"  Ions         : {self.nions}")
        print(f"  NEDOS        : {nedos}")

        start_pdos = 6 + nedos + 1          # skip total-DOS block

        for ion in range(self.nions):
            ion_start = start_pdos + ion * (nedos + 1)   # +1 for header line
            ion_data  = []

            for i in range(nedos):
                idx  = ion_start + 1 + i   # +1 to skip the per-ion header
                if idx >= len(lines):
                    break
                row = lines[idx].strip()
                if not row or row.startswith('#'):
                    continue
                try:
                    vals = [float(x) for x in row.split()]
                    ion_data.append(vals)
                except ValueError:
                    pass

            if not ion_data:
                continue

            # normalise row lengths
            max_len = max(len(r) for r in ion_data)
            for r in ion_data:
                while len(r) < max_len:
                    r.append(0.0)

            arr = np.array(ion_data, dtype=float)

            if arr.shape[1] < 19:
                # not enough columns for d-orbitals — skip silently
                continue

            self.pdos_data[ion] = {
                'energy'  : arr[:, 0],
                'dxy'     : {'up': arr[:, 9],  'dn': arr[:, 10]},
                'dyz'     : {'up': arr[:, 11], 'dn': arr[:, 12]},
                'dz2'     : {'up': arr[:, 13], 'dn': arr[:, 14]},
                'dxz'     : {'up': arr[:, 15], 'dn': arr[:, 16]},
                'dx2-y2'  : {'up': arr[:, 17], 'dn': arr[:, 18]},
            }

        print(f"  PDOS loaded  : {len(self.pdos_data)} ions")
        return len(self.pdos_data) > 0

    # ------------------------------------------------------------------

    def load_G0_doscar(self, doscar_G0_path):
        """
        Load the scalar-relativistic (LSORBIT=F) DOSCAR into self.pdos_G0_data.

        WHY THIS IS REQUIRED FOR K^(2):
          The second-order formula uses G0, the unperturbed Green's function:
            E^(2) = -1/2 Int Im Tr[G0 Vso G0 Vso]
          Using the SOC DOSCAR (LSORBIT=T) instead contaminates G0 with
          SOC feedback, double-counting part of the SOC energy.

        VASP INCAR for the G0 run:
          LSORBIT = .FALSE.
          ISPIN   = 2
          LORBIT  = 11      (lm-decomposed PDOS)
          ISMEAR  = -5      (tetrahedron)
          ICHARG  = 11      (non-self-consistent on SOC charge density)

        Parameters
        ----------
        doscar_G0_path : path to DOSCAR from the LSORBIT=F calculation

        After calling this, all susceptibility calculations automatically
        use the G0 spectral weights.
        """
        print("=" * 70)
        print(f"Loading G0 DOSCAR (scalar-relativistic, LSORBIT=F)")
        print(f"  File: {doscar_G0_path}")

        with open(doscar_G0_path, 'r') as fh:
            lines = fh.readlines()

        nions      = int(lines[0].split()[0])
        fermi_line = lines[5].split()
        efermi_G0  = float(fermi_line[3])
        nedos      = int(fermi_line[2])

        print(f"  E_F (G0 run) : {efermi_G0:.4f} eV")
        print(f"  Ions         : {nions}")
        print(f"  NEDOS        : {nedos}")

        if abs(efermi_G0 - self.efermi) > 0.5:
            print(f"  WARNING: E_F difference = {abs(efermi_G0-self.efermi):.3f} eV "
                  f"(SOC={self.efermi:.4f}, G0={efermi_G0:.4f}).")
            print("           Large difference suggests mismatched structures.")

        self.efermi_G0 = efermi_G0
        start_pdos = 6 + nedos + 1

        for ion in range(nions):
            ion_start = start_pdos + ion * (nedos + 1)
            ion_data  = []

            for i in range(nedos):
                idx = ion_start + 1 + i
                if idx >= len(lines): break
                row = lines[idx].strip()
                if not row or row.startswith('#'): continue
                try:
                    vals = [float(x) for x in row.split()]
                    ion_data.append(vals)
                except ValueError:
                    pass

            if not ion_data: continue

            max_len = max(len(r) for r in ion_data)
            for r in ion_data:
                while len(r) < max_len: r.append(0.0)

            arr = np.array(ion_data, dtype=float)
            if arr.shape[1] < 19: continue

            self.pdos_G0_data[ion] = {
                'energy'  : arr[:, 0],
                'dxy'     : {'up': arr[:, 9],  'dn': arr[:, 10]},
                'dyz'     : {'up': arr[:, 11], 'dn': arr[:, 12]},
                'dz2'     : {'up': arr[:, 13], 'dn': arr[:, 14]},
                'dxz'     : {'up': arr[:, 15], 'dn': arr[:, 16]},
                'dx2-y2'  : {'up': arr[:, 17], 'dn': arr[:, 18]},
            }

        print(f"  G0 PDOS loaded: {len(self.pdos_G0_data)} ions")
        print("  K^(2) susceptibilities will now use this G0 data.")
        return len(self.pdos_G0_data) > 0

    # ------------------------------------------------------------------
    # Moment input: three routes (highest → lowest accuracy)
    # ------------------------------------------------------------------

    def set_moments_directly(self, ion_idx,
                              L_inplane, L_outofplane,
                              Sz_inplane, Sz_outofplane):
        """
        ROUTE A — supply orbital and spin moments directly (most reliable).

        Sign convention:
          K = E_SOC(110) - E_SOC(001)  [< 0 => in-plane easy axis]

          K^(1) = Sum_i xi_i * [L_110_i * Sz_110_i  -  L_001_i * Sz_001_i]

        Spin moments MUST be supplied separately for each magnetisation
        direction because the spin-orbit mixing changes the local spin
        moment when the quantisation axis is rotated (see Table I of your
        paper: Co1 changes from 1.445 uB along [001] to 1.632 uB along [110]).

        Parameters
        ----------
        ion_idx        : 0-based ion index
        L_inplane      : orbital moment from OUTCAR with SAXIS=1 1 0  [hbar]
        L_outofplane   : orbital moment from OUTCAR with SAXIS=0 0 1  [hbar]
        Sz_inplane     : spin moment from OUTCAR with SAXIS=1 1 0  [muB or hbar/2]
        Sz_outofplane  : spin moment from OUTCAR with SAXIS=0 0 1  [same units]

        Sr3Co2O7 values (paper Table I + Eq. 2):
            # Co1
            analyzer.set_moments_directly(ion,
                L_inplane=0.30, L_outofplane=0.90,
                Sz_inplane=1.632, Sz_outofplane=1.445)
            # Co2
            analyzer.set_moments_directly(ion,
                L_inplane=0.15, L_outofplane=0.50,
                Sz_inplane=2.253, Sz_outofplane=2.220)
        """
        self.orbital_moments.setdefault(ion_idx, {})
        self.spin_moments.setdefault(ion_idx, {})

        # For SAXIS=[1,1,0]: the orbital moment vector lies along [110].
        # Its Cartesian components are  Lx = Ly = L_inplane / sqrt(2), Lz = 0.
        # Storing (L_inplane, L_inplane, 0) would cause calculate_first_order_mae
        # to compute sqrt(Lx^2+Ly^2) = L_inplane*sqrt(2)  — a sqrt(2) error.
        # Correct storage: Lx = Ly = L_inplane / sqrt(2), so that
        # sqrt(Lx^2+Ly^2+Lz^2) = L_inplane  as intended.
        _Lc = L_inplane / np.sqrt(2)
        self.orbital_moments[ion_idx]['110'] = (_Lc, _Lc, 0.0)
        self.orbital_moments[ion_idx]['001'] = (0.0, 0.0, L_outofplane)
        self.spin_moments[ion_idx]['110']    = Sz_inplane
        self.spin_moments[ion_idx]['001']    = Sz_outofplane

        contrib = L_inplane * Sz_inplane - L_outofplane * Sz_outofplane
        print(f"  Ion {ion_idx+1}:"
              f"  L_110={L_inplane:+.4f} Sz_110={Sz_inplane:+.4f}"
              f"  |  L_001={L_outofplane:+.4f} Sz_001={Sz_outofplane:+.4f}"
              f"  |  (L*Sz)_110-(L*Sz)_001 = {contrib:+.5f}")

    def read_outcar_single(self, outcar_path, tag):
        """
        Read orbital moments (Lx, Ly, Lz) and spin moments (Sz) from one
        VASP SOC OUTCAR (LSORBIT=T).

        VASP writes three separate blocks:
          orbital moment (x)  →  Lx per ion  (tot column = p + d)
          orbital moment (y)  →  Ly per ion
          orbital moment (z)  →  Lz per ion
          magnetization (z)   →  Sz per ion  (last occurrence = converged)

        Stores:
          self.orbital_moments[ion_0idx][tag] = (Lx, Ly, Lz)
          self.spin_moments[ion_0idx][tag]    = Sz

        For a [110] run (tag='110'):  Lx≈Ly, Lz≈0  →  |L_110|=sqrt(Lx²+Ly²)
        For a [001] run (tag='001'):  Lx≈Ly≈0, Lz≠0 →  |L_001|=|Lz|

        Parameters
        ----------
        outcar_path : str  — full path to OUTCAR
        tag         : str  — '110' or '001' (used as dict key internally)
        """
        if not os.path.isfile(outcar_path):
            print(f"  [OUTCAR] not found: {outcar_path}")
            return False

        print(f"\n  Reading OUTCAR: {outcar_path}  (tag='{tag}')")
        with open(outcar_path) as fh:
            lines = fh.readlines()

        # ----------------------------------------------------------------
        # Helper: parse ONE "orbital moment (x/y/z)" block.
        # Returns {ion_0idx: tot_value} from the 'tot' column.
        # VASP format:
        #   orbital moment (x)
        #
        #   # of ion     p       d       tot
        #   ----------------------------------------
        #    1        ...     ...     ...
        # ----------------------------------------------------------------
        def parse_orbital_block(lines, start_line):
            """
            Parse one 'orbital moment (x/y/z)' block.
            VASP format:
              orbital moment (x)       <- start_line (already matched)
                                       <- blank
              # of ion   p    d    tot <- header (skip)
              --------------------     <- dashes (skip)
               1       ...  ...  ...   <- data lines we want
              ...
              --------------------     <- closing dashes (stop)
            Returns {ion_0idx: tot_value}
            """
            result = {}
            seen_data = False
            for k in range(start_line + 1, min(start_line + 80, len(lines))):
                row = lines[k].strip()
                if not row:
                    continue
                # Skip the column header line
                if row.startswith('#'):
                    continue
                # Dashes: skip FIRST occurrence (after header), stop on SECOND
                if row.startswith('---') or row.startswith('===='):
                    if seen_data:
                        break   # closing dashes — end of block
                    else:
                        continue  # opening dashes — skip
                parts = row.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    try:
                        ion_0idx = int(parts[0]) - 1
                        tot = float(parts[-1])  # last column = tot
                        result[ion_0idx] = tot
                        seen_data = True
                    except ValueError:
                        pass
            return result

        # ----------------------------------------------------------------
        # Scan all lines for the relevant blocks.
        # We want the LAST occurrence of each (= fully converged SCF).
        # ----------------------------------------------------------------
        Lx_all = {}   # ion_0idx -> Lx (tot from orbital moment (x))
        Ly_all = {}
        Lz_all = {}
        Sz_all = {}

        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped == 'orbital moment (x)':
                Lx_all = parse_orbital_block(lines, i)

            elif stripped == 'orbital moment (y)':
                Ly_all = parse_orbital_block(lines, i)

            elif stripped == 'orbital moment (z)':
                Lz_all = parse_orbital_block(lines, i)

            elif stripped == 'magnetization (z)':
                seen_data_mag = False
                for k in range(i + 1, min(i + 60, len(lines))):
                    row = lines[k].strip()
                    if not row: continue
                    if row.startswith('#'): continue
                    if row.startswith('---'):
                        if seen_data_mag: break
                        else: continue
                    parts = row.split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        try:
                            ion_0idx = int(parts[0]) - 1
                            Sz_all[ion_0idx] = float(parts[-1])
                            seen_data_mag = True
                        except ValueError:
                            pass

        # ----------------------------------------------------------------
        # Store combined (Lx, Ly, Lz) and Sz
        # ----------------------------------------------------------------
        all_ions = set(Lx_all) | set(Ly_all) | set(Lz_all)
        for ion_0idx in all_ions:
            lx = Lx_all.get(ion_0idx, 0.0)
            ly = Ly_all.get(ion_0idx, 0.0)
            lz = Lz_all.get(ion_0idx, 0.0)
            self.orbital_moments.setdefault(ion_0idx, {})[tag] = (lx, ly, lz)

        for ion_0idx, sz in Sz_all.items():
            self.spin_moments.setdefault(ion_0idx, {})[tag] = sz

        # ----------------------------------------------------------------
        # Print summary for Co ions (|L| > 0.1 hbar)
        # ----------------------------------------------------------------
        L_tot = {}
        for i in self.orbital_moments:
            if tag in self.orbital_moments[i]:
                lx, ly, lz = self.orbital_moments[i][tag]
                L_tot[i] = np.sqrt(lx**2 + ly**2 + lz**2)

        co_ions = [i for i, v in L_tot.items() if v > 0.1]

        print(f"    Orbital moments loaded : {len(all_ions)} ions")
        print(f"    Spin moments loaded    : {len(Sz_all)} ions")
        if co_ions:
            print(f"\n    {'Ion':>4}  {'Lx':>8}  {'Ly':>8}  {'Lz':>8}  "
                  f"{'|L|':>8}  {'Sz':>8}")
            print("    " + "-" * 52)
            for i in sorted(co_ions):
                lx, ly, lz = self.orbital_moments[i][tag]
                lmag = np.sqrt(lx**2 + ly**2 + lz**2)
                sz = self.spin_moments.get(i, {}).get(tag, float('nan'))
                print(f"    {i+1:>4}  {lx:>8.4f}  {ly:>8.4f}  {lz:>8.4f}  "
                      f"{lmag:>8.4f}  {sz:>8.4f}")
        return len(all_ions) > 0

    def read_outcar_moments(self, outcar_110, outcar_001):
        """Convenience: read both SOC OUTCARs."""
        ok1 = self.read_outcar_single(outcar_110, tag='110')
        ok2 = self.read_outcar_single(outcar_001, tag='001')
        return ok1, ok2

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_pdos(self, ion_idx, energy_range=(-8, 6),
                  use_G0=False, ion_label=None, filename=None):
        """
        Plot m-grouped d-PDOS for one ion.

        Parameters
        ----------
        ion_idx      : 0-based ion index
        energy_range : (emin, emax) relative to E_F  [eV]
        use_G0       : if True, plot the non-SOC G0 PDOS (LSORBIT=F);
                       if False (default), plot the SOC PDOS (LSORBIT=T)
        ion_label    : override the title label (e.g. 'Co1', 'Co2')
        filename     : if given, save figure to this path

        Groups by magnetic quantum number |m|:
          |m|=2  →  dxy + dx²-y²
          |m|=1  →  dxz + dyz
           m=0   →  dz²
        """
        if use_G0:
            if not self.pdos_G0_data:
                print("  No G0 PDOS loaded. Call load_G0_doscar() first.")
                return
            if ion_idx not in self.pdos_G0_data:
                print(f"  Ion {ion_idx} not in G0 PDOS data.")
                return
            data   = self.pdos_G0_data[ion_idx]
            ef     = self.efermi_G0 if self.efermi_G0 is not None else self.efermi
            source = 'Non-SOC (G0, LSORBIT=F)'
        else:
            if ion_idx not in self.pdos_data:
                print(f"  Ion {ion_idx} not in SOC PDOS data.")
                return
            data   = self.pdos_data[ion_idx]
            ef     = self.efermi
            source = 'SOC (LSORBIT=T)'

        energy = data['energy']
        mask   = (energy >= ef + energy_range[0]) & (energy <= ef + energy_range[1])
        e_plot = energy[mask] - ef

        groups  = {
            r'$|m|=2$  (dxy + dx²−y²)': ['dxy', 'dx2-y2'],
            r'$|m|=1$  (dxz + dyz)':    ['dxz', 'dyz'],
            r'$m=0$    (dz²)':           ['dz2'],
        }
        colours = ['#d62728', '#1f77b4', '#2ca02c']
        tag     = ion_label or f'Ion {ion_idx+1}'

        fig, ax = plt.subplots(figsize=(9, 6))
        for (glabel, orbs), col in zip(groups.items(), colours):
            up = np.zeros(mask.sum())
            dn = np.zeros(mask.sum())
            for o in orbs:
                up += data[o]['up'][mask]
                dn += data[o]['dn'][mask]
            ax.fill_between(e_plot,  0,  up, color=col, alpha=0.35)
            ax.fill_between(e_plot,  0, -dn, color=col, alpha=0.35)
            ax.plot(e_plot,  up, color=col, lw=1.5, label=glabel + ' ↑')
            ax.plot(e_plot, -dn, color=col, lw=1.5, linestyle='--',
                    label=glabel + ' ↓')

        ax.axvline(0, color='k', ls=':', lw=1, label='$E_F$')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('$E - E_F$ (eV)', fontsize=12)
        ax.set_ylabel('DOS (states/eV)', fontsize=12)
        ax.set_title(f'{tag} — d-PDOS  [{source}]', fontsize=13)
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.set_xlim(energy_range)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"  Saved PDOS plot: {filename}")
        plt.show()

    def plot_pdos_comparison(self, ion_idx, energy_range=(-8, 6),
                             ion_label=None, filename=None):
        """
        Side-by-side comparison: SOC (left) vs non-SOC G0 (right) PDOS.
        Requires both pdos_data and pdos_G0_data to be loaded.
        """
        if ion_idx not in self.pdos_data or ion_idx not in self.pdos_G0_data:
            print(f"  Need both SOC and G0 PDOS loaded for comparison plot.")
            return

        groups  = {
            r'$|m|=2$': ['dxy', 'dx2-y2'],
            r'$|m|=1$': ['dxz', 'dyz'],
            r'$m=0$'  : ['dz2'],
        }
        colours = ['#d62728', '#1f77b4', '#2ca02c']
        tag     = ion_label or f'Ion {ion_idx+1}'

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        datasets = [
            (self.pdos_data[ion_idx],    self.efermi,    'SOC  (LSORBIT=T)',      axes[0]),
            (self.pdos_G0_data[ion_idx], self.efermi_G0 or self.efermi,
             'Non-SOC G0  (LSORBIT=F)', axes[1]),
        ]

        for data, ef, src_label, ax in datasets:
            energy = data['energy']
            mask   = ((energy >= ef + energy_range[0]) &
                      (energy <= ef + energy_range[1]))
            e_plot = energy[mask] - ef

            for (glabel, orbs), col in zip(groups.items(), colours):
                up = np.zeros(mask.sum())
                dn = np.zeros(mask.sum())
                for o in orbs:
                    up += data[o]['up'][mask]
                    dn += data[o]['dn'][mask]
                ax.fill_between(e_plot,  0,  up, color=col, alpha=0.35)
                ax.fill_between(e_plot,  0, -dn, color=col, alpha=0.35)
                ax.plot(e_plot,  up, color=col, lw=1.5, label=glabel + ' ↑')
                ax.plot(e_plot, -dn, color=col, lw=1.5, linestyle='--',
                        label=glabel + ' ↓')

            ax.axvline(0, color='k', ls=':', lw=1, label='$E_F$')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('$E - E_F$ (eV)', fontsize=12)
            ax.set_title(f'{tag} — {src_label}', fontsize=12)
            ax.legend(fontsize=8, loc='upper left', ncol=1)
            ax.set_xlim(energy_range)
            ax.grid(alpha=0.25)

        axes[0].set_ylabel('DOS (states/eV)', fontsize=12)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"  Saved comparison plot: {filename}")
        plt.show()



    # ==================================================================
    # ███  FIRST-ORDER SOC  ███
    # ==================================================================

    def _integrate_pdos(self, ion_idx, orbital, spin, energy_range):
        """
        ∫^{EF}_{e_min} dE  ρ_{orb,σ}(E)    (occupied integral)
        Returns scalar.
        """
        if ion_idx not in self.pdos_data:
            return 0.0
        data   = self.pdos_data[ion_idx]
        energy = data['energy']
        dos    = data[orbital][spin]

        e_min  = self.efermi + energy_range[0]
        mask   = (energy >= e_min) & (energy <= self.efermi)
        if mask.sum() < 2:
            return 0.0
        return trapz(dos[mask], energy[mask])

    # ------------------------------------------------------------------

    def calculate_spin_moment_from_pdos(self, ion_idx, energy_range=(-10, 0)):
        """
        Spin moment of ion from PDOS:
          <Sz>_i = (1/2) ∫^{EF} dE [ρ↑(E) − ρ↓(E)]   (summed over d-orbitals)

        Returns (Sz_total, {orbital: Sz_orbital}) in units of ħ/2.
        """
        if ion_idx not in self.pdos_data:
            return 0.0, {}

        sz_dict = {}
        sz_total = 0.0
        for orb in D_ORBITALS:
            n_up = self._integrate_pdos(ion_idx, orb, 'up', energy_range)
            n_dn = self._integrate_pdos(ion_idx, orb, 'dn', energy_range)
            sz   = 0.5 * (n_up - n_dn)          # in units ħ/2
            sz_dict[orb] = sz
            sz_total += sz

        return sz_total, sz_dict

    # ------------------------------------------------------------------

    def construct_complex_orbital_pdos(self, ion_idx):
        """
        Build approximate ρ_{±1}(E) for the (dxz, dyz) doublet from real-PDOS.

        THEORY (Sec. 7.3):
          |±1⟩ = (1/√2)(|xz⟩ ± i|yz⟩)
          P_{±1} = (1/2)[P_xz + P_yz  ±  i(|xz⟩⟨yz| − |yz⟩⟨xz|)]

        Diagonal approximation (only real PDOS available):
          ρ_{+1}(E) ≈ (1/2)[ρ_xz(E) + ρ_yz(E)]
          ρ_{-1}(E) ≈ (1/2)[ρ_xz(E) + ρ_yz(E)]

        NOTE: In this diagonal approximation ρ_{+1} = ρ_{-1} so the net
              <Lz> is exactly zero.  A non-zero ⟨Lz⟩ requires the
              off-diagonal spectral weight A_{xz,yz}(E) = −Im⟨xz|G|yz⟩/π
              (Eq. 43 of the paper).  If Wannier or PROCAR off-diagonal
              data are available, pass them to calculate_orbital_moment_complex().

        Returns: dict with 'rho_p1_up', 'rho_m1_up', 'rho_p1_dn', 'rho_m1_dn', 'energy'
        """
        if ion_idx not in self.pdos_data:
            return None

        data = self.pdos_data[ion_idx]
        e    = data['energy']

        for spin in ('up', 'dn'):
            xz = data['dxz'][spin]
            yz = data['dyz'][spin]
            rho_p = 0.5 * (xz + yz)
            rho_m = 0.5 * (xz + yz)

        # Build per-spin
        result = {'energy': e}
        for spin in ('up', 'dn'):
            xz  = data['dxz'][spin]
            yz  = data['dyz'][spin]
            result[f'rho_p1_{spin}'] = 0.5 * (xz + yz)
            result[f'rho_m1_{spin}'] = 0.5 * (xz + yz)

        return result

    # ------------------------------------------------------------------

    def calculate_orbital_moment_complex(
            self, ion_idx,
            off_diag_xzyz_up=None, off_diag_xzyz_dn=None,
            energy_range=(-10, 0)):
        """
        Calculate site orbital moment ⟨Lz⟩_i from complex-orbital PDOS (Eq. 45):
          ⟨Lz⟩_i = ħ Σ_σ ∫^{EF} dE [ρ_{+1,σ}(E) − ρ_{-1,σ}(E)]

        When off-diagonal spectral densities A_{xz,yz}(E) are available (from
        Wannier post-processing), the exact complex PDOS is:
          ρ_{+1,σ}(E) − ρ_{-1,σ}(E) = 2 · Im A_{xz,yz,σ}(E)     [Eq. 43 route]

        Parameters
        ----------
        off_diag_xzyz_up, off_diag_xzyz_dn : 1-D arrays on same energy grid,
            = Im⟨xz|G|yz⟩_σ / (−π)  [i.e. the off-diagonal PDOS]
            If None → diagonal approximation (returns Lz ≈ 0 with a warning).

        Returns: (Lz_hbar, diagnostic_dict)
        """
        if ion_idx not in self.pdos_data:
            return 0.0, {}

        data   = self.pdos_data[ion_idx]
        energy = data['energy']
        e_min  = self.efermi + energy_range[0]
        mask   = (energy >= e_min) & (energy <= self.efermi)

        diag  = self.construct_complex_orbital_pdos(ion_idx)

        if off_diag_xzyz_up is None:
            warnings.warn(
                f"Ion {ion_idx}: No off-diagonal A_{{xz,yz}} supplied.\n"
                "  In the diagonal real-PDOS approximation ρ_{{+1}} = ρ_{{-1}}, so ⟨Lz⟩ = 0.\n"
                "  Provide Wannier off-diagonal spectral weights for a non-trivial result.",
                UserWarning, stacklevel=2)
            # Still compute the diagonal sum for diagnostics
            rho_diff_up = diag['rho_p1_up'] - diag['rho_m1_up']
            rho_diff_dn = diag['rho_p1_dn'] - diag['rho_m1_dn']
        else:
            # Exact route: ρ_{+1} − ρ_{-1} = 2 · Im[A_{xz,yz}]  (Sec 7.3)
            rho_diff_up = 2.0 * off_diag_xzyz_up
            rho_diff_dn = 2.0 * off_diag_xzyz_dn

        int_up = trapz(rho_diff_up[mask], energy[mask]) if mask.sum() > 1 else 0.0
        int_dn = trapz(rho_diff_dn[mask], energy[mask]) if mask.sum() > 1 else 0.0
        Lz = int_up + int_dn          # in units of ħ (Eq. 45)

        diag_out = {
            'int_up': int_up, 'int_dn': int_dn,
            'Lz_hbar': Lz,
            'note': 'diagonal approx' if off_diag_xzyz_up is None else 'off-diagonal used'
        }
        return Lz, diag_out

    # ==================================================================
    # ███  DIRECT MAE FROM E_soc  (most accurate route)  ███
    # ==================================================================

    def read_esoc_from_outcar(self, outcar_path, tag):
        """
        Read the site-resolved SOC energy (E_soc) and l=2 SOC matrix
        from a VASP OUTCAR for a given SAXIS direction.

        These appear in the 'Spin-Orbit-Coupling matrix elements' block
        that VASP writes at the end of each SOC calculation.

        E_soc per ion is the expectation value of H_soc = xi L.S for
        that ion in the converged density — i.e. the actual SOC energy
        contribution.  The MAE is then simply:

          K = E_soc_total(110) - E_soc_total(001)

        which is equivalent to the force-theorem total-energy difference
        but decomposed per ion and per orbital.

        Parameters
        ----------
        outcar_path : str — path to OUTCAR file
        tag         : str — '110' or '001', used as dict key internally

        Populates
        ---------
        self.esoc[tag]       : {ion_idx(0-based): E_soc_eV}
        self.soc_matrix[tag] : {ion_idx(0-based): 5x5 numpy array (l=2 block)}
        """
        import re

        if not hasattr(self, 'esoc'):       self.esoc = {}
        if not hasattr(self, 'soc_matrix'): self.soc_matrix = {}

        if not os.path.isfile(outcar_path):
            print(f"  [E_soc] {outcar_path} not found.")
            return False

        print(f"  Reading E_soc and SOC matrices from {outcar_path}  (tag='{tag}') ...")
        with open(outcar_path) as fh:
            lines = fh.readlines()

        esoc_dict   = {}
        matrix_dict = {}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # ---- detect Ion block: "Ion:  N  E_soc:  VALUE" ---------------
            if line.startswith('Ion:') and 'E_soc:' in line:
                parts = line.split()
                try:
                    ion_1idx = int(parts[1])
                    esoc_val = float(parts[3])
                except (IndexError, ValueError):
                    i += 1
                    continue

                ion_0idx = ion_1idx - 1
                esoc_dict[ion_0idx] = esoc_val

                # ---- scan ahead for l=1, l=2, l=3 blocks ------------------
                j = i + 1
                l2_rows = []

                while j < len(lines) and j < i + 60:   # safety window
                    inner = lines[j].strip()

                    # stop when we hit the next Ion block
                    if inner.startswith('Ion:') and 'E_soc:' in inner:
                        break

                    # detect l= header
                    if inner.startswith('l='):
                        try:
                            l_val = int(inner.split('=')[1].strip())
                        except (IndexError, ValueError):
                            j += 1
                            continue

                        if l_val == 2:
                            # read the next 5 data lines
                            k = j + 1
                            while k < len(lines) and len(l2_rows) < 5:
                                data_line = lines[k].strip()
                                # stop at next l= or Ion: line
                                if data_line.startswith('l=') or data_line.startswith('Ion:'):
                                    break
                                if data_line:   # non-empty
                                    try:
                                        nums = [float(x) for x in data_line.split()]
                                        if len(nums) == 5:
                                            l2_rows.append(nums)
                                    except ValueError:
                                        pass
                                k += 1
                            j = k
                            continue

                    j += 1

                if len(l2_rows) == 5:
                    matrix_dict[ion_0idx] = np.array(l2_rows)

                i = j   # jump past the processed block
                continue

            i += 1

        self.esoc[tag]       = esoc_dict
        self.soc_matrix[tag] = matrix_dict

        # Summary
        n_ions = len(esoc_dict)
        print(f"    Loaded E_soc for {n_ions} ions, l=2 matrices for {len(matrix_dict)} ions")

        # Print Co ions
        Co_ions = [i for i, v in esoc_dict.items() if abs(v) > 0.01]
        if Co_ions:
            print(f"    {'Ion(0-idx)':<12} {'Ion(1-idx)':<12} {'E_soc (eV)':>12}  {'E_soc (meV)':>13}")
            print("    " + "-" * 52)
            for i in sorted(Co_ions):
                print(f"    {i:<12} {i+1:<12} {esoc_dict[i]:>12.7f}  {esoc_dict[i]*1000:>13.4f}")

        total = sum(esoc_dict.values())
        print(f"    E_soc_total({tag}) = {total:+.6f} eV = {total*1000:+.2f} meV")
        return True

    # ------------------------------------------------------------------

    def compute_mae_direct(self, tag_easy='110', tag_hard='001'):
        """
        Compute K directly as  K = E_soc_total(easy) - E_soc_total(hard)
        decomposed per species.

        This is the most accurate approach — equivalent to the DFT
        total-energy force-theorem difference, but decomposed per site.

        Requires read_esoc_from_outcar() called for both tags.

        Parameters
        ----------
        tag_easy : str — tag for the easy-axis OUTCAR (default '110')
        tag_hard : str — tag for the hard-axis OUTCAR (default '001')

        Returns
        -------
        dict with 'K_total_meV', 'K_per_ion', 'K_Co1_meV', 'K_Co2_meV', etc.
        """
        if not hasattr(self, 'esoc') or tag_easy not in self.esoc or tag_hard not in self.esoc:
            print("  ERROR: Call read_esoc_from_outcar() for both tags first.")
            return {}

        esoc_e = self.esoc[tag_easy]
        esoc_h = self.esoc[tag_hard]

        all_ions = sorted(set(esoc_e) | set(esoc_h))

        print("\n" + "=" * 70)
        print(f"DIRECT MAE:  K = E_soc({tag_easy}) - E_soc({tag_hard})")
        print("  (force-theorem / ΔE_soc, most accurate, no PDOS approximation)")
        print("=" * 70)
        print(f"  {'Ion':<6} {'Type':<6} {'E_soc({})'.format(tag_easy):>14} "
              f"{'E_soc({})'.format(tag_hard):>14} {'ΔE_soc (meV)':>14}")
        print("  " + "-" * 58)

        K_per_ion = {}
        K_Sr = K_Co1 = K_Co2 = K_O = 0.0

        # Identify Co ions from spin moments or by E_soc magnitude
        co1_set = set(getattr(self, '_co1_ions', []))
        co2_set = set(getattr(self, '_co2_ions', []))

        for ion in all_ions:
            e = esoc_e.get(ion, 0.0)
            h = esoc_h.get(ion, 0.0)
            delta = (e - h) * 1000   # meV

            # Classify ion
            if abs(e) > 0.08 or ion in co1_set:
                itype = 'Co1'
                K_Co1 += delta
            elif abs(e) > 0.04 or ion in co2_set:
                itype = 'Co2'
                K_Co2 += delta
            elif abs(e) > 0.01:
                itype = 'Sr'
                K_Sr  += delta
            else:
                itype = 'O'
                K_O   += delta

            K_per_ion[ion] = delta
            flag = "  ***" if abs(delta) > 10 else ""
            print(f"  {ion+1:<6} {itype:<6} {e:>14.7f} {h:>14.7f} {delta:>14.4f}{flag}")

        K_total = sum(K_per_ion.values())
        print("  " + "-" * 58)
        print(f"  {'TOTAL':<12}                            {K_total:>14.4f} meV")
        print()
        print(f"  By species:")
        print(f"    K_Sr   = {K_Sr:+.4f} meV")
        print(f"    K_Co1  = {K_Co1:+.4f} meV")
        print(f"    K_Co2  = {K_Co2:+.4f} meV")
        print(f"    K_O    = {K_O:+.4f} meV")
        print()
        sign = "in-plane" if K_total < 0 else "out-of-plane"
        print(f"  K_total = {K_total:+.4f} meV  =>  {sign} easy axis")

        return {
            'K_total_meV': K_total,
            'K_per_ion_meV': K_per_ion,
            'K_Co1_meV': K_Co1, 'K_Co2_meV': K_Co2,
            'K_Sr_meV': K_Sr,   'K_O_meV': K_O,
        }

    # ------------------------------------------------------------------

    def print_soc_matrix(self, tag, ion_idx):
        """
        Pretty-print the l=2 SOC matrix for one ion.
        Basis order: [dxy, dyz, dz2, dxz, dx2-y2]
        """
        if not hasattr(self, 'soc_matrix') or tag not in self.soc_matrix:
            print("  No SOC matrix data. Call read_esoc_from_outcar() first.")
            return
        if ion_idx not in self.soc_matrix[tag]:
            print(f"  No l=2 matrix for ion {ion_idx}.")
            return

        mat = self.soc_matrix[tag][ion_idx]
        orbs = ['dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2']
        E_soc = self.esoc[tag].get(ion_idx, 0.0)

        print(f"\n  l=2 SOC matrix — ion {ion_idx+1}  E_soc={E_soc*1000:+.3f} meV"
              f"  (tag='{tag}')")
        print(f"  Basis: [dxy, dyz, dz2, dxz, dx2-y2]  (values in eV)")
        print(f"  {'':8}", end="")
        for o in orbs: print(f"  {o:>10}", end="")
        print()
        for i, row in enumerate(mat):
            print(f"  {orbs[i]:8}", end="")
            for v in row:
                s = f"{v:+.5f}" if abs(v) > 1e-7 else "    0    "
                print(f"  {s:>10}", end="")
            print()

    # ------------------------------------------------------------------

    def calculate_first_order_mae(
            self,
            ion_indices,
            xi_values,
            Lz_z_dict=None, Lz_x_dict=None,
            spin_z_dict=None,
            energy_range=(-10, 0)):
        """
        First-order MAE  K^(1) (Eq. 27/46):
          K^(1) ≈ Σ_i  ξ_i <S>_i  ΔL_i
          ΔL_i = |⟨L∥⟩_i| − |⟨L⊥⟩_i|   (easy-axis − hard-axis orbital moment)

        Three input routes (highest to lowest accuracy):
          A. Provide Lz_z_dict / Lz_x_dict  (orbital moments from two OUTCAR files)
          B. Provide off-diagonal Wannier PDOS (via calculate_orbital_moment_complex)
          C. Diagonal PDOS approximation  → Lz ≈ 0, reports warning.

        Parameters
        ----------
        ion_indices : list of ion indices to sum over
        xi_values   : dict {ion_idx: ξ_i}  SOC constants in eV
        Lz_z_dict   : {ion_idx: Lz_z}  orbital moments (z-magnetisation) in ħ
        Lz_x_dict   : {ion_idx: Lz_x}  orbital moments (x-magnetisation) in ħ
        spin_z_dict : {ion_idx: Sz}    spin moment (z) in ħ/2; if None, compute from PDOS

        Returns: (K1_meV, site_contributions)
        """
        print("\n" + "=" * 70)
        print("FIRST-ORDER MAE  K^(1)  [paper Eqs. 15, 17, 20]")
        print("  K^(1) = Sum_i  xi_i * [L_001_i*Sz_001_i  -  L_110_i*Sz_110_i]")
        print("  (easy axis has LARGER orbital moment -> more negative E^(1))")
        print("  K^(1) < 0  =>  in-plane (110) easy axis  [your system: ~-177 meV]")
        print("=" * 70)

        K1_total  = 0.0
        site_info = {}
        any_missing_L = False

        for ion in ion_indices:
            xi = xi_values.get(ion, 0.065)

            # ---- Spin moments: separate value per direction -----------------
            # Priority: spin_z_dict (applied to both dirs, legacy) >
            #           self.spin_moments['110'] / ['001'] > PDOS integral
            if spin_z_dict and ion in spin_z_dict:
                Sz_110 = spin_z_dict[ion]
                Sz_001 = spin_z_dict[ion]
                Sz_src = "direct (same both dirs)"
            else:
                sm = self.spin_moments.get(ion, {})
                if '110' in sm and '001' in sm:
                    Sz_110 = sm['110']
                    Sz_001 = sm['001']
                    Sz_src = "OUTCAR/set"
                elif '110' in sm:
                    Sz_110 = sm['110']
                    Sz_001 = sm['110']   # fallback: same value
                    Sz_src = "OUTCAR/set (110 only)"
                elif '001' in sm:
                    Sz_110 = sm['001']
                    Sz_001 = sm['001']
                    Sz_src = "OUTCAR/set (001 only)"
                else:
                    Sz_110, _ = self.calculate_spin_moment_from_pdos(ion, energy_range)
                    Sz_001    = Sz_110
                    Sz_src    = "PDOS integral"

            # ---- In-plane orbital moment  L(110) ----------------------------
            # For SAXIS=[1,1,0]: Lx≈Ly, Lz≈0  =>  |L_110| = sqrt(Lx²+Ly²+Lz²)
            if Lz_z_dict and ion in Lz_z_dict:
                L_ip     = Lz_z_dict[ion]
                L_ip_src = "direct"
            elif '110' in self.orbital_moments.get(ion, {}):
                lx, ly, lz = self.orbital_moments[ion]['110']
                L_ip     = np.sqrt(lx**2 + ly**2 + lz**2)
                L_ip_src = f"OUTCAR (Lx={lx:+.4f} Ly={ly:+.4f} Lz={lz:+.4f})"
            else:
                L_ip     = None
                L_ip_src = "MISSING"

            # ---- Out-of-plane orbital moment  L(001) ------------------------
            # For SAXIS=[0,0,1]: Lx≈Ly≈0, Lz≠0  =>  |L_001| = |Lz|
            if Lz_x_dict and ion in Lz_x_dict:
                L_op     = Lz_x_dict[ion]
                L_op_src = "direct"
            elif '001' in self.orbital_moments.get(ion, {}):
                lx, ly, lz = self.orbital_moments[ion]['001']
                L_op     = np.sqrt(lx**2 + ly**2 + lz**2)
                L_op_src = f"OUTCAR (Lx={lx:+.4f} Ly={ly:+.4f} Lz={lz:+.4f})"
            else:
                L_op     = None
                L_op_src = "MISSING"

            # ---- handle missing orbital moments ----------------------------
            if L_ip is None or L_op is None:
                any_missing_L = True
                missing = [n for n, v in
                           [("L_inplane(110)", L_ip), ("L_outofplane(001)", L_op)]
                           if v is None]
                print(f"\n  *** Ion {ion+1}: {', '.join(missing)} NOT available. ***")
                print("      Call set_moments_directly(ion,")
                print("          L_inplane, L_outofplane, Sz_inplane, Sz_outofplane)")
                if L_ip is None: L_ip = 0.0
                if L_op is None: L_op = 0.0

            # ---- K^(1) with direction-specific Sz --------------------------
            # Physical sign convention:
            #   The easy axis has LARGER orbital moment -> LOWER SOC energy.
            #   E^(1)(n) = -1/2 Int Im Tr[G0 Vso]  (Eq.15, negative prefactor)
            #   Larger L*Sz -> more negative E^(1) -> lower energy -> easy axis.
            #   K^(1) = E^(1)(110) - E^(1)(001)
            #         = -xi * [L_110*Sz_110 - L_001*Sz_001]
            #         =  xi * [L_001*Sz_001 - L_110*Sz_110]
            #   If L_110 > L_001 (your system) -> K^(1) < 0 -> in-plane ✓
            term_110   = L_ip * Sz_110
            term_001   = L_op * Sz_001
            K1_ion_eV  = xi * (term_001 - term_110)   # note: 001 minus 110

            print(f"\n  Ion {ion+1}:")
            print(f"    xi                         = {xi:.4f} eV")
            print(f"    L_inplane    (110 OUTCAR)  = {L_ip:+.4f} hbar  [{L_ip_src}]")
            print(f"    Sz_inplane   (110 OUTCAR)  = {Sz_110:+.4f} muB   [{Sz_src}]")
            print(f"    L*Sz (110)                 = {term_110:+.5f}")
            print(f"    L_outofplane (001 OUTCAR)  = {L_op:+.4f} hbar  [{L_op_src}]")
            print(f"    Sz_outofplane(001 OUTCAR)  = {Sz_001:+.4f} muB   [{Sz_src}]")
            print(f"    L*Sz (001)                 = {term_001:+.5f}")
            print(f"    (L*Sz)_110 - (L*Sz)_001   = {term_110-term_001:+.5f}")
            print(f"    K^(1)_ion  = xi * diff     = {K1_ion_eV*1000:+.4f} meV")

            site_info[ion] = {
                'xi': xi,
                'L_inplane'   : L_ip,   'L_ip_src'  : L_ip_src,
                'L_outofplane': L_op,   'L_op_src'  : L_op_src,
                'Sz_110'      : Sz_110, 'Sz_001'    : Sz_001,
                'Sz_src'      : Sz_src,
                'delta_L'     : L_ip - L_op,
                'K1_meV'      : K1_ion_eV * 1000,
            }
            K1_total += K1_ion_eV

        K1_total_meV = K1_total * 1000
        sign_str = "in-plane (110)" if K1_total_meV < 0 else "out-of-plane (001)"
        print(f"\n  K^(1) total = {K1_total_meV:+.4f} meV  =>  {sign_str} easy axis")

        if any_missing_L:
            print("\n  *** WARNING: Some orbital moments missing (set to 0). "
                  "K^(1) is incomplete.")

        return K1_total_meV, site_info

    # ------------------------------------------------------------------

    def analyze_degenerate_subspace(self, ion_idx, xi, energy_range=(-0.5, 0.5)):
        """
        Degenerate first-order analysis for (dxz, dyz) doublet (Sec. 6).

        Quantifies whether a near-degenerate (dxz, dyz) manifold exists at E_F,
        computes the expected SOC-driven linear splitting, and estimates the
        contribution to K^(1) from this active subspace (Sec. 8 / Eq. 47).

        Physics
        -------
        Within span{|xz⟩, |yz⟩}, Lz has off-diagonal form (Eq. 30):
          Lz|{xz,yz}} = ħ [[0, -i], [i, 0]]
        Eigenvectors are  |±1⟩ = (1/√2)(|xz⟩ ± i|yz⟩)  with  E^(1)_± = ±ξ Sz ħ
        (Eqs. 33–35).

        The splitting is linear in ξ and proportional to the DOS at E_F.

        Returns: diagnostic dict
        """
        if ion_idx not in self.pdos_data:
            return {}

        print("\n" + "=" * 70)
        print(f"DEGENERATE SUBSPACE ANALYSIS — Ion {ion_idx+1}  [Sec. 6]")
        print("=" * 70)
        print("  Doublet: span{|dxz⟩, |dyz⟩}  ←→  m = ±1 complex orbitals")
        print(f"  Energy window relative to E_F: [{energy_range[0]}, {energy_range[1]}] eV")
        print()

        data   = self.pdos_data[ion_idx]
        energy = data['energy']
        ef     = self.efermi

        # DOS at Fermi level for dxz, dyz
        fermi_idx = np.argmin(np.abs(energy - ef))
        dos_xz_up = data['dxz']['up'][fermi_idx]
        dos_xz_dn = data['dxz']['dn'][fermi_idx]
        dos_yz_up = data['dyz']['up'][fermi_idx]
        dos_yz_dn = data['dyz']['dn'][fermi_idx]

        # Spectral weight in window
        e_lo = ef + energy_range[0]
        e_hi = ef + energy_range[1]
        mask = (energy >= e_lo) & (energy <= e_hi)

        N_xz_up = trapz(data['dxz']['up'][mask], energy[mask]) if mask.sum() > 1 else 0.0
        N_xz_dn = trapz(data['dxz']['dn'][mask], energy[mask]) if mask.sum() > 1 else 0.0
        N_yz_up = trapz(data['dyz']['up'][mask], energy[mask]) if mask.sum() > 1 else 0.0
        N_yz_dn = trapz(data['dyz']['dn'][mask], energy[mask]) if mask.sum() > 1 else 0.0

        # Energy splitting from Eq. (35):  E^(1)_± = ±ξ Sz ħ
        sz_up = +0.5
        sz_dn = -0.5
        split_up = xi * sz_up          # eV
        split_dn = xi * sz_dn          # eV

        # Estimate K^(1)_active from orbital-moment route (Sec. 8):
        # The up-spin m=+1 state gains energy +ξ/2, m=-1 loses ξ/2.
        # Net orbital moment of occupied (E < EF) complex states:
        #   <Lz>_active ≈ N_{+1} − N_{-1}
        # In diagonal approx N_{+1} ≈ N_{-1} so we report this as a bound.
        N_p1_up = 0.5 * (N_xz_up + N_yz_up)
        N_m1_up = 0.5 * (N_xz_up + N_yz_up)
        N_p1_dn = 0.5 * (N_xz_dn + N_yz_dn)
        N_m1_dn = 0.5 * (N_xz_dn + N_yz_dn)

        # Near-degeneracy criterion: both dxz and dyz have significant DOS at EF
        dos_xz_total = dos_xz_up + dos_xz_dn
        dos_yz_total = dos_yz_up + dos_yz_dn
        is_near_deg  = (min(dos_xz_total, dos_yz_total) /
                        max(dos_xz_total, dos_yz_total + 1e-10) > 0.3)

        print(f"  DOS at E_F:")
        print(f"    dxz : {dos_xz_up:.4f}(↑)  {dos_xz_dn:.4f}(↓)  total={dos_xz_total:.4f}")
        print(f"    dyz : {dos_yz_up:.4f}(↑)  {dos_yz_dn:.4f}(↓)  total={dos_yz_total:.4f}")
        print()
        print(f"  Near-degeneracy at E_F? → {'YES ✓' if is_near_deg else 'NO (weak doublet)'}")
        print()

        print(f"  Linear splitting from Eq. (35)  E^(1)_± = ±ξ·Sz·ħ:")
        print(f"    E^(1)_+ (↑ spin) = +{abs(split_up)*1000:.2f} meV   "
              f"[|+1⟩ = (|xz⟩ + i|yz⟩)/√2  gains energy for ↑]")
        print(f"    E^(1)_− (↑ spin) = −{abs(split_up)*1000:.2f} meV")
        print(f"    E^(1)_+ (↓ spin) = −{abs(split_dn)*1000:.2f} meV")
        print(f"    E^(1)_− (↓ spin) = +{abs(split_dn)*1000:.2f} meV")
        print()

        print("  Complex-orbital occupancies (diagonal approx — see warning above):")
        print(f"    N_{{+1,↑}} ≈ {N_p1_up:.4f}   N_{{-1,↑}} ≈ {N_m1_up:.4f}")
        print(f"    N_{{+1,↓}} ≈ {N_p1_dn:.4f}   N_{{-1,↓}} ≈ {N_m1_dn:.4f}")
        print(f"    Orbital polarisation ΔN = N_{{+1}} − N_{{-1}} ≈ 0.0 (diagonal approx)")
        print()

        if is_near_deg:
            print("  *** Doublet is active → first-order SOC is relevant (Sec. 8). ***")
            print("      Treat SOC exactly within {xz,yz}; treat coupling to rest 2nd order.")
        else:
            print("  Doublet is not strongly active → 2nd-order treatment is sufficient.")

        return {
            'is_near_degenerate': is_near_deg,
            'dos_xz_ef': dos_xz_total,
            'dos_yz_ef': dos_yz_total,
            'split_meV': abs(split_up) * 1000,
            'N_complex_up': (N_p1_up, N_m1_up),
            'N_complex_dn': (N_p1_dn, N_m1_dn),
        }

    # ==================================================================
    # ███  SECOND-ORDER SOC  ███
    # ==================================================================

    def _susceptibility_green(self, ion_idx, orb1, orb2, spin1, spin2,
                               energy_range, eta=0.01):
        """
        χ^{σσ'}_{mm'}(EF) = ∫^{EF} dE/π  Im[g0^σ_m(E) g0^{σ'}_{m'}(E)]

        Uses the SCALAR-RELATIVISTIC G0 spectral weights (pdos_G0_data) when
        available — call load_G0_doscar() first for correct K^(2).
        Falls back to pdos_data (SOC DOSCAR) with a warning if G0 not loaded.

        Im g0^σ_m(E) ≈ -π ρ0^σ_m(E)    [Eq. 5 of paper, G0 version]
        Re g0 recovered via Kramers-Kronig Hilbert transform.
        """
        # ---- choose G0 PDOS (preferred) or SOC PDOS (fallback) --------
        if self.pdos_G0_data:
            if ion_idx not in self.pdos_G0_data:
                return 0.0
            data   = self.pdos_G0_data[ion_idx]
            efermi = self.efermi_G0 if self.efermi_G0 is not None else self.efermi
        else:
            if ion_idx not in self.pdos_data:
                return 0.0
            data   = self.pdos_data[ion_idx]
            efermi = self.efermi

        energy = data['energy']
        if orb1 not in data or orb2 not in data:
            return 0.0

        dos1 = data[orb1][spin1]
        dos2 = data[orb2][spin2]

        e_min  = efermi + energy_range[0]
        e_max  = efermi + energy_range[1]
        e_mask = (energy >= e_min) & (energy <= e_max)

        # ---- occupied states only (integration limit = E_F) -----------
        occ_mask = e_mask & (energy <= efermi)
        if occ_mask.sum() < 2:
            return 0.0

        e_occ  = energy[occ_mask]
        d1_occ = dos1[occ_mask]
        d2_occ = dos2[occ_mask]

        # ---- Re g0 via Kramers-Kronig Hilbert transform ---------------
        e_full  = energy[e_mask]
        d1_full = dos1[e_mask]
        d2_full = dos2[e_mask]

        def real_green(E_val, e_grid, dos_grid, eta_val):
            denom = E_val - e_grid
            return np.trapz(dos_grid * denom / (denom**2 + eta_val**2), e_grid)

        # Im[g1·g2] = Re(g1)·Im(g2) + Im(g1)·Re(g2)
        integrand = np.zeros(occ_mask.sum())
        for k, E in enumerate(e_occ):
            Re_g1 = real_green(E, e_full, d1_full, eta)
            Re_g2 = real_green(E, e_full, d2_full, eta)
            Im_g1 = -np.pi * d1_occ[k]
            Im_g2 = -np.pi * d2_occ[k]
            integrand[k] = Re_g1 * Im_g2 + Im_g1 * Re_g2

        chi = trapz(integrand, e_occ) / np.pi
        return chi

    # ------------------------------------------------------------------

    def calculate_all_susceptibilities(self, ion_idx, energy_range=(-7.5, 6.0)):
        """
        Compute the full 5×5×2×2 set of  χ^{σσ'}_{mm'}  (Eq. 17)
        and the derived  χ'_{mm'} = χ^{↑↑} + χ^{↓↓} − χ^{↑↓} − χ^{↓↑}  (Eq. 11 analogue).

        Returns (susceptibilities, chi_prime_values)
        """
        print("\n" + "=" * 70)
        print(f"SECOND-ORDER SUSCEPTIBILITIES  chi^{{ss'}}  [Sec. 4 / Eq. (17)]")
        print(f"Ion {ion_idx+1}   E-range: {energy_range[0]} to {energy_range[1]} eV (rel. E_F)")
        if self.pdos_G0_data:
            print("  Spectral weights: G0 (scalar-relativistic DOSCAR, LSORBIT=F)  [CORRECT]")
        else:
            print("  WARNING: No G0 DOSCAR loaded — using SOC DOSCAR (LSORBIT=T).")
            print("  Call load_G0_doscar(path) before this for correct K^(2).")
        print("=" * 70)

        spins = ['up', 'dn']
        sus   = {}

        total_pairs = len(D_ORBITALS)**2 * 4
        done = 0
        for orb1 in D_ORBITALS:
            for orb2 in D_ORBITALS:
                for s1 in spins:
                    for s2 in spins:
                        key = f"chi_{s1[0]}{s2[0]}_{orb1},{orb2}"
                        sus[key] = self._susceptibility_green(
                            ion_idx, orb1, orb2, s1, s2, energy_range)
                        done += 1
                        if done % 20 == 0:
                            print(f"    … {done}/{total_pairs} computed", end='\r')

        print(f"    … {total_pairs}/{total_pairs} computed ✓")

        # ---- χ'_{mm'} ------------------------------------------------
        chi_prime = {}
        for orb1 in D_ORBITALS:
            for orb2 in D_ORBITALS:
                uu = sus[f"chi_uu_{orb1},{orb2}"]
                dd = sus[f"chi_dd_{orb1},{orb2}"]
                ud = sus[f"chi_ud_{orb1},{orb2}"]
                du = sus[f"chi_du_{orb1},{orb2}"]
                chi_prime[f"chi'_{orb1},{orb2}"] = uu + dd - ud - du

        # ---- Print summary of significant terms ----------------------
        chi_prime_hdr = "chi'"
        print("\n  chi'_{mm'} = chi^{uu} + chi^{dd} - chi^{ud} - chi^{du}  (directional part):")
        print(f"  {'Pair':<25}  {'chi_uu':>10}  {'chi_dd':>10}  "
              f"{'chi_ud':>10}  {'chi_du':>10}  {chi_prime_hdr:>12}")
        print("  " + "-"*82)
        for orb1 in D_ORBITALS:
            for orb2 in D_ORBITALS:
                pkey  = f"chi'_{orb1},{orb2}"
                cp    = chi_prime[pkey]
                if abs(cp) < 1e-6:
                    continue
                uu = sus[f"chi_uu_{orb1},{orb2}"]
                dd = sus[f"chi_dd_{orb1},{orb2}"]
                ud = sus[f"chi_ud_{orb1},{orb2}"]
                du = sus[f"chi_du_{orb1},{orb2}"]
                flag = "  ★★★" if abs(cp) > 0.01 else ("  ★★" if abs(cp) > 0.001 else "")
                print(f"  {orb1}↔{orb2:<18}  {uu:>10.5f}  {dd:>10.5f}  "
                      f"{ud:>10.5f}  {du:>10.5f}  {cp:>12.6f}{flag}")

        return sus, chi_prime

    # ------------------------------------------------------------------

    def calculate_second_order_mae(self, ion_idx, chi_prime, xi):
        """
        K^(2) from E^(2) [paper Eqs. 16, 27-28] for K = E(110) - E(001).

        Formula (Ke & van Schilfgaarde 2015, their Eq. 15) gives
        4K/xi^2 for K = E(100) - E(001).  For K = E(110) - E(001) the
        spin matrix-element weights are different; the dominant m=+/-1
        terms carry the same sign so the formula is used as a proxy here.
        A negative K^(2) means in-plane easy axis.

          4K^(2)/xi^2 = 4*chi'_{dxy,dx2-y2}
                      +   chi'_{dyz,dxz}
                      - 3/2 * [chi'_{dyz,dz2} + chi'_{dz2,dxz}]
                      - 1/2 * [chi'_{dxy,dyz} + chi'_{dxy,dxz}
                               + chi'_{dyz,dx2-y2} + chi'_{dxz,dx2-y2}]
        """
        print("\n" + "=" * 70)
        print("SECOND-ORDER MAE  K^(2)  [paper Eq. 27-28 / Ke-vSG 2015 Eq. 15]")
        print("  Convention: K^(2) = E^(2)(110) - E^(2)(001)")
        print("  K^(2) < 0  =>  in-plane (110) easy axis")
        print("=" * 70)
        print(f"  xi(Co) = {xi} eV")

        def cp(o1, o2):
            v  = chi_prime.get(f"chi'_{o1},{o2}", 0.0)
            vT = chi_prime.get(f"chi'_{o2},{o1}", 0.0)
            return 0.5 * (v + vT)

        terms = {}
        terms['4*chi\'_{dxy,dx2-y2}']               = 4   * cp('dxy', 'dx2-y2')
        terms['chi\'_{dyz,dxz}']                    = 1   * cp('dyz', 'dxz')
        terms['-3/2*[chi\'_{dyz,dz2}+chi\'_{dz2,dxz}]'] = -1.5 * (cp('dyz', 'dz2') +
                                                                     cp('dz2', 'dxz'))
        inter = (cp('dxy', 'dyz') + cp('dxy', 'dxz') +
                 cp('dyz', 'dx2-y2') + cp('dxz', 'dx2-y2'))
        terms['-1/2*[inter-orbital mix]']            = -0.5 * inter

        total_4K_xi2 = sum(terms.values())
        K2_meV = total_4K_xi2 * xi**2 / 4 * 1000   # meV

        col1 = 44
        print(f"\n  {'Term':<{col1}}  {'4K/xi2 contrib':>14}")
        print("  " + "-" * (col1 + 18))
        for name, val in terms.items():
            flag = "  *** large" if abs(val) > 0.05 else ("  ** notable" if abs(val) > 0.01 else "")
            print(f"  {name:<{col1}}  {val:>14.6f}{flag}")
        print("  " + "-" * (col1 + 18))
        print(f"  {'4K^(2)/xi^2  (total)':<{col1}}  {total_4K_xi2:>14.6f}")
        sign_str = "in-plane (110)" if K2_meV < 0 else "out-of-plane (001)"
        print(f"\n  K^(2) = {K2_meV:+.4f} meV  =>  {sign_str}")

        return K2_meV, terms

    # ==================================================================
    # ███  COMBINED MAE + DIAGNOSTICS  ███
    # ==================================================================

    def full_mae_analysis(
            self, ion_idx, xi,
            energy_range=(-7.5, 6.0),
            # Optional first-order inputs
            Lz_z=None, Lz_x=None,
            off_diag_xzyz_up=None, off_diag_xzyz_dn=None,
            spin_z=None):
        """
        Complete analysis for one ion — first + second order.

        Steps
        -----
        1. Fermi-level DOS diagnostics
        2. Degenerate (dxz, dyz) subspace analysis (Sec. 6)
        3. Complex orbital PDOS construction (Sec. 7)
        4. First-order K^(1) (Sec. 5)
        5. Second-order χ'_{mm'} and K^(2) (Sec. 4)
        6. Combined K ≈ K^(1) + K^(2)    (Sec. 8 / Eq. 47)
        """
        print("\n" + "#" * 70)
        print(f"#  FULL MAE ANALYSIS — Ion {ion_idx+1}    ξ = {xi} eV")
        print("#" + "#" * 69)

        # ---- 1. Fermi-level diagnostics --------------------------------
        self._fermi_diagnostics(ion_idx)

        # ---- 2. Degenerate subspace ------------------------------------
        deg_info = self.analyze_degenerate_subspace(ion_idx, xi,
                                                     energy_range=(-0.5, 0.5))

        # ---- 3. Orbital moment (complex PDOS) -------------------------
        Lz_calc, lz_diag = self.calculate_orbital_moment_complex(
            ion_idx,
            off_diag_xzyz_up=off_diag_xzyz_up,
            off_diag_xzyz_dn=off_diag_xzyz_dn,
            energy_range=(energy_range[0], 0))

        # ---- 4. First-order K^(1) -------------------------------------
        # Prefer externally supplied Lz values (from OUTCAR) over PDOS approx
        _Lz_z_dict = {ion_idx: Lz_z}   if Lz_z is not None else None
        _Lz_x_dict = {ion_idx: Lz_x}   if Lz_x is not None else None
        _spin_dict  = {ion_idx: spin_z} if spin_z is not None else None

        K1_meV, site_info = self.calculate_first_order_mae(
            ion_indices=[ion_idx],
            xi_values={ion_idx: xi},
            Lz_z_dict=_Lz_z_dict,
            Lz_x_dict=_Lz_x_dict,
            spin_z_dict=_spin_dict,
            energy_range=(energy_range[0], 0))

        # ---- 5. Second-order K^(2) ------------------------------------
        sus, chi_prime = self.calculate_all_susceptibilities(ion_idx, energy_range)
        K2_meV, terms  = self.calculate_second_order_mae(ion_idx, chi_prime, xi)

        # ---- 6. Combined MAE ------------------------------------------
        K_total_meV = K1_meV + K2_meV

        print("\n" + "=" * 70)
        print("COMBINED MAE RESULT  [Sec. 8 / Eq. (47)]:  K ≈ K^(1) + K^(2)")
        print("=" * 70)
        print(f"  K^(1)   = {K1_meV:+.4f} meV   (first-order, degenerate SOC)")
        print(f"  K^(2)   = {K2_meV:+.4f} meV   (second-order, orbital-pair susceptibility)")
        print(f"  ─────────────────────────────")
        print(f"  K_total = {K_total_meV:+.4f} meV")
        print()
        if K_total_meV > 0:
            print("  Easy axis: OUT-OF-PLANE  (K > 0)")
        else:
            print("  Easy axis: IN-PLANE  (K < 0)")

        if deg_info.get('is_near_degenerate'):
            print("\n  ⚠  Near-degenerate (dxz, dyz) doublet detected at E_F.")
            print("     K^(1) may be underestimated without off-diagonal PDOS or OUTCAR moments.")
            print("     Run two VASP+SOC calculations (SAXIS = 001 and 100) and")
            print("     supply Lz_z / Lz_x from OUTCAR for a reliable K^(1).")

        return {
            'K1_meV': K1_meV,
            'K2_meV': K2_meV,
            'K_total_meV': K_total_meV,
            'degenerate_subspace': deg_info,
            'site_first_order': site_info,
            'chi_prime': chi_prime,
            'K2_terms': terms,
        }

    # ------------------------------------------------------------------

    def _fermi_diagnostics(self, ion_idx):
        """Print DOS at/around E_F for all d-orbitals."""
        if ion_idx not in self.pdos_data:
            return
        data   = self.pdos_data[ion_idx]
        energy = data['energy']
        ef     = self.efermi
        win    = 0.3
        mask   = np.abs(energy - ef) < win

        print(f"\nFermi-level DOS diagnostics — Ion {ion_idx+1}:")
        print(f"  {'Orbital':<10}  {'DOS(↑)@EF':>12}  {'DOS(↓)@EF':>12}  {'d-band occ (↑)':>15}  {'d-band occ (↓)':>15}")
        print("  " + "-" * 70)

        for orb in D_ORBITALS:
            fi  = np.argmin(np.abs(energy - ef))
            d_up_ef = data[orb]['up'][fi]
            d_dn_ef = data[orb]['dn'][fi]

            occ_mask = energy <= ef
            occ_up = trapz(data[orb]['up'][occ_mask], energy[occ_mask])
            occ_dn = trapz(data[orb]['dn'][occ_mask], energy[occ_mask])

            flag = "  ← high DOS" if (d_up_ef + d_dn_ef) > 0.5 else ""
            print(f"  {orb:<10}  {d_up_ef:>12.4f}  {d_dn_ef:>12.4f}  "
                  f"{occ_up:>15.4f}  {occ_dn:>15.4f}{flag}")

    # ------------------------------------------------------------------

    def plot_chi_prime_heatmap(self, chi_prime, ion_idx, filename=None):
        """Visualise the 5×5 χ'_{mm'} matrix as a heatmap."""
        N   = len(D_ORBITALS)
        mat = np.zeros((N, N))
        for i, o1 in enumerate(D_ORBITALS):
            for j, o2 in enumerate(D_ORBITALS):
                mat[i, j] = chi_prime.get(f"chi'_{o1},{o2}", 0.0)

        vmax = np.max(np.abs(mat)) + 1e-9
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="χ'$_{mm'}$ (states/eV)")
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(D_ORBITALS, rotation=45, ha='right')
        ax.set_yticklabels(D_ORBITALS)
        ax.set_title(f"Ion {ion_idx+1}  χ'$_{{mm'}}$ = χ$^{{↑↑}}$+χ$^{{↓↓}}$−χ$^{{↑↓}}$−χ$^{{↓↑}}$",
                     fontsize=12)
        # Annotate values
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha='center', va='center',
                        fontsize=7, color='k' if abs(mat[i,j]) < 0.5*vmax else 'w')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"  Saved χ' heatmap → {filename}")
        plt.show()

    # ------------------------------------------------------------------

    def extract_xi_from_soc_matrix(self, tag):
        """
        Extract the effective SOC parameter xi_i for each ion from the
        l=2 diagonal block of the SOC matrix stored in self.soc_matrix[tag].

        The on-site SOC energy is:
            E_soc^i = Tr[rho_i * H_soc^i]  = xi_i * Tr[rho_i * (L.S)]

        A practical estimate of xi_i comes from the off-diagonal matrix
        elements of H_soc (l=2 block). For d-electrons the dominant
        coupling is between dxz/dyz (m=+/-1), and the matrix element
        <dxy|H_soc|dyz> = -i*xi/2 in the real basis, giving:

            xi_i ≈ 2 * |<dxy|H_soc|dyz>|   (from [0,1] element of the matrix)

        This is consistent with the standard Co xi ~ 65 meV but allows
        site-resolved values reflecting the actual local environment.

        Returns: {ion_0idx: xi_eV}
        """
        if not hasattr(self, 'soc_matrix') or tag not in self.soc_matrix:
            print("  No SOC matrix data. Call read_esoc_from_outcar() first.")
            return {}

        xi_dict = {}
        # dxy=row0, dyz=row1, dz2=row2, dxz=row3, dx2-y2=row4
        # Key off-diagonal pairs and their theoretical prefactor:
        #   <dxy|H|dyz> = -xi/2   =>  xi = 2 * |element|
        #   <dxy|H|dxz> = -xi/2   =>  xi = 2 * |element|
        #   <dyz|H|dz2> = -sqrt(3)*xi/2  =>  xi = 2/sqrt(3) * |element|
        #   average for robustness
        import math
        for ion_0idx, mat in self.soc_matrix[tag].items():
            estimates = []
            if abs(mat[0, 1]) > 1e-6:
                estimates.append(2.0 * abs(mat[0, 1]))          # dxy-dyz
            if abs(mat[0, 3]) > 1e-6:
                estimates.append(2.0 * abs(mat[0, 3]))          # dxy-dxz
            if abs(mat[1, 2]) > 1e-6:
                estimates.append(2.0 / math.sqrt(3) * abs(mat[1, 2]))  # dyz-dz2
            if abs(mat[2, 3]) > 1e-6:
                estimates.append(2.0 / math.sqrt(3) * abs(mat[2, 3]))  # dz2-dxz
            xi_dict[ion_0idx] = float(np.mean(estimates)) if estimates else 0.065

        return xi_dict

    # ------------------------------------------------------------------

    def extract_xi_from_esoc(self, tag):
        """
        Extract per-ion xi from the OUTCAR E_soc values using:

            xi_i = |E_soc_i| / |<L_i · S_i>|
                 ≈ |E_soc_i| / (|L_i| · |Sz_i|)        [collinear FM approx]

        This is the most direct approach: E_soc is the actual SOC energy
        expectation value for each ion, so dividing by the L·S product gives
        the effective coupling constant screened by the crystal environment.

        For Co in a crystal field this is typically 20–50 meV, smaller than
        the free-atom value of ~65 meV due to hybridisation.

        Requires:
          - read_esoc_from_outcar(path, tag)    → self.esoc[tag]
          - read_outcar_single(path, tag)       → self.orbital_moments[ion][tag]
                                                   self.spin_moments[ion][tag]

        Parameters
        ----------
        tag : '110' or '001'

        Returns
        -------
        dict {ion_0idx: xi_eV}
        """
        if not hasattr(self, 'esoc') or tag not in self.esoc:
            print(f"  No E_soc data for tag '{tag}'. "
                  "Call read_esoc_from_outcar() first.")
            return {}

        xi_dict = {}
        print(f"\n  xi from E_soc (tag='{tag}'):")
        print(f"  {'Ion':>4}  {'|E_soc|(meV)':>14}  "
              f"{'|L|(hbar)':>12}  {'|Sz|(muB)':>11}  "
              f"{'xi (meV)':>10}  {'xi (eV)':>10}")
        print("  " + "-" * 68)

        for ion in sorted(self.esoc[tag]):
            esoc_val = self.esoc[tag][ion]

            # Orbital moment magnitude
            om = self.orbital_moments.get(ion, {}).get(tag)
            if om is not None:
                lx, ly, lz = om
                L_mag = np.sqrt(lx**2 + ly**2 + lz**2)
            else:
                L_mag = None

            # Spin moment
            Sz = self.spin_moments.get(ion, {}).get(tag)

            if L_mag is not None and Sz is not None and L_mag * abs(Sz) > 1e-6:
                xi = abs(esoc_val) / (L_mag * abs(Sz))
                xi_dict[ion] = xi
                print(f"  {ion+1:>4}  {abs(esoc_val)*1000:>14.4f}  "
                      f"{L_mag:>12.4f}  {abs(Sz):>11.4f}  "
                      f"{xi*1000:>10.3f}  {xi:>10.6f}")
            else:
                # Fallback: use SOC matrix off-diagonal if available
                xi_raw = self.extract_xi_from_soc_matrix(tag)
                xi = xi_raw.get(ion, 0.065)
                xi_dict[ion] = xi
                reason = "L=0" if L_mag is not None and L_mag < 1e-6 else "L/Sz missing"
                print(f"  {ion+1:>4}  {abs(esoc_val)*1000:>14.4f}  "
                      f"  [{reason} — using SOC matrix: xi={xi*1000:.2f} meV]")

        return xi_dict

    # ------------------------------------------------------------------

    def print_key_formulas_used(self):
        """Print the key formulas from the paper that are implemented here."""
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║            KEY FORMULAS IMPLEMENTED  (H. Sabri 2026)                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Dyson:     G = G0 + G0·Vso·G0 + G0·Vso·G0·Vso·G0 + …          (Sec 3) ║
║                                                                          ║
║  E^(1)(n̂) = −(1/2) ∫^{EF} dE/π  Im Tr[G0(E) Vso(n̂)]           (Eq 12) ║
║           ≈ Σ_i ξ_i <L_n̂>_i <S_n̂>_i              (collinear, Eq 26)   ║
║                                                                          ║
║  E^(2)(n̂) = −(1/2) ∫^{EF} dE/π  Im Tr[G0 Vso G0 Vso]          (Eq 13) ║
║           = −(1/2) Σ_i ξ_i² Σ_{mm'σσ'}                                 ║
║              |<m,σ|l·s|m',σ'>|²  χ^{σσ'}_{mm'}                 (Eq 18) ║
║                                                                          ║
║  χ^{σσ'}_{mm'} = ∫^{EF} dE/π Im[g^σ_m(E) g^{σ'}_{m'}(E)]      (Eq 17) ║
║                                                                          ║
║  Degenerate (xz,yz): |±1⟩ = (|xz⟩ ± i|yz⟩)/√2                 (Eq 33) ║
║                      E^(1)_± = ±ξ Sz ħ                          (Eq 35) ║
║                                                                          ║
║  ⟨Lz⟩_i = ħ Σ_σ ∫^{EF} dE [ρ_{+1,σ} − ρ_{-1,σ}]              (Eq 45) ║
║           (needs off-diagonal A_{xz,yz} for non-zero result)    (Eq 43) ║
║                                                                          ║
║  Combined:  K ≈ K^(1)_active + K^(2)_rest                       (Eq 47) ║
╚══════════════════════════════════════════════════════════════════════════╝
        """)


# ===========================================================================
# Main driver
# ===========================================================================

def main():
    # ===================================================================
    #  PATH CONFIGURATION — set your folder paths here
    # ===================================================================

    # SOC calculation with SAXIS = 1 1 0  (in-plane magnetisation)
    FOLDER_SOC_110 = '/Users/houssam/327/SYM/SOC_NSCF/FM110/'

    # SOC calculation with SAXIS = 0 0 1  (out-of-plane magnetisation)
    FOLDER_SOC_001 = '/Users/houssam/327/SYM/SOC_NSCF/FM001/'

    # Non-SOC collinear calculation (LSORBIT=F, ISPIN=2, LORBIT=11, ISMEAR=-5)
    # Provides G0 spectral weights for the second-order susceptibility K^(2)
    FOLDER_NONREL  = '/Users/houssam/327/SYM/NoSOC/SCF_New/SCF/'

    # Ion indices (0-based) for the 4 Co atoms in the unit cell
    # Verify from your POSCAR atom ordering or from the Sz values in
    # the magnetization(z) block of your OUTCAR:
    #   Co1 (displaced, large orbital moment): OUTCAR ions 8 and 10  -> 0-idx 7 and 9
    #   Co2 (less displaced):                  OUTCAR ions 7 and 9   -> 0-idx 6 and 8
    CO1_IONS = [7, 9]   # 0-based indices of Co1 sites
    CO2_IONS = [6, 8]   # 0-based indices of Co2 sites

    # DFT total-energy reference (for final comparison)
    K_DFT_MEV = -234.0  # meV,  K = E_SOC(110) - E_SOC(001)

    # ===================================================================
    #  END OF PATH CONFIGURATION
    # ===================================================================

    print("\n" + "#" * 70)
    print("#  Sr3Co2O7  —  MAE Decomposition: K^(1) + K^(2)")
    print("#  K = E_SOC(110) - E_SOC(001)   [< 0 => in-plane easy axis]")
    print(f"#  DFT reference: K = {K_DFT_MEV:.1f} meV")
    print("#" * 70)

    all_co_ions = CO1_IONS + CO2_IONS

    # Build file paths
    doscar_soc_110 = os.path.join(FOLDER_SOC_110, 'DOSCAR')
    doscar_nonrel  = os.path.join(FOLDER_NONREL,  'DOSCAR')
    outcar_110     = os.path.join(FOLDER_SOC_110, 'OUTCAR')
    outcar_001     = os.path.join(FOLDER_SOC_001, 'OUTCAR')

    # ===================================================================
    # STEP 1 — Load SOC DOSCAR (110 run)
    # Used for Fermi-level diagnostics and PDOS plots only.
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 1 — Load SOC DOSCAR (LSORBIT=T, SAXIS=110)")
    print("=" * 70)
    analyzer = VaspMAEAnalyzer(doscar_soc_110)
    analyzer.print_key_formulas_used()

    if not analyzer.read_doscar():
        print(f"ERROR: cannot read {doscar_soc_110}")
        return

    analyzer._co1_ions = CO1_IONS
    analyzer._co2_ions = CO2_IONS

    for ion in all_co_ions:
        if ion not in analyzer.pdos_data:
            print(f"ERROR: ion {ion} not found in DOSCAR.")
            print(f"  Available ions: {sorted(analyzer.pdos_data)}")
            print("  Adjust CO1_IONS / CO2_IONS in the PATH CONFIGURATION block.")
            return

    # ===================================================================
    # STEP 2 — Load non-SOC (G0) DOSCAR for K^(2)
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 2 — Load non-SOC G0 DOSCAR (LSORBIT=F) for K^(2)")
    print("=" * 70)
    if os.path.isfile(doscar_nonrel):
        analyzer.load_G0_doscar(doscar_nonrel)
    else:
        print(f"  WARNING: not found: {doscar_nonrel}")
        print("  K^(2) will use SOC DOSCAR as fallback (less accurate).")

    # ===================================================================
    # STEP 3 — Read both SOC OUTCARs
    # Extracts per-ion: Lx, Ly, Lz, Sz, E_soc, l=2 SOC matrix
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 3 — Read SOC OUTCARs (Lx, Ly, Lz, Sz, E_soc, SOC matrix)")
    print("=" * 70)

    have_110 = os.path.isfile(outcar_110)
    have_001 = os.path.isfile(outcar_001)

    if have_110:
        analyzer.read_outcar_single(outcar_110, tag='110')
        analyzer.read_esoc_from_outcar(outcar_110, tag='110')
    else:
        print(f"  WARNING: not found: {outcar_110}")
        print("  Orbital moments for [110] direction unavailable.")

    if have_001:
        analyzer.read_outcar_single(outcar_001, tag='001')
        analyzer.read_esoc_from_outcar(outcar_001, tag='001')
    else:
        print(f"  WARNING: not found: {outcar_001}")
        print("  Orbital moments for [001] direction unavailable.")

    if not have_110 and not have_001:
        print("\n  ERROR: At least one OUTCAR is required to compute K^(1).")
        print("  Check FOLDER_SOC_110 and FOLDER_SOC_001 in PATH CONFIGURATION.")
        return

    # ===================================================================
    # STEP 4 — Extract per-ion xi from the l=2 SOC matrix
    # xi_i = 2 * |<dxy|H_soc|dyz>|  (off-diagonal element, real basis)
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 4 — Per-ion xi from E_soc")
    print("  xi_i = |E_soc_i| / (|L_i| * |Sz_i|)")
    print("  E_soc from OUTCAR 'Spin-Orbit-Coupling matrix elements' block")
    print("  L, Sz from OUTCAR 'orbital moment' and 'magnetization(z)' blocks")
    print("=" * 70)

    xi_per_ion = {}
    tag_for_xi = '110' if have_110 else '001'

    if hasattr(analyzer, 'esoc') and tag_for_xi in analyzer.esoc:
        xi_per_ion = analyzer.extract_xi_from_esoc(tag_for_xi)
        # Fill any missing Co ions with SOC-matrix fallback
        for ion in all_co_ions:
            if ion not in xi_per_ion:
                xi_fb = analyzer.extract_xi_from_soc_matrix(tag_for_xi)
                xi_per_ion[ion] = xi_fb.get(ion, 0.065)
                print(f"  Ion {ion+1}: xi set from SOC matrix fallback = "
                      f"{xi_per_ion[ion]*1000:.2f} meV")
    else:
        print("  No E_soc data — using SOC matrix off-diagonal elements.")
        if hasattr(analyzer, 'soc_matrix') and tag_for_xi in analyzer.soc_matrix:
            xi_per_ion = analyzer.extract_xi_from_soc_matrix(tag_for_xi)
        else:
            print("  No SOC matrix either — using default xi = 65 meV.")
            xi_per_ion = {ion: 0.065 for ion in all_co_ions}

    # ===================================================================
    # STEP 5 — K^(1) from OUTCAR orbital and spin moments
    #
    # Formula: K^(1) = Sum_i  xi_i * [L_001_i*Sz_001_i - L_110_i*Sz_110_i]
    #
    # |L_110| = sqrt(Lx²+Ly²+Lz²) from orbital moment(x/y/z) of 110 OUTCAR
    # |L_001| = sqrt(Lx²+Ly²+Lz²) from orbital moment(x/y/z) of 001 OUTCAR
    # Sz_110  = tot column of magnetization(z) in 110 OUTCAR
    # Sz_001  = tot column of magnetization(z) in 001 OUTCAR
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 5 — K^(1) from OUTCAR orbital and spin moments")
    print("  Formula: K^(1) = Sum_i  xi_i * [L_001*Sz_001 - L_110*Sz_110]")
    print("  (larger orbital moment along easy axis => lower energy)")
    print("=" * 70)

    K1_total, site_info = analyzer.calculate_first_order_mae(
        ion_indices=all_co_ions,
        xi_values=xi_per_ion,
        energy_range=(-10, 0),
    )

    # ===================================================================
    # STEP 6 — K^(2) from second-order susceptibility (uses G0 DOSCAR)
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 6 — K^(2) from orbital-pair susceptibilities chi'_{mm'}")
    print("  Uses G0 (non-SOC DOSCAR) for correct perturbation theory.")
    print("=" * 70)

    K2_total = 0.0
    chi_prime_all = {}

    for rep_ion, label, n_sites in [(CO1_IONS[0], 'Co1', len(CO1_IONS)),
                                     (CO2_IONS[0], 'Co2', len(CO2_IONS))]:
        xi_rep = xi_per_ion.get(rep_ion, 0.065)
        print(f"\n  {label}: representative ion {rep_ion+1}"
              f"   xi = {xi_rep*1000:.2f} meV   x{n_sites} sites")
        sus, chi_prime = analyzer.calculate_all_susceptibilities(
            rep_ion, energy_range=(-7.5, 6.0))
        K2_one, _ = analyzer.calculate_second_order_mae(rep_ion, chi_prime, xi_rep)
        K2_type = n_sites * K2_one
        print(f"  K^(2)_{label} (x{n_sites}) = {K2_type:+.4f} meV")
        chi_prime_all[label] = chi_prime
        K2_total += K2_type
        analyzer.plot_chi_prime_heatmap(chi_prime, rep_ion)

    # ===================================================================
    # STEP 7 — Direct K = Delta E_soc (needs both OUTCARs)
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 7 — Direct MAE cross-check: K = E_soc(110) - E_soc(001)")
    print("=" * 70)

    if have_110 and have_001:
        mae_direct = analyzer.compute_mae_direct(tag_easy='110', tag_hard='001')
    else:
        missing = ([FOLDER_SOC_110] if not have_110 else []) + \
                  ([FOLDER_SOC_001] if not have_001 else [])
        print(f"  Skipped — OUTCAR missing from: {', '.join(missing)}")
        if have_110 and hasattr(analyzer, 'esoc') and '110' in analyzer.esoc:
            E110 = sum(analyzer.esoc['110'].values()) * 1000
            print(f"\n  E_soc_total(110) = {E110:+.2f} meV")
            print(f"  Inferred E_soc_total(001) = {E110 - K_DFT_MEV:+.2f} meV"
                  f"  [from K_DFT = {K_DFT_MEV:.1f} meV]")

    # ===================================================================
    # STEP 8 — Degenerate subspace diagnostics
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 8 — Degenerate (dxz, dyz) subspace diagnostics")
    print("=" * 70)
    for rep_ion, label in [(CO1_IONS[0], 'Co1'), (CO2_IONS[0], 'Co2')]:
        xi_rep = xi_per_ion.get(rep_ion, 0.065)
        print(f"\n  {label} — ion {rep_ion+1}:")
        analyzer.analyze_degenerate_subspace(rep_ion, xi_rep, energy_range=(-0.5, 0.5))

    # ===================================================================
    # STEP 9 — PDOS plots: SOC, non-SOC, and side-by-side comparison
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 9 — PDOS plots")
    print("=" * 70)

    for rep_ion, label in [(CO1_IONS[0], 'Co1'), (CO2_IONS[0], 'Co2')]:
        # SOC PDOS (LSORBIT=T)
        analyzer.plot_pdos(rep_ion, energy_range=(-8, 5),
                           use_G0=False, ion_label=label)

        # Non-SOC G0 PDOS (LSORBIT=F)
        if analyzer.pdos_G0_data:
            analyzer.plot_pdos(rep_ion, energy_range=(-8, 5),
                               use_G0=True, ion_label=label)

        # Side-by-side comparison (if both available)
        if analyzer.pdos_G0_data:
            analyzer.plot_pdos_comparison(rep_ion, energy_range=(-8, 5),
                                          ion_label=label)

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    K_total = K1_total + K2_total
    K1_Co1  = sum(site_info[i]['K1_meV'] for i in CO1_IONS if i in site_info)
    K1_Co2  = sum(site_info[i]['K1_meV'] for i in CO2_IONS if i in site_info)

    print("\n" + "#" * 70)
    print("#  FINAL SUMMARY — Sr3Co2O7  (2xCo1 + 2xCo2)")
    print("#  K = E_SOC(110) - E_SOC(001),  K < 0 => in-plane (110) easy axis")
    print("#" * 70)
    print()
    print(f"  {'Contribution':<32}  {'meV':>10}")
    print("  " + "-" * 45)
    print(f"  {'K^(1) Co1  (2 sites)':<32}  {K1_Co1:>+10.2f}")
    print(f"  {'K^(1) Co2  (2 sites)':<32}  {K1_Co2:>+10.2f}")
    print(f"  {'K^(1) total':<32}  {K1_total:>+10.2f}")
    print(f"  {'K^(2) total  (from G0 PDOS)':<32}  {K2_total:>+10.2f}")
    print("  " + "-" * 45)
    print(f"  {'K^(1) + K^(2)':<32}  {K_total:>+10.2f}")
    print(f"  {'DFT reference':<32}  {K_DFT_MEV:>+10.2f}")
    print(f"  {'Difference (K^(2) residual)':<32}  {K_DFT_MEV - K_total:>+10.2f}")
    print()
    print(f"  Easy axis: {'in-plane (110)' if K_total < 0 else 'out-of-plane (001)'}")
    print()
    print("  Per-ion details:")
    print(f"  {'Ion':>4} {'Type':<5} {'xi(meV)':>8} "
          f"{'|L_110|':>8} {'Sz_110':>8} "
          f"{'|L_001|':>8} {'Sz_001':>8} "
          f"{'K^(1)(meV)':>11}")
    print("  " + "-" * 72)
    for ion in all_co_ions:
        if ion in site_info:
            s     = site_info[ion]
            itype = 'Co1' if ion in CO1_IONS else 'Co2'
            xi_meV = xi_per_ion.get(ion, 0.065) * 1000
            print(f"  {ion+1:>4} {itype:<5} {xi_meV:>8.2f} "
                  f"{s['L_inplane']:>8.4f} {s['Sz_110']:>8.4f} "
                  f"{s['L_outofplane']:>8.4f} {s['Sz_001']:>8.4f} "
                  f"{s['K1_meV']:>+11.4f}")
    print()


if __name__ == "__main__":
    main()
