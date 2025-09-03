# QE Slab Builder

A graphical user interface (GUI) tool for constructing slab models, preparing Quantum ESPRESSO input files, and analyzing output results.
It is designed for surface energy calculations and workflow management in **Quantum ESPRESSO (pw\.x)**.

## Features

* **Structure Handling**

  * Load bulk structures from CIF files.
  * Build slab models using Miller indices, number of layers, and vacuum spacing.
  * Save slab structures as CIF files.
  * Supercell expansion (visualization only).

* **Quantum ESPRESSO Input Generation**

  * Generate separate input files for **bulk** and **slab** systems.
  * Customizable QE parameters:

    * `calculation`, `ecutwfc`, `ecutrho`, `k-points`, `conv_thr`, `occupations`, `smearing`, `degauss`, `nspin`, `nbnd`.
  * Handles pseudopotentials:

    * Separate **search folder** (for locating pseudos) and **input folder** (`pseudo_dir` in QE).
    * Optionally copy required pseudopotentials into the chosen `pseudo_dir`.
  * Spin-polarized setup (`nspin=2`) with optional `starting_magnetization`.

* **Output Parsing & Analysis**

  * Load QE `.out` files for bulk and slab.
  * Extract total energies (Ry → eV).
  * Automatic calculation of **surface energy**:

    $$
    E_\text{surf} = \frac{E_\text{slab} - N_\text{slab} E_\text{bulk}/N_\text{bulk}}{2A}
    $$

    (in eV/Å²).

* **Visualization**

  * 3D viewer (py3Dmol + Qt WebEngine) with **ball-and-stick** representation.
  * Toggle between bulk and slab views.
  * Supercell visualization.
  * Unit cell rendering.

* **Session Management**

  * Save and load session settings in JSON format (structures, QE parameters, results).

* **Terminal Log**

  * Integrated log panel for status and error messages.

## Requirements

* Python 3.8+
* Dependencies:

  ```bash
  pip install PySide6 ase py3Dmol numpy
  ```

## Usage

1. Run the application:

   ```bash
   python qe_slab_builder.py
   ```
2. **Load a CIF file** under the *Parameters → Structure* tab.
3. Build a **slab model** (set Miller index, layers, vacuum).
4. Switch to *QE Settings* tab to configure QE input parameters.
5. Generate **Bulk** or **Slab** input files (`.in`).
6. After QE calculation, load `.out` files under the *Results* tab to compute **surface energy**.
7. Save or reload session data via JSON.

## Notes

* **Supercell settings** are for visualization only (not applied to `.in` files).
* For slabs, default **K\_POINTS = 1 1 1**, unless “Use custom K\_POINTS” is checked.
* When `nspin=2`, three starting magnetization fields are enabled; values of `0.0` are omitted in `.in`.
* Pseudopotentials:

  * If "copy pseudos" is enabled, files are copied into the selected `pseudo_dir`.
  * Otherwise, `.in` will reference filenames directly; ensure pseudos exist in the specified path.

## License

Apache 2.0 License (check repository for details).


