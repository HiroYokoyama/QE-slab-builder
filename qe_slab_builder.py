# qe_slab_builder.py
# github.com/HiroYokoyama/QE-slab-builder
# - Pseudopotential: separate UI fields for Search folder and Input folder (pseudo_dir)
# - When saving .in, optionally copy required pseudo files into the chosen input pseudo_dir (relative to .in)
# - Both bulk and slab INs: do NOT apply supercell
# - nspin == 2: enable 3 starting_magnetization input fields; default 0.0 -> not written to .in
# - Slab KPOINTS default 1 1 1 unless override_kpoints checked
# - starting_magnetization fields are stacked vertically (one per row)
import sys
import os
import re
import io
import json
import shutil
import filecmp
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QTabWidget, QLabel, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QComboBox,
    QTextEdit, QSplitter, QFormLayout, QSizePolicy, QRadioButton, QButtonGroup, QCheckBox
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import Qt
from ase.io import read, write
from ase.build import surface
from ase import Atoms
import py3Dmol

RY_TO_EV = 13.605693009

def atoms_to_json(atoms):
    if atoms is None:
        return None
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist()
    }

def atoms_from_json(data):
    if data is None:
        return None
    return Atoms(
        symbols=data["symbols"],
        positions=np.array(data["positions"]),
        cell=np.array(data["cell"]),
        pbc=tuple(data["pbc"])
    )

class SlabApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QE Slab Builder & I/O Tool")
        self.bulk = None
        self.slab = None
        self.settings = {}

        # viewer-only supercell controls
        self.super_x = QSpinBox(); self.super_x.setValue(1)
        self.super_y = QSpinBox(); self.super_y.setValue(1)
        self.super_z = QSpinBox(); self.super_z.setValue(1)
        self.super_x.valueChanged.connect(self.update_viewer)
        self.super_y.valueChanged.connect(self.update_viewer)
        self.super_z.valueChanged.connect(self.update_viewer)

        tabs = QTabWidget()
        tabs.addTab(self.parameters_tab(), "Parameters")
        tabs.addTab(self.results_tab(), "Results")

        self.viewer_widget = self.viewer_tab()

        splitter = QSplitter()
        splitter.addWidget(tabs)
        splitter.addWidget(self.viewer_widget)
        splitter.setSizes([420, 760])

        # Terminal
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("background-color: black; color: lime; font-family: Consolas;")
        self.terminal.setFixedHeight(110)

        # Save/Load Data buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_save_json = QPushButton("💾 Save Data")
        btn_save_json.clicked.connect(self.save_settings)
        btn_load_json = QPushButton("📂 Load Data")
        btn_load_json.clicked.connect(self.load_settings)
        btn_layout.addWidget(btn_save_json)
        btn_layout.addWidget(btn_load_json)
        btn_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.terminal)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def parameters_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignTop)

        subtabs = QTabWidget()

        # Structure tab
        struct_widget = QWidget()
        struct_layout = QVBoxLayout()
        struct_layout.setSpacing(6)
        struct_layout.setContentsMargins(6, 6, 6, 6)
        struct_layout.setAlignment(Qt.AlignTop)

        self.file_label = QLabel("CIF: None")
        btn_load = QPushButton("📂 Load CIF")
        btn_load.clicked.connect(self.load_cif)

        hbox = QHBoxLayout()
        hbox.setSpacing(4)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.h = QSpinBox(); self.h.setValue(1)
        self.k = QSpinBox(); self.k.setValue(1)
        self.l = QSpinBox(); self.l.setValue(1)
        hbox.addWidget(QLabel("Miller index:"))
        hbox.addWidget(self.h); hbox.addWidget(self.k); hbox.addWidget(self.l)

        self.layers = QSpinBox(); self.layers.setValue(6)
        self.vacuum = QDoubleSpinBox(); self.vacuum.setValue(10.0)

        btn_update = QPushButton("🔄 Update (Build Slab)")
        btn_update.clicked.connect(self.build_slab)

        btn_save_cif = QPushButton("💾 Save Slab as CIF")
        btn_save_cif.clicked.connect(self.save_cif)

        struct_layout.addWidget(self.file_label)
        struct_layout.addWidget(btn_load)
        struct_layout.addLayout(hbox)
        struct_layout.addWidget(QLabel("Layers"))
        struct_layout.addWidget(self.layers)
        struct_layout.addWidget(QLabel("Vacuum (Å)"))
        struct_layout.addWidget(self.vacuum)
        struct_layout.addWidget(btn_update)
        struct_layout.addWidget(btn_save_cif)
        struct_widget.setLayout(struct_layout)

        # QE tab
        qe_widget = QWidget()
        qe_layout = QFormLayout()
        qe_layout.setVerticalSpacing(6)
        qe_layout.setHorizontalSpacing(8)

        self.calculation = QComboBox()
        self.calculation.addItems(["scf", "relax", "nscf"])
        self.ecutwfc = QSpinBox(); self.ecutwfc.setValue(40)
        # changed: ecutrho range and default
        self.ecutrho = QSpinBox(); self.ecutrho.setRange(0, 9999); self.ecutrho.setValue(400)
        self.kx = QSpinBox(); self.kx.setValue(4)
        self.ky = QSpinBox(); self.ky.setValue(4)
        self.kz = QSpinBox(); self.kz.setValue(1)
        self.prefix = QLineEdit("qe_calc")

        # Pseudopotential: separate fields for search (where to find pseudos) and input (pseudo_dir)
        self.pp_search_folder = QLineEdit()
        btn_browse_pp_search = QPushButton("Browse")
        btn_browse_pp_search.setToolTip("Browse folder to search for existing pseudopotentials (hint only)")
        btn_browse_pp_search.clicked.connect(self.browse_pp_search)
        pp_search_box = QHBoxLayout()
        pp_search_box.setSpacing(4)
        pp_search_box.setContentsMargins(0,0,0,0)
        pp_search_box.addWidget(self.pp_search_folder)
        pp_search_box.addWidget(btn_browse_pp_search)
        pp_search_widget = QWidget(); pp_search_widget.setLayout(pp_search_box)

        # Input pseudo_dir (what to write in &CONTROL as pseudo_dir, e.g. './pseudo' or 'pseudo')
        self.pp_input_folder = QLineEdit("./pseudo")
        btn_browse_pp_input = QPushButton("Browse")
        btn_browse_pp_input.setToolTip("Optional: choose a default input pseudo folder (used as pseudo_dir in .in). It will be created next to the .in file when saving if 'copy pseudos' is enabled.")
        btn_browse_pp_input.clicked.connect(self.browse_pp_input)
        pp_input_box = QHBoxLayout()
        pp_input_box.setSpacing(4)
        pp_input_box.setContentsMargins(0,0,0,0)
        pp_input_box.addWidget(self.pp_input_folder)
        pp_input_box.addWidget(btn_browse_pp_input)
        pp_input_widget = QWidget(); pp_input_widget.setLayout(pp_input_box)

        # NEW parameter: whether to copy pseudos into pseudo_dir when saving .in
        self.copy_pseudos_checkbox = QCheckBox("Copy pseudopotentials into pseudo_dir when saving (.in)")
        self.copy_pseudos_checkbox.setChecked(False)  # default: do NOT copy

        # Input/other params
        self.outdir = QLineEdit("./out")
        self.conv_thr = QDoubleSpinBox()
        self.conv_thr.setDecimals(12)
        self.conv_thr.setRange(1e-12, 1.0)
        self.conv_thr.setValue(1e-8)

        self.occupations = QComboBox()
        self.occupations.addItems(["fixed", "smearing", "tetrahedra"])
        self.smearing = QComboBox()
        self.smearing.addItems(["gaussian", "methfessel-paxton", "marzari-vanderbilt"])
        self.degauss = QDoubleSpinBox()
        self.degauss.setDecimals(6)
        self.degauss.setRange(0.0, 10.0)
        self.degauss.setValue(0.01)

        self.nspin = QSpinBox(); self.nspin.setRange(1, 2); self.nspin.setValue(1)
        self.nspin.valueChanged.connect(self._on_nspin_changed)
        self.nbnd = QSpinBox(); self.nbnd.setRange(0, 10000); self.nbnd.setValue(0)

        # NEW: starting magnetizations (3 fields) disabled by default; enabled only when nspin==2
        self.start_mag_1 = QDoubleSpinBox(); self.start_mag_1.setDecimals(6); self.start_mag_1.setRange(-10.0, 10.0); self.start_mag_1.setValue(0.0); self.start_mag_1.setEnabled(False)
        self.start_mag_2 = QDoubleSpinBox(); self.start_mag_2.setDecimals(6); self.start_mag_2.setRange(-10.0, 10.0); self.start_mag_2.setValue(0.0); self.start_mag_2.setEnabled(False)
        self.start_mag_3 = QDoubleSpinBox(); self.start_mag_3.setDecimals(6); self.start_mag_3.setRange(-10.0, 10.0); self.start_mag_3.setValue(0.0); self.start_mag_3.setEnabled(False)

        # NEW: checkbox to allow custom k-points for slab
        self.override_kpoints = QCheckBox("Use custom K_POINTS for slab (if unchecked, slab uses 1 1 1)")
        self.override_kpoints.setChecked(True)

        # build form
        qe_layout.addRow("calculation", self.calculation)
        qe_layout.addRow("ecutwfc (Ry)", self.ecutwfc)
        qe_layout.addRow("ecutrho (Ry)", self.ecutrho)
        kpoints_box = QHBoxLayout()
        kpoints_box.setSpacing(4)
        kpoints_box.setContentsMargins(0, 0, 0, 0)
        kpoints_box.addWidget(self.kx); kpoints_box.addWidget(self.ky); kpoints_box.addWidget(self.kz)
        kpoints_widget = QWidget(); kpoints_widget.setLayout(kpoints_box)
        qe_layout.addRow("k-points (kx,ky,kz)", kpoints_widget)
        qe_layout.addRow(self.override_kpoints)
        qe_layout.addRow("prefix", self.prefix)
        qe_layout.addRow("Pseudo Search Folder (hint)", pp_search_widget)
        qe_layout.addRow("Pseudo Input Folder (pseudo_dir in .in)", pp_input_widget)
        qe_layout.addRow(self.copy_pseudos_checkbox)
        qe_layout.addRow("outdir", self.outdir)
        qe_layout.addRow("conv_thr", self.conv_thr)
        qe_layout.addRow("occupations", self.occupations)
        qe_layout.addRow("smearing type", self.smearing)
        qe_layout.addRow("degauss (Ry)", self.degauss)
        qe_layout.addRow("nspin (1 or 2)", self.nspin)
        qe_layout.addRow("nbnd (0 = auto)", self.nbnd)

        # starting magnetizations arranged vertically (one per row)
        sm_vbox = QVBoxLayout()
        sm_vbox.setSpacing(4)
        # row 1
        sm_row1 = QHBoxLayout()
        sm_row1.addWidget(QLabel("start_mag 1"))
        sm_row1.addWidget(self.start_mag_1)
        sm_vbox.addLayout(sm_row1)
        # row 2
        sm_row2 = QHBoxLayout()
        sm_row2.addWidget(QLabel("start_mag 2"))
        sm_row2.addWidget(self.start_mag_2)
        sm_vbox.addLayout(sm_row2)
        # row 3
        sm_row3 = QHBoxLayout()
        sm_row3.addWidget(QLabel("start_mag 3"))
        sm_row3.addWidget(self.start_mag_3)
        sm_vbox.addLayout(sm_row3)
        sm_widget = QWidget(); sm_widget.setLayout(sm_vbox)
        qe_layout.addRow("Starting magnetizations (per type, only if nspin=2)", sm_widget)

        # Buttons
        btn_bulk_input = QPushButton("📝 Generate Bulk Input")
        btn_bulk_input.setToolTip("Generate QE input for Bulk (no supercell applied to output).")
        btn_bulk_input.clicked.connect(lambda: self.generate_qe_input(mode="bulk"))
        btn_slab_input = QPushButton("📝 Generate Slab Input")
        btn_slab_input.setToolTip("Generate QE input for Slab (no supercell applied to output).")
        btn_slab_input.clicked.connect(lambda: self.generate_qe_input(mode="slab"))

        qe_container = QWidget()
        qe_container_layout = QVBoxLayout()
        qe_container_layout.setSpacing(6)
        qe_container_layout.setContentsMargins(6, 6, 6, 6)
        qe_container_layout.setAlignment(Qt.AlignTop)
        qe_container_layout.addLayout(qe_layout)
        qe_container_layout.addWidget(btn_bulk_input)
        qe_container_layout.addWidget(btn_slab_input)
        qe_container.setLayout(qe_container_layout)

        qe_widget_layout = QVBoxLayout()
        qe_widget_layout.setSpacing(0)
        qe_widget_layout.setContentsMargins(0, 0, 0, 0)
        qe_widget_layout.addWidget(qe_container)
        qe_widget.setLayout(qe_widget_layout)

        subtabs.addTab(struct_widget, "Structure")
        subtabs.addTab(qe_widget, "QE Settings")
        layout.addWidget(subtabs)
        widget.setLayout(layout)
        return widget

    def browse_pp_search(self):
        folder = QFileDialog.getExistingDirectory(self, "Select pseudopotential SEARCH folder", "")
        if folder:
            self.pp_search_folder.setText(folder)

    def browse_pp_input(self):
        # For input pseudo folder we allow free text (relative path), but user can also pick a folder
        folder = QFileDialog.getExistingDirectory(self, "Select pseudopotential INPUT folder (used as pseudo_dir in .in)", "")
        if folder:
            self.pp_input_folder.setText(folder)

    def results_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignTop)

        btn_load_bulk = QPushButton("📂 Load Bulk QE Output")
        btn_load_bulk.clicked.connect(lambda: self.load_qe_output("bulk"))
        self.bulk_energy_label = QLabel("Bulk energy: None")

        btn_load_slab = QPushButton("📂 Load Slab QE Output")
        btn_load_slab.clicked.connect(lambda: self.load_qe_output("slab"))
        self.slab_energy_label = QLabel("Slab energy: None")

        self.surface_energy_label = QLabel("Surface energy: Not calculated")

        layout.addWidget(btn_load_bulk)
        layout.addWidget(self.bulk_energy_label)
        layout.addWidget(btn_load_slab)
        layout.addWidget(self.slab_energy_label)
        layout.addWidget(self.surface_energy_label)

        widget.setLayout(layout)
        return widget

    def viewer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignTop)

        radio_layout = QHBoxLayout()
        radio_layout.setSpacing(8)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        self.rb_bulk = QRadioButton("Bulk")
        self.rb_slab = QRadioButton("Slab")
        self.rb_bulk.setChecked(True)
        self.rb_group = QButtonGroup()
        self.rb_group.addButton(self.rb_bulk)
        self.rb_group.addButton(self.rb_slab)
        self.rb_bulk.toggled.connect(self.update_viewer)
        self.rb_slab.toggled.connect(self.update_viewer)

        radio_layout.addWidget(self.rb_bulk)
        radio_layout.addWidget(self.rb_slab)
        radio_layout.addStretch(1)

        sc_layout = QHBoxLayout()
        sc_layout.setSpacing(6)
        sc_layout.setContentsMargins(0, 0, 0, 0)
        sc_layout.addWidget(QLabel("Supercell:"))
        sc_layout.addWidget(self.super_x)
        sc_layout.addWidget(self.super_y)
        sc_layout.addWidget(self.super_z)
        sc_layout.addStretch(1)
        sc_widget = QWidget(); sc_widget.setLayout(sc_layout)

        self.viewer = QWebEngineView()
        self.viewer.setMinimumHeight(380)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addLayout(radio_layout)
        layout.addWidget(sc_widget)
        layout.addWidget(self.viewer)
        widget.setLayout(layout)
        return widget

    def _on_nspin_changed(self, val):
        enabled = (val == 2)
        self.start_mag_1.setEnabled(enabled)
        self.start_mag_2.setEnabled(enabled)
        self.start_mag_3.setEnabled(enabled)
        if not enabled:
            # reset to default (0.0) when disabled
            self.start_mag_1.setValue(0.0)
            self.start_mag_2.setValue(0.0)
            self.start_mag_3.setValue(0.0)

    def log_message(self, msg):
        self.terminal.append(msg)
        print(msg)

    def load_cif(self):
        try:
            file, _ = QFileDialog.getOpenFileName(self, "Open CIF", "", "CIF Files (*.cif)")
            if file:
                self.bulk = read(file)
                self.file_label.setText(f"CIF: {file}")
                self.log_message(f"Loaded CIF: {file}")
                self.update_viewer()
        except Exception as e:
            self.log_message(f"[Error] load_cif: {e}")

    def build_slab(self):
        try:
            if self.bulk is None:
                self.log_message("Error: No bulk structure loaded.")
                return
            self.slab = surface(
                self.bulk,
                (self.h.value(), self.k.value(), self.l.value()),
                self.layers.value(),
                vacuum=self.vacuum.value()
            )
            self.log_message("Slab built/updated successfully.")
            self.update_viewer()
        except Exception as e:
            self.log_message(f"[Error] build_slab: {e}")

    def save_cif(self):
        try:
            if self.slab is None:
                self.log_message("Error: No slab to save.")
                return
            sx, sy, sz = self.super_x.value(), self.super_y.value(), self.super_z.value()
            slab_to_save = self.slab * (sx, sy, sz) if (sx > 1 or sy > 1 or sz > 1) else self.slab
            file, _ = QFileDialog.getSaveFileName(self, "Save Slab CIF", "", "CIF Files (*.cif)")
            if file:
                write(file, slab_to_save)
                self.log_message(f"Slab saved to: {file} (supercell: {sx},{sy},{sz})")
        except Exception as e:
            self.log_message(f"[Error] save_cif: {e}")

    def save_settings(self):
        try:
            file, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "JSON Files (*.json)")
            if not file:
                return
            self.settings = {
                "structures": {
                    "bulk": atoms_to_json(self.bulk),
                    "slab": atoms_to_json(self.slab)
                },
                "miller_index": [self.h.value(), self.k.value(), self.l.value()],
                "layers": self.layers.value(),
                "vacuum": self.vacuum.value(),
                "supercell": [self.super_x.value(), self.super_y.value(), self.super_z.value()],
                "qe_input": {
                    "calculation": self.calculation.currentText(),
                    "ecutwfc": self.ecutwfc.value(),
                    "ecutrho": self.ecutrho.value(),
                    "kpoints": [self.kx.value(), self.ky.value(), self.kz.value()],
                    "override_kpoints": self.override_kpoints.isChecked(),
                    "prefix": self.prefix.text(),
                    "pp_search_folder": self.pp_search_folder.text(),
                    "pp_input_folder": self.pp_input_folder.text(),
                    "copy_pseudos": self.copy_pseudos_checkbox.isChecked(),
                    "outdir": self.outdir.text(),
                    "conv_thr": self.conv_thr.value(),
                    "occupations": self.occupations.currentText(),
                    "smearing": self.smearing.currentText(),
                    "degauss": self.degauss.value(),
                    "nspin": self.nspin.value(),
                    "nbnd": self.nbnd.value(),
                    "starting_mags": [self.start_mag_1.value(), self.start_mag_2.value(), self.start_mag_3.value()]
                },
                "results": self.settings.get("results", {})
            }
            with open(file, "w") as f:
                json.dump(self.settings, f, indent=2)
            self.log_message(f"Data saved to {file}")
        except Exception as e:
            self.log_message(f"[Error] save_settings: {e}")

    def load_settings(self):
        try:
            file, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "JSON Files (*.json)")
            if not file:
                return
            with open(file) as f:
                self.settings = json.load(f)
            s = self.settings

            self.bulk = atoms_from_json(s["structures"]["bulk"])
            self.slab = atoms_from_json(s["structures"]["slab"])

            self.h.setValue(s["miller_index"][0])
            self.k.setValue(s["miller_index"][1])
            self.l.setValue(s["miller_index"][2])
            self.layers.setValue(s["layers"])
            self.vacuum.setValue(s["vacuum"])
            sc = s.get("supercell", [1, 1, 1])
            self.super_x.setValue(sc[0]); self.super_y.setValue(sc[1]); self.super_z.setValue(sc[2])

            qe = s["qe_input"]
            self.calculation.setCurrentText(qe.get("calculation", "scf"))
            self.ecutwfc.setValue(qe.get("ecutwfc", 40))
            self.ecutrho.setValue(qe.get("ecutrho", 400))
            kpts = qe.get("kpoints", [4,4,1])
            self.kx.setValue(kpts[0]); self.ky.setValue(kpts[1]); self.kz.setValue(kpts[2])
            self.override_kpoints.setChecked(qe.get("override_kpoints", False))
            self.prefix.setText(qe.get("prefix", "qe_calc"))
            self.pp_search_folder.setText(qe.get("pp_search_folder", ""))
            self.pp_input_folder.setText(qe.get("pp_input_folder", "./pseudo"))
            self.copy_pseudos_checkbox.setChecked(qe.get("copy_pseudos", False))
            self.outdir.setText(qe.get("outdir", "./out"))
            self.conv_thr.setValue(qe.get("conv_thr", 1e-8))
            self.occupations.setCurrentText(qe.get("occupations", "fixed"))
            self.smearing.setCurrentText(qe.get("smearing", "gaussian"))
            self.degauss.setValue(qe.get("degauss", 0.01))
            self.nspin.setValue(qe.get("nspin", 1))
            self.nbnd.setValue(qe.get("nbnd", 0))
            mags = qe.get("starting_mags", [0.0,0.0,0.0])
            self.start_mag_1.setValue(mags[0] if len(mags)>0 else 0.0)
            self.start_mag_2.setValue(mags[1] if len(mags)>1 else 0.0)
            self.start_mag_3.setValue(mags[2] if len(mags)>2 else 0.0)
            # ensure enabled/disabled matches nspin
            self._on_nspin_changed(self.nspin.value())

            self.log_message(f"Data loaded from {file}")
            self.update_viewer()
        except Exception as e:
            self.log_message(f"[Error] load_settings: {e}")


    def _parse_qe_out_file(self, filepath):
        """
        QE .out から total energy (Ry) と number of atoms を抽出するヘルパー。
        戻り値: (energy_Ry_or_None, natoms_or_None)
        """
        energy = None
        natoms = None
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            self.log_message(f"[Error] could not open {filepath}: {e}")
            return None, None
    
        # 1) number of atoms/cell を探す（大文字小文字混在に対応）
        for line in lines:
            m_n = re.search(r'number of atoms\s*/\s*cell\s*=\s*(\d+)', line, flags=re.I)
            if m_n:
                try:
                    natoms = int(m_n.group(1))
                    break
                except:
                    natoms = None

        # 2) total energy を探す（'!' を含む行）
        for line in lines:
            if '!' in line and 'total energy' in line.lower():
                m_e = re.search(r'=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                if m_e:
                    try:
                        energy = float(m_e.group(1))
                    except:
                        energy = None
                break

        # フォールバック：ATOMIC_POSITIONS ブロックから原子数をカウント
        if natoms is None:
            for i, line in enumerate(lines):
                if 'ATOMIC_POSITIONS' in line.upper():
                    count = 0
                    for j in range(i+1, len(lines)):
                        l = lines[j].strip()
                        if l == '' or re.match(r'^[A-Z _0-9()-]+:$', l):  # 空行か次セクションの可能性
                            break
                        # 行が座標行らしければカウント
                        toks = l.split()
                        if len(toks) >= 4:
                            count += 1
                    if count > 0:
                        natoms = count
                        break

        # 最終フォールバック：出力内の "atomic positions" 表示などが無かったら None のまま
        # 追加のフォールバック：もし energy が未検出なら最後に数字が出てくる行の数字を使う（慎重）
        if energy is None:
            for line in reversed(lines[-200:]):  # 最後の 200 行だけ拾う
                found = re.findall(r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                if found:
                    try:
                        energy = float(found[-1])
                        break
                    except:
                        continue

        return energy, natoms


    def load_qe_output(self, mode):
        """
        GUI 用の改良版 load_qe_output:
        - mode: "bulk" or "slab"
        - .out から energy (Ry) と natoms を抽出し、エネルギーは eV に変換して保存する
        """
        try:
            file, _ = QFileDialog.getOpenFileName(self, "Open QE Output", "", "QE Output (*.out *.out.gz);;All files (*)")
            if not file:
                return

            e_Ry, natoms = self._parse_qe_out_file(file)

            if e_Ry is None:
                self.log_message(f"Could not parse total energy from: {file}")
                return

            e_eV = e_Ry * RY_TO_EV

            results = self.settings.setdefault("results", {})
            if mode == "bulk":
                results["bulk_energy_eV"] = e_eV
                results["bulk_energy_Ry"] = e_Ry
                results["bulk_natoms"] = natoms
                self.bulk_energy_label.setText(f"Bulk energy = {e_eV:.6f} eV ( {e_Ry:.6f} Ry )")
                self.log_message(f"Bulk energy loaded: {e_eV:.6f} eV ( {e_Ry:.6f} Ry ), natoms={natoms}")
            elif mode == "slab":
                results["slab_energy_eV"] = e_eV
                results["slab_energy_Ry"] = e_Ry
                results["slab_natoms"] = natoms
                self.slab_energy_label.setText(f"Slab energy = {e_eV:.6f} eV ( {e_Ry:.6f} Ry )")
                self.log_message(f"Slab energy loaded: {e_eV:.6f} eV ( {e_Ry:.6f} Ry ), natoms={natoms}")
            else:
                self.log_message(f"[Error] unknown mode for load_qe_output: {mode}")
                return

            # 両方揃えば表面エネルギーを計算
            if "bulk_energy_eV" in results and "slab_energy_eV" in results:
                self.calculate_surface_energy()
        except Exception as e:
            self.log_message(f"[Error] load_qe_output: {e}")



    def calculate_surface_energy(self):
        """
        結果 dict にある bulk/slab energies (eV) と natoms を使って表面エネルギーを eV/Å^2 で計算する。
        E_surf = (E_slab - N_slab * E_bulk_per_atom) / (2 * A)
        A は slab のセルから xy 面積を計算（self.slab が ase.Atoms の場合を想定）
        """
        try:
            results = self.settings.get("results", {})
            if "bulk_energy_eV" not in results or "slab_energy_eV" not in results:
                self.log_message("[Error] both bulk and slab energies must be loaded before calculation.")
                return

            e_bulk_total_eV = results["bulk_energy_eV"]
            e_slab_total_eV = results["slab_energy_eV"]

            N_bulk = results.get("bulk_natoms")
            if N_bulk is None or N_bulk == 0:
                # もし GUI に読み込まれている self.bulk があればそれを使う
                if getattr(self, "bulk", None) is not None:
                    N_bulk = len(self.bulk)
                    self.log_message(f"bulk_natoms missing in output; using loaded bulk structure natoms = {N_bulk}")
                else:
                    self.log_message("[Error] bulk atom count unknown; cannot compute per-atom energy.")
                    return

            N_slab = results.get("slab_natoms")
            if N_slab is None or N_slab == 0:
                if getattr(self, "slab", None) is not None:
                    N_slab = len(self.slab)
                    self.log_message(f"slab_natoms missing in output; using loaded slab structure natoms = {N_slab}")
                else:
                    self.log_message("[Error] slab atom count unknown; cannot compute surface energy.")
                    return

            # bulk per-atom (eV)
            e_bulk_per_atom_eV = e_bulk_total_eV / float(N_bulk)

            # 面積 A: slab のセルの x-y 面積を使う（self.slab が ase.Atoms で get_cell() を持つ想定）
            if getattr(self, "slab", None) is not None:
                try:
                    cell = self.slab.get_cell()
                    # cell vectors might be array-like 3x3
                    a = np.array(cell[0])
                    b = np.array(cell[1])
                    A = np.linalg.norm(np.cross(a, b))
                except Exception as e:
                    self.log_message(f"[Warning] could not get cell from slab object: {e}. Using fallback area=1.0 Å^2")
                    A = 1.0
            else:
                self.log_message("[Warning] slab object not present; using fallback area=1.0 Å^2")
                A = 1.0

            # surface energy (two surfaces)
            E_surf_eV_per_A2 = (e_slab_total_eV - float(N_slab) * e_bulk_per_atom_eV) / (2.0 * A)

            # 保存・表示
            results["surface_energy_eV_per_A2"] = E_surf_eV_per_A2
            self.surface_energy_label.setText(f"Surface energy = {E_surf_eV_per_A2:.6f} eV/Å²")
            self.log_message(f"Calculated surface energy: {E_surf_eV_per_A2:.6f} eV/Å² (A={A:.6f} Å², N_bulk={N_bulk}, N_slab={N_slab})")
        except Exception as e:
            self.log_message(f"[Error] calculate_surface_energy: {e}")


    def _viewer_mode_text(self):
        return "Slab" if self.rb_slab.isChecked() else "Bulk"

    def update_viewer(self):
        try:
            atoms = None
            mode = "Bulk"
            if hasattr(self, "rb_slab") and self.rb_slab.isChecked():
                mode = "Slab"
            # Viewer: apply supercell only to visualization (not to IN generation)
            if mode == "Bulk" and self.bulk:
                atoms = self.bulk * (self.super_x.value(), self.super_y.value(), self.super_z.value())
            elif mode == "Slab" and self.slab:
                atoms = self.slab * (self.super_x.value(), self.super_y.value(), self.super_z.value())

            if atoms:
                buf = io.StringIO()
                write(buf, atoms, format="xyz")
                xyz_str = buf.getvalue()


                view = py3Dmol.view(width=700, height=600)
                view.addModel(xyz_str, "xyz")
                view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
                try:
                    view.addUnitCell({"color": "green", "box": True})
                except Exception:
                    pass
                view.zoomTo()
                html = view._make_html()
                self.viewer.setHtml(html)
        except Exception as e:
            self.log_message(f"[Error] update_viewer: {e}")

    # Helper: find pseudo files (returns full path for each element)
    def _find_pseudos_for_elements(self, elements, pp_search_folder_hint):
        """
        elements: list of element symbols (unique)
        pp_search_folder_hint: optional folder to search first (may be '')
        returns: dict element -> fullpath_to_pseudo
        """
        element_to_full = {}
        missing = []

        candidates = []
        if pp_search_folder_hint and os.path.isdir(pp_search_folder_hint):
            candidates = [os.path.join(pp_search_folder_hint, f) for f in os.listdir(pp_search_folder_hint)
                          if os.path.isfile(os.path.join(pp_search_folder_hint, f))]

        def elem_in_fname(elem, fname):
            base = os.path.basename(fname)
            pattern = r'(?<![A-Za-z])' + re.escape(elem) + r'(?![A-Za-z])'
            return re.search(pattern, base, flags=re.IGNORECASE) is not None

        ext_pref = ['.upf', '.UPF', '.psp', '.PSP', '.psf', '.PSF', '.pseudo', '.PSEUDO', '.dat', '.DAT']

        for elem in elements:
            found = [f for f in candidates if elem_in_fname(elem, f)]
            if found:
                def score(fpath):
                    _, ext = os.path.splitext(fpath)
                    try:
                        p = ext_pref.index(ext)
                    except ValueError:
                        p = len(ext_pref)
                    return (p, len(os.path.basename(fpath)))
                found.sort(key=score)
                element_to_full[elem] = found[0]
            else:
                missing.append(elem)

        # prompt user for missing elements (allow selection from anywhere)
        if missing:
            for elem in missing:
                self.log_message(f"Pseudo for element '{elem}' not found in search folder.")
                file, _ = QFileDialog.getOpenFileName(self, f"Select pseudopotential for {elem}", pp_search_folder_hint or "", "Pseudopotential files (*.upf *.UPF *.psp *.PSP *.psf *.PSF *.pseudo *.PSEUDO *.dat *.DAT);;All files (*)")
                if not file:
                    raise FileNotFoundError(f"No pseudopotential provided for element {elem}")
                if not os.path.isfile(file):
                    raise FileNotFoundError(f"Selected pseudopotential does not exist: {file}")
                element_to_full[elem] = file

        return element_to_full

    def _copy_pseudos_to_target(self, elem_to_fullpath, dest_dir):
        """
        Copy fullpath pseudos to dest_dir, handling name conflicts.
        Returns dict element -> basename_in_dest_dir
        """
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)

        elem_to_basename = {}
        for elem, full in elem_to_fullpath.items():
            basename = os.path.basename(full)
            dest = os.path.join(dest_dir, basename)
            if os.path.exists(dest):
                try:
                    if filecmp.cmp(full, dest, shallow=False):
                        self.log_message(f"Reusing existing {basename} in {dest_dir}")
                        elem_to_basename[elem] = basename
                        continue
                    else:
                        base, ext = os.path.splitext(basename)
                        idx = 1
                        while True:
                            newname = f"{base}_{idx}{ext}"
                            newdest = os.path.join(dest_dir, newname)
                            if not os.path.exists(newdest):
                                shutil.copy2(full, newdest)
                                elem_to_basename[elem] = newname
                                self.log_message(f"Copied {full} -> {newdest}")
                                break
                            idx += 1
                except Exception:
                    base, ext = os.path.splitext(basename)
                    idx = 1
                    while True:
                        newname = f"{base}_{idx}{ext}"
                        newdest = os.path.join(dest_dir, newname)
                        if not os.path.exists(newdest):
                            shutil.copy2(full, newdest)
                            elem_to_basename[elem] = newname
                            self.log_message(f"Copied {full} -> {newdest}")
                            break
                        idx += 1
            else:
                shutil.copy2(full, dest)
                elem_to_basename[elem] = basename
                self.log_message(f"Copied {full} -> {dest}")
        return elem_to_basename

    def generate_qe_input(self, mode="bulk"):
        """
        mode: "bulk" or "slab"
        - For BOTH modes: do NOT apply supercell when writing .in (write original cell)
        - For slab: K_POINTS = 1 1 1 unless override_kpoints is checked
        - Use pp_search_folder as search hint; use pp_input_folder as pseudo_dir string in &CONTROL and as destination folder name (relative to .in dir if not absolute)
        - Optionally copy required pseudo files into that pseudo_dir (next to .in) if 'copy_pseudos' is enabled
        - If nspin == 2 and starting_mags are non-zero, write starting_magnetization(i) for the first types (i=1..)
        """
        try:
            # select atoms to write (no supercell)
            if mode == "bulk":
                if self.bulk is None:
                    self.log_message("Error: No bulk structure loaded.")
                    return
                atoms = self.bulk
                kpoints = (self.kx.value(), self.ky.value(), self.kz.value())
            elif mode == "slab":
                if self.slab is None:
                    self.log_message("Error: No slab built.")
                    return
                atoms = self.slab
                if self.override_kpoints.isChecked():
                    kpoints = (self.kx.value(), self.ky.value(), self.kz.value())
                else:
                    kpoints = (1, 1, 1)
            else:
                self.log_message(f"Unknown mode: {mode}")
                return

            if atoms is None:
                self.log_message("Error: No atoms to write.")
                return

            # get UI pseudo folders
            pp_search_hint = self.pp_search_folder.text().strip()
            pp_input_user = self.pp_input_folder.text().strip() or "./pseudo"
            do_copy = self.copy_pseudos_checkbox.isChecked()

            # find pseudos (full paths)
            symbols = atoms.get_chemical_symbols()
            unique_symbols = []
            for s in symbols:
                if s not in unique_symbols:
                    unique_symbols.append(s)

            elem_to_full = self._find_pseudos_for_elements(unique_symbols, pp_search_hint)

            # ask save location for .in
            file_suggest = f"{self.prefix.text().strip() or 'qe_calc'}_{mode}.in"
            file, _ = QFileDialog.getSaveFileName(self, f"Save QE input ({mode})", file_suggest, "QE input (*.in);;All files (*)")
            if not file:
                self.log_message("Saving cancelled.")
                return
            in_dir = os.path.dirname(os.path.abspath(file))

            # determine destination pseudo folder path and behaviour
            if do_copy:
                # If copying: ensure destination directory exists (absolute or relative to in_dir)
                if os.path.isabs(pp_input_user):
                    pseudo_dest_dir = pp_input_user
                    os.makedirs(pseudo_dest_dir, exist_ok=True)
                    try:
                        rel = os.path.relpath(pseudo_dest_dir, in_dir)
                        pseudo_dir_for_in = ("./" + rel) if not rel.startswith("..") else pseudo_dest_dir
                    except Exception:
                        pseudo_dir_for_in = pseudo_dest_dir
                else:
                    pseudo_dest_dir = os.path.join(in_dir, pp_input_user)
                    os.makedirs(pseudo_dest_dir, exist_ok=True)
                    if pp_input_user.startswith("./") or pp_input_user.startswith("../"):
                        pseudo_dir_for_in = pp_input_user
                    else:
                        pseudo_dir_for_in = "./" + pp_input_user if not pp_input_user.startswith("/") else pp_input_user

                # copy pseudos into pseudo_dest_dir
                elem_to_basename = self._copy_pseudos_to_target(elem_to_full, pseudo_dest_dir)
            else:
                # If not copying: do NOT create dest dir or copy files.
                # For ATOMIC_SPECIES we will use basename of the found pseudo files,
                # and set pseudo_dir in .in to the user's pp_input_folder string (treated as before).
                if os.path.isabs(pp_input_user):
                    pseudo_dir_for_in = pp_input_user
                else:
                    # keep relative form like './pseudo'
                    if pp_input_user.startswith("./") or pp_input_user.startswith("../"):
                        pseudo_dir_for_in = pp_input_user
                    else:
                        pseudo_dir_for_in = "./" + pp_input_user if not pp_input_user.startswith("/") else pp_input_user
                elem_to_basename = {elem: os.path.basename(path) for elem, path in elem_to_full.items()}

            # build ATOMIC_SPECIES
            masses = atoms.get_masses()
            elem_masses = {}
            for i, s in enumerate(symbols):
                if s not in elem_masses:
                    elem_masses[s] = masses[i]

            atomic_species_lines = []
            for elem in unique_symbols:
                mass = elem_masses.get(elem, 0.0)
                basename = elem_to_basename.get(elem)
                if basename is None:
                    raise RuntimeError(f"Pseudopotential not found for element {elem}")
                atomic_species_lines.append(f"{elem} {mass:.6f} {basename}")

            # positions & cell
            pos_lines = []
            for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
                pos_lines.append(f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
            cell = atoms.get_cell()
            cell_lines = [f"{cell[i,0]:.8f} {cell[i,1]:.8f} {cell[i,2]:.8f}" for i in range(3)]

            # QE params
            calculation = self.calculation.currentText()
            ecutwfc = self.ecutwfc.value()
            ecutrho = self.ecutrho.value()
            prefix = self.prefix.text().strip() or "qe_calc"
            outdir = self.outdir.text().strip() or "./out"
            conv_thr = self.conv_thr.value()
            occupations = self.occupations.currentText()
            smearing = self.smearing.currentText()
            degauss = self.degauss.value()
            nspin = self.nspin.value()
            nbnd = self.nbnd.value()

            nat = len(symbols)
            ntyp = len(unique_symbols)

            # compose input text
            lines = []
            lines.append("&CONTROL")
            lines.append(f"  calculation = '{calculation}',")
            lines.append(f"  prefix = '{prefix}',")
            lines.append(f"  outdir = '{outdir}',")
            lines.append(f"  pseudo_dir = '{pseudo_dir_for_in}',")
            lines.append("/")
            lines.append("&SYSTEM")
            lines.append("  ibrav = 0,")
            lines.append(f"  nat = {nat},")
            lines.append(f"  ntyp = {ntyp},")
            lines.append(f"  ecutwfc = {ecutwfc},")
            lines.append(f"  ecutrho = {ecutrho},")
            if nspin > 1:
                lines.append(f"  nspin = {nspin},")
            if nbnd > 0:
                lines.append(f"  nbnd = {nbnd},")
            lines.append(f"  occupations = '{occupations}',")
            if occupations == "smearing":
                sm_map = {"gaussian": "gaussian", "methfessel-paxton": "methfessel-paxton", "marzari-vanderbilt": "marzari-vanderbilt"}
                sm = sm_map.get(smearing, "gaussian")
                lines.append(f"  smearing = '{sm}',")
                lines.append(f"  degauss = {degauss},")
            # starting_magnetization: only if nspin==2 and user provided non-zero for some species
            if nspin == 2:
                starting_vals = [self.start_mag_1.value(), self.start_mag_2.value(), self.start_mag_3.value()]
                for idx, val in enumerate(starting_vals):
                    if idx >= ntyp:
                        break
                    if abs(val) > 1e-12:
                        lines.append(f"  starting_magnetization({idx+1}) = {val},")
            lines.append("/")
            lines.append("&ELECTRONS")
            lines.append(f"  conv_thr = {conv_thr},")
            lines.append("/\n")

            lines.append("ATOMIC_SPECIES")
            lines += atomic_species_lines
            lines.append("\nCELL_PARAMETERS angstrom")
            lines += cell_lines
            lines.append("\nATOMIC_POSITIONS angstrom")
            lines += pos_lines

            kx, ky, kz = kpoints
            lines.append("\nK_POINTS automatic")
            lines.append(f"{kx} {ky} {kz} 0 0 0")

            input_text = "\n".join(lines)

            # write .in
            with open(file, "w") as f:
                f.write(input_text)

            self.log_message(f"QE input ({mode}) saved to: {file}")
            if do_copy:
                self.log_message(f"Pseudopotentials copied into: {pseudo_dest_dir}")
                self.log_message(f"pseudo_dir set in .in as: {pseudo_dir_for_in}")
            else:
                self.log_message("Pseudopotentials were NOT copied (checkbox unchecked).")
                self.log_message(f".in references pseudo_dir = {pseudo_dir_for_in} -- ensure pseudos exist there when running pw.x.")
            self.log_message("Run pw.x from the directory containing the .in file (so the relative pseudo_dir is found).")
        except Exception as e:
            self.log_message(f"[Error] generate_qe_input: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SlabApp()
    win.resize(1250, 820)
    win.show()
    sys.exit(app.exec())
