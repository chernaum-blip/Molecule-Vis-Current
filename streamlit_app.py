
import io
from typing import List, Tuple, Set
import streamlit as st
import py3Dmol
import urllib.request
import pandas as pd

# Optional speedup (falls back gracefully if missing)
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False


# Biopython
from Bio.PDB import PDBParser

st.set_page_config(page_title="Ligand H-bonds & Disulfides", layout="wide")

# ---------------- Sidebar UI ----------------
st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload a .pdb file", type=["pdb"])
pdb_id_default = st.sidebar.text_input("...or fetch by PDB ID", value="3PTB").strip()
cutoff = st.sidebar.slider("H-bond cutoff (Å)", 2.6, 5.0, 3.5, 0.1)
use_tube = st.sidebar.toggle("Use tube fallback", value=False)
show_disulfides = st.sidebar.toggle("Show disulfides", value=True)
run_btn = st.sidebar.button("Render / Recompute")

st.sidebar.markdown("### Move ligand")
tx = st.sidebar.slider("Translate X (Å)", -15.0, 15.0, 0.0, 0.1)
ty = st.sidebar.slider("Translate Y (Å)", -15.0, 15.0, 0.0, 0.1)
tz = st.sidebar.slider("Translate Z (Å)", -15.0, 15.0, 0.0, 0.1)
rx = st.sidebar.slider("Rotate X (°)", -180.0, 180.0, 0.0, 1.0)
ry = st.sidebar.slider("Rotate Y (°)", -180.0, 180.0, 0.0, 1.0)
rz = st.sidebar.slider("Rotate Z (°)", -180.0, 180.0, 0.0, 1.0)

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pdb_text(pdb_id: str) -> str:
    """Download PDB text directly from RCSB."""
    pdb_id = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")

import math

def split_pdb_protein_ligand(pdb_text: str) -> tuple[str, str]:
    """Return (protein_pdb, ligand_pdb) strings; ligand excludes waters."""
    prot = []
    lig = []
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM"):
            prot.append(ln)
        elif ln.startswith("HETATM"):
            resn = ln[17:20].strip().upper()
            if resn != "HOH":
                lig.append(ln)
    # 3Dmol likes explicit END
    return ("\n".join(prot) + "\nEND\n") if prot else "", ("\n".join(lig) + "\nEND\n") if lig else ""

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def compose_trs_matrix(tx: float, ty: float, tz: float, rx_deg: float, ry_deg: float, rz_deg: float):
    """
    Build a 4x4 column-major transform matrix matching 3Dmol.js expectations.
    Rotation order: Rz * Ry * Rx, then translate (tx, ty, tz) in Å.
    """
    rx, ry, rz = deg2rad(rx_deg), deg2rad(ry_deg), deg2rad(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rotation matrices (row-major here)
    Rx = [[1, 0, 0],
          [0, cx, -sx],
          [0, sx,  cx]]
    Ry = [[ cy, 0, sy],
          [  0, 1,  0],
          [-sy, 0, cy]]
    Rz = [[cz, -sz, 0],
          [sz,  cz, 0],
          [ 0,   0, 1]]

    def matmul3(A,B):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    R = matmul3(matmul3(Rz, Ry), Rx)

    # 4x4 column-major as list of 16 values (what 3Dmol expects)
    # [ r00 r10 r20 0  r01 r11 r21 0  r02 r12 r22 0  tx ty tz 1 ]
    m = [
        R[0][0], R[1][0], R[2][0], 0.0,
        R[0][1], R[1][1], R[2][1], 0.0,
        R[0][2], R[1][2], R[2][2], 0.0,
        tx,      ty,      tz,      1.0,
    ]
    return m

def parse_structure_atoms(pdb_text: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("STRUCT", io.StringIO(pdb_text))
    protein_atoms, ligand_atoms, cys_sg = [], [], []
    for model in structure:
        for chain in model:
            for res in chain:
                resn = res.get_resname().strip().upper()
                het = res.id[0].strip()
                if resn == "HOH":
                    continue
                if het == "":  # polymer / protein
                    for a in res:
                        protein_atoms.append(a)
                        if resn == "CYS" and a.get_id().upper() == "SG":
                            cys_sg.append(a)
                else:  # ligands & other het groups
                    for a in res:
                        ligand_atoms.append(a)
    return protein_atoms, ligand_atoms, cys_sg, structure


def elem_of(a):
    e = getattr(a, "element", "").strip()
    return e if e else a.get_id().strip()[:1]


def d2(a, b):
    v = a.get_vector() - b.get_vector()
    return v * v  # dot product (squared distance)


def bonded_residue_key(atom):
    res = atom.get_parent()
    return (res.get_parent().id.strip(), int(res.id[1]), res.get_resname().strip().upper())


def compute_hbonds_and_ss(pdb_text: str, cutoff_ang: float = 3.5):
    protein_atoms, ligand_atoms, cys_sg, structure = parse_structure_atoms(pdb_text)
    lig_NO = [a for a in ligand_atoms if elem_of(a) in ("N", "O")]
    prot_NOS = [a for a in protein_atoms if elem_of(a) in ("N", "O", "S")]

    hbonds: List[Tuple] = []
    bonded_residues: Set[Tuple[str, int, str]] = set()
    cut2 = cutoff_ang ** 2

    if HAVE_KDTREE and lig_NO and prot_NOS:
        # Build KD-tree on protein N/O/S atoms
        prot_xyz = [(float(a.coord[0]), float(a.coord[1]), float(a.coord[2])) for a in prot_NOS]
        tree = KDTree(prot_xyz)
        for la in lig_NO:
            q = (float(la.coord[0]), float(la.coord[1]), float(la.coord[2]))
            idxs = tree.query_ball_point(q, r=cutoff_ang)
            for i in idxs:
                pa = prot_NOS[i]
                if d2(la, pa) <= cut2:
                    hbonds.append((la, pa))
                    bonded_residues.add(bonded_residue_key(pa))
    else:
        for la in lig_NO:
            for pa in prot_NOS:
                if d2(la, pa) <= cut2:
                    hbonds.append((la, pa))
                    bonded_residues.add(bonded_residue_key(pa))

    # Disulfides (SG···SG within ~2.2 Å)
    ss_pairs = []
    ss2 = 2.2 ** 2
    for i in range(len(cys_sg)):
        for j in range(i + 1, len(cys_sg)):
            if d2(cys_sg[i], cys_sg[j]) <= ss2:
                ss_pairs.append((cys_sg[i], cys_sg[j]))

    return hbonds, bonded_residues, ss_pairs


def polymer_chains_from_pdb_text(pdb_text: str):
    chains, seen = [], set()
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM"):
            ch = ln[21].strip() or "A"
            if ch not in seen:
                seen.add(ch)
                chains.append(ch)
    return chains


def render_view(pdb_text: str, cutoff_ang: float = 3.5, use_tube=False, show_ss=True):
    hbonds, bonded_residues, ss_pairs = compute_hbonds_and_ss(pdb_text, cutoff_ang=cutoff_ang)

    chains = polymer_chains_from_pdb_text(pdb_text)
    view = py3Dmol.view(width=1000, height=700)
    view.setBackgroundColor("#111731")
    view.addModel(pdb_text, "pdb")

    if use_tube:
        for ch in (chains or ['A']):
            view.setStyle({"chain": ch}, {"tube": {"radius": 0.5, "color": "spectrum"}})
    else:
        for ch in (chains or ['A']):
            view.setStyle({"chain": ch}, {"cartoon": {"color": "spectrum"}})

    # Ligands (non-water) as sticks
    view.setStyle(
        {"hetflag": True, "not": {"resn": "HOH"}},
        {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}}
    )

    # H-bond lines
    for la, pa in hbonds:
        L = {"x": float(la.coord[0]), "y": float(la.coord[1]), "z": float(la.coord[2])}
        P = {"x": float(pa.coord[0]), "y": float(pa.coord[1]), "z": float(pa.coord[2])}
        view.addLine({"start": L, "end": P, "dashed": True, "color": "yellow", "linewidth": 2})

    # Disulfides
    if show_ss:
        for a, b in ss_pairs:
            A = {"x": float(a.coord[0]), "y": float(a.coord[1]), "z": float(a.coord[2])}
            B = {"x": float(b.coord[0]), "y": float(b.coord[1]), "z": float(b.coord[2])}
            view.addLine({"start": A, "end": B, "color": "green", "linewidth": 3})

    # Highlight & label bonded residues
    for ch, resi, resn in bonded_residues:
        view.setStyle({"chain": ch, "resi": int(resi)}, {"stick": {"radius": 0.3, "color": "magenta"}})
        view.addResLabels(
            {"chain": ch, "resi": int(resi)},
            {"fontColor": "white", "fontSize": 12, "backgroundOpacity": 0.6}
        )

    # Correct zoom: protein == not a HET
    view.zoomTo({"hetflag": False})
    return view, hbonds, bonded_residues, ss_pairs


# ---------------- Load input ----------------
if uploaded is not None:
    try:
        pdb_text = uploaded.read().decode("utf-8", errors="ignore")
        source = f"Uploaded file: {uploaded.name}"
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    try:
        pdb_id = pdb_id_default if pdb_id_default else "3PTB"
        pdb_text = fetch_pdb_text(pdb_id)
        source = f"RCSB PDB ID: {pdb_id.upper()}"
    except Exception as e:
        st.error(f"Could not download PDB {pdb_id_default}: {e}")
        st.stop()

# Trigger render automatically or on button
if (uploaded is not None) or run_btn or ("_ran_once" not in st.session_state):
    st.session_state["_ran_once"] = True

    with st.spinner(f"Rendering from {source} ..."):
        view, hbonds, bonded_residues, ss_pairs = render_view(
            pdb_text, cutoff_ang=cutoff, use_tube=use_tube, show_ss=show_disulfides
        )

    # Embed viewer
    html = view._make_html()
    st.components.v1.html(html, height=720, scrolling=False)

    # Right panel: data summary + CSV
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Hydrogen-bonded residues (protein side)")
        if bonded_residues:
            rows = [
                {"chain": ch, "resi": resi, "resn": resn}
                for (ch, resi, resn) in sorted(bonded_residues, key=lambda x: (x[0], x[1]))
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No H-bonded residues found at this cutoff. Try a larger cutoff.")

    with col2:
        st.subheader("H-bond pairs (atom–atom)")
        if hbonds:
            rows = []
            for la, pa in hbonds:
                dist = ((la.get_vector() - pa.get_vector()) * (la.get_vector() - pa.get_vector())) ** 0.5
                res = pa.get_parent()
                chain = res.get_parent().id.strip()
                rows.append({
                    "ligand_atom": la.get_id(),
                    "protein_atom": pa.get_id(),
                    "protein_resn": res.get_resname().strip().upper(),
                    "protein_resi": int(res.id[1]),
                    "protein_chain": chain,
                    "distance_A": float(dist),
                })
            st.dataframe(rows, hide_index=True, use_container_width=True)
            df = pd.DataFrame(rows)
            st.download_button(
                "Download H-bond table (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="hbonds.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No H-bonds detected at current cutoff.")

    if show_disulfides:
        st.caption(f"Disulfide pairs detected: {len(ss_pairs)}")

def render_view(pdb_text: str, cutoff_ang: float = 3.5, use_tube=False, show_ss=True,
                lig_tx=0.0, lig_ty=0.0, lig_tz=0.0, lig_rx=0.0, lig_ry=0.0, lig_rz=0.0):
    hbonds, bonded_residues, ss_pairs = compute_hbonds_and_ss(pdb_text, cutoff_ang=cutoff_ang)

    # Split into two PDBs so we can transform ligand independently
    prot_pdb, lig_pdb = split_pdb_protein_ligand(pdb_text)

    chains = polymer_chains_from_pdb_text(pdb_text)
    view = py3Dmol.view(width=1000, height=700)
    view.setBackgroundColor("#111731")

    # Add models in fixed order: 0 = protein, 1 = ligand (if present)
    if prot_pdb:
        view.addModel(prot_pdb, "pdb")     # model 0
    if lig_pdb:
        view.addModel(lig_pdb, "pdb")      # model 1

    # Styles
    if use_tube:
        for ch in (chains or ['A']):
            view.setStyle({"model": 0, "chain": ch}, {"tube": {"radius": 0.5, "color": "spectrum"}})
    else:
        for ch in (chains or ['A']):
            view.setStyle({"model": 0, "chain": ch}, {"cartoon": {"color": "spectrum"}})

    if lig_pdb:
        view.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.28}})

    # Apply ligand transform (translate/rotate) if there is a ligand
    if lig_pdb:
        M = compose_trs_matrix(lig_tx, lig_ty, lig_tz, lig_rx, lig_ry, lig_rz)
        # py3Dmol exposes the underlying 3Dmol Model via getModel(index)
        lig_model = view.getModel(1)
        lig_model.setMatrix(M)

    # H-bond lines
    for la, pa in hbonds:
        L = {"x": float(la.coord[0]), "y": float(la.coord[1]), "z": float(la.coord[2])}
        P = {"x": float(pa.coord[0]), "y": float(pa.coord[1]), "z": float(pa.coord[2])}
        view.addLine({"start": L, "end": P, "dashed": True, "color": "yellow", "linewidth": 2})

    # Disulfides
    if show_ss:
        for a, b in ss_pairs:
            A = {"x": float(a.coord[0]), "y": float(a.coord[1]), "z": float(a.coord[2])}
            B = {"x": float(b.coord[0]), "y": float(b.coord[1]), "z": float(b.coord[2])}
            view.addLine({"start": A, "end": B, "color": "green", "linewidth": 3})

    # Highlight & label bonded residues on the protein model
    for ch, resi, resn in bonded_residues:
        view.setStyle({"model": 0, "chain": ch, "resi": int(resi)},
                      {"stick": {"radius": 0.3, "color": "magenta"}})
        view.addResLabels({"model": 0, "chain": ch, "resi": int(resi)},
                          {"fontColor": "white", "fontSize": 12, "backgroundOpacity": 0.6})

    # Zoom to protein for context
    view.zoomTo({"model": 0})
    return view, hbonds, bonded_residues, ss_pairs


st.markdown("""---
**Tips** 
• Type a PDB ID on the left or upload a `.pdb` file. 
• Adjust the hydrogen-bond cutoff to explore more/less interactions. 
• Use *Use tube fallback* if cartoons don’t render on your device.
""")






