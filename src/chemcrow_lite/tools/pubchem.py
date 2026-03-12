from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
USER_AGENT = "ChemCrow-Lite/0.1"
RDLogger.DisableLog("rdApp.error")


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def is_smiles(text: str) -> bool:
    try:
        molecule = Chem.MolFromSmiles(text, sanitize=False)
    except Exception:
        return False
    return molecule is not None


def canonicalize_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES string")
    return Chem.MolToSmiles(molecule, canonical=True)


def _compound_mode(query: str) -> str:
    return "smiles" if is_smiles(query) else "name"


def query_to_cid(query: str) -> int:
    mode = _compound_mode(query)
    encoded = quote(query, safe="")
    payload = _get_json(f"{PUBCHEM_BASE}/pug/compound/{mode}/{encoded}/cids/JSON")
    return int(payload["IdentifierList"]["CID"][0])


def name_to_smiles(query: str) -> dict[str, Any]:
    if is_smiles(query):
        return {"query": query, "smiles": canonicalize_smiles(query), "source": "input"}

    encoded = quote(query, safe="")
    url = f"{PUBCHEM_BASE}/pug/compound/name/{encoded}/property/IsomericSMILES/JSON"
    payload = _get_json(url)
    properties = payload["PropertyTable"]["Properties"][0]
    smiles = properties.get("IsomericSMILES") or properties.get("SMILES")
    if not smiles:
        raise ValueError("PubChem did not return a SMILES field.")
    return {"query": query, "smiles": canonicalize_smiles(smiles), "source": "pubchem"}


def query_to_cas(query: str) -> dict[str, Any]:
    cid = query_to_cid(query)
    payload = _get_json(f"{PUBCHEM_BASE}/pug_view/data/compound/{cid}/JSON")

    record_sections = payload.get("Record", {}).get("Section", [])
    for section in record_sections:
        if section.get("TOCHeading") != "Names and Identifiers":
            continue
        for subsection in section.get("Section", []):
            if subsection.get("TOCHeading") != "Other Identifiers":
                continue
            for item in subsection.get("Section", []):
                if item.get("TOCHeading") != "CAS":
                    continue
                cas = item["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
                return {"query": query, "cid": cid, "cas": cas}
    raise ValueError("CAS number not found in PubChem.")


def smiles_to_name(smiles: str) -> dict[str, Any]:
    canonical = canonicalize_smiles(smiles)
    encoded = quote(canonical, safe="")
    payload = _get_json(f"{PUBCHEM_BASE}/pug/compound/smiles/{encoded}/synonyms/JSON")
    names = payload["InformationList"]["Information"][0]["Synonym"]
    cas_pattern = re.compile(r"^\d{2,7}-\d{2}-\d$")
    for name in names:
        if not cas_pattern.match(name):
            return {"smiles": canonical, "name": name}
    raise ValueError("No non-CAS synonym found.")


def smiles_to_weight(smiles: str) -> dict[str, Any]:
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError("Invalid SMILES string")
    return {
        "smiles": canonical,
        "molecular_weight": round(float(Descriptors.ExactMolWt(molecule)), 4),
    }


FUNCTIONAL_GROUPS = {
    "alcohol": "[OX2H]",
    "phenol": "c[OX2H]",
    "amide": "C(=O)N",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic acid": "C(=O)[OH]",
    "nitro": "[N+](=O)[O-]",
    "nitrile": "C#N",
    "alkyne": "C#C",
    "terminal alkyne": "[CX2]#[CX2H1]",
    "halogen": "[F,Cl,Br,I]",
    "thiourea": "NC(=S)N",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
}


def functional_groups(smiles: str) -> dict[str, Any]:
    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError("Invalid SMILES string")

    detected: list[str] = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and molecule.HasSubstructMatch(pattern):
            detected.append(name)
    return {"smiles": canonical, "functional_groups": detected}
