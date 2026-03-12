from chemcrow_lite.tools.pubchem import functional_groups


def test_functional_groups_detects_terminal_alkyne() -> None:
    result = functional_groups("C#Cc1ccc(Cl)cc1")
    assert "alkyne" in result["functional_groups"]
    assert "terminal alkyne" in result["functional_groups"]
