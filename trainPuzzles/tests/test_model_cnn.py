"""Pruebas de la arquitectura CNN (``trainPuzzles/model_cnn.py``)."""
import numpy as np
import torch

from game import SuperTateti
from model import SuperTatetiNet, _canonicalize
from trainPuzzles import train as T
from trainPuzzles.model_cnn import SuperTatetiCNN, _expand_macro
from trainPuzzles.puzzles import make_puzzle
from trainPuzzles.tests.test_puzzles import mate_in_1_position


def _state_and_mask_after_move():
    """Estado crudo tras una jugada: solo 9 movimientos válidos (tablero activo fijo)."""
    g = SuperTateti()
    g.make_move(4, 0)                       # activa el tablero 0
    return torch.from_numpy(g.get_state()).unsqueeze(0), torch.from_numpy(g.get_valid_moves()).unsqueeze(0)


def test_forward_shapes_y_mascara():
    state, mask = _state_and_mask_after_move()
    net = SuperTatetiCNN().eval()
    with torch.no_grad():
        policy, value = net(state, mask)

    assert policy.shape == (1, 81)
    assert value.shape == (1, 1)
    assert abs(float(policy.sum()) - 1.0) < 1e-4
    assert -1.0 <= float(value) <= 1.0
    # Los movimientos inválidos deben quedar en probabilidad ~0.
    assert float((policy * (1.0 - mask)).sum()) < 1e-6
    assert int((policy > 0).sum()) == int(mask.sum())


def test_forward_sin_mascara_es_finito():
    g = SuperTateti()
    state = torch.from_numpy(g.get_state()).unsqueeze(0)
    policy, value = SuperTatetiCNN().eval()(state)
    assert torch.isfinite(policy).all()
    assert torch.isfinite(value).all()


def test_expand_macro_alinea_bloques():
    planes = torch.zeros(1, 3, 3)
    planes[0, 1, 2] = 1.0                   # celda macro (fila 1, col 2)
    out = _expand_macro(planes)
    assert out.shape == (1, 9, 9)
    assert float(out[0, 3:6, 6:9].sum()) == 9.0
    assert float(out.sum()) == 9.0


def test_canal0_usa_la_misma_canonicalizacion_que_mlp():
    g = SuperTateti()
    g.make_move(4, 4)                       # turno impar: mueve O -> canonicalización invierte signos
    state = torch.from_numpy(g.get_state()).unsqueeze(0)

    feats = SuperTatetiCNN()._features(_canonicalize(state))
    esperado = _canonicalize(state)[:, 0:81].reshape(-1, 9, 9)
    assert torch.equal(feats[:, 0], esperado)


def test_presupuesto_de_parametros_media():
    n = sum(p.numel() for p in SuperTatetiCNN().parameters())
    assert 100_000 < n < 200_000            # config "Media" (~132k, comparable al MLP ~143k)


def test_gradiente_fluye_y_cambia_pesos():
    g = mate_in_1_position()
    p = make_puzzle(g, 1, [2 * 9 + 2])
    torch.manual_seed(0)
    net = SuperTatetiCNN()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    trainer = T.PuzzleTrainer(net, lr=1e-2, seed=0)
    loss, ploss, vloss = trainer.train_epoch([p, p, p, p], batch_size=2)
    assert all(np.isfinite(x) for x in (loss, ploss, vloss))
    assert any(not torch.equal(net.state_dict()[k], before[k]) for k in before)


def test_guardado_y_carga_roundtrip(tmp_path):
    net = SuperTatetiCNN()
    out = tmp_path / "cnn.pth"
    T.save_net(net, out)

    otro = SuperTatetiCNN()
    otro.load_state_dict(torch.load(out, weights_only=True, map_location="cpu"))
    assert set(otro.state_dict()) == set(net.state_dict())


def test_integracion_topk_con_tta():
    g = mate_in_1_position()
    all_moves = tuple(int(m) for m in g.get_valid_moves().nonzero()[0])
    p = make_puzzle(g, 1, all_moves)
    net = SuperTatetiCNN().eval()
    m = T.topk_accuracy(net, [p, p], ks=(1, 2, 3), tta=True)
    assert m[1] == 1.0                       # todas las jugadas son óptimas
    assert m[1] <= m[2] <= m[3]


def test_build_net_factory():
    assert isinstance(T.build_net("cnn"), SuperTatetiCNN)
    assert isinstance(T.build_net("mlp"), SuperTatetiNet)
    try:
        T.build_net("rnn")
    except ValueError:
        pass
    else:
        raise AssertionError("build_net debería rechazar arquitecturas desconocidas")


def test_load_net_ignora_init_incompatible(tmp_path):
    """Un init MLP no debe romper al cargar el CNN: se entrena desde cero."""
    mlp = SuperTatetiNet()
    out = tmp_path / "mlp.pth"
    T.save_net(mlp, out)
    net = T.load_net(out, arch="cnn")        # no debe lanzar
    assert isinstance(net, SuperTatetiCNN)
